import os
import numpy as np
from typing import Dict, List, Union, Tuple, Callable, Optional
from omegaconf import DictConfig
import warnings

from modulus.solver import Solver
from modulus.domain import Domain
from modulus.loss.aggregator import NTK, Sum


import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler
import torch.cuda.profiler as profiler
import torch.distributed as dist
from contextlib import ExitStack

from modulus.utils.training.stop_criterion import StopCriterion
from modulus.constants import TF_SUMMARY, JIT_PYTORCH_VERSION
from modulus.hydra import (
    instantiate_optim,
    instantiate_sched,
    instantiate_agg,
    add_hydra_run_path,
)
from modulus.distributed.manager import DistributedManager
from omegaconf import DictConfig, OmegaConf


class Solver_NoRecordConstraints(Solver):
    '''
    This class is adapted from NVIDIAModulus v22.09 solver.Solver
    to allow user to disable recording of constraints for setup which have intermediate variables not suitable for recording
    '''
    def __init__(self, cfg: DictConfig,
                 domain: Domain):
        super(Solver_NoRecordConstraints, self).__init__(cfg,domain)
    def record_constraints(self):
        pass
# base class for solver
class Solver_ReduceLROnPlateauLoss(Solver):
    """
    This class is adapted from NVIDIAModulus v22.09 solver.Solver
    to allow the use of pytorch ReduceLROnPlateauLoss

    Parameters
    ----------
    cfg : DictConfig
        Hydra dictionary of configs.
    domain : Domain
        Domain to solve for.
    """

    def __init__(self, cfg: DictConfig,
                 domain: Domain,
                 ReduceLROnPlateau_DictConFig=None,
                 batch_per_epoch=1000,
                 use_moving_average=True,
                 remove_record_constraints=False):
        super(Solver_ReduceLROnPlateauLoss, self).__init__(cfg,domain)
        if ReduceLROnPlateau_DictConFig is None:
            self.ReduceLROnPlateau_DictConFig={}
        else:
            self.ReduceLROnPlateau_DictConFig=ReduceLROnPlateau_DictConFig
            
        if "patience" not in self.ReduceLROnPlateau_DictConFig:
            
            if use_moving_average:
                self.ReduceLROnPlateau_DictConFig["patience"]=int(batch_per_epoch/10)
            else:
                self.ReduceLROnPlateau_DictConFig["patience"]=batch_per_epoch
        self.invPatience=1./self.ReduceLROnPlateau_DictConFig["patience"]
        if "factor" not in self.ReduceLROnPlateau_DictConFig:
            self.ReduceLROnPlateau_DictConFig["factor"]=0.9
        if "threshold" not in self.ReduceLROnPlateau_DictConFig:
            self.ReduceLROnPlateau_DictConFig["threshold"]=1e-4
        #correcting threshold
        if use_moving_average:
            self.ReduceLROnPlateau_DictConFig["threshold"]=self.ReduceLROnPlateau_DictConFig["threshold"]*self.invPatience
        
        if "threshold_mode" not in self.ReduceLROnPlateau_DictConFig:
            self.ReduceLROnPlateau_DictConFig["threshold_mode"]='rel'
        if "cooldown" not in self.ReduceLROnPlateau_DictConFig:
            self.ReduceLROnPlateau_DictConFig["cooldown"]=10
        if "verbose" not in self.ReduceLROnPlateau_DictConFig:
            self.ReduceLROnPlateau_DictConFig["verbose"]=True
        self.movingAverageLoss=None
        self.use_moving_average=use_moving_average
        if remove_record_constraints:
            def record_constraints(self):
                pass
            self.record_constraints=record_constraints
    def _train_loop(
        self,
        sigterm_handler=None,
    ):  # TODO this train loop may be broken up into methods if need for future children classes

        # make directory if doesn't exist
        if self.manager.rank == 0:
            # exist_ok=True to skip creating directory that already exists
            os.makedirs(self.network_dir, exist_ok=True)

        # create global model for restoring and saving
        self.saveable_models = self.get_saveable_models()
        self.global_optimizer_model = self.create_global_optimizer_model()

        # initialize optimizer from hydra
        self.compute_gradients = getattr(
            self, self.cfg.optimizer._params_.compute_gradients
        )
        self.apply_gradients = getattr(
            self, self.cfg.optimizer._params_.apply_gradients
        )
        self.optimizer = instantiate_optim(self.cfg, model=self.global_optimizer_model)

        # initialize scheduler from hydra
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.ReduceLROnPlateau_DictConFig)

        # initialize aggregator from hydra
        self.aggregator = instantiate_agg(
            self.cfg,
            model=self.global_optimizer_model.parameters(),
            num_losses=self.get_num_losses(),
        )

        if self.cfg.jit:
            # Warn user if pytorch version difference
            if not torch.__version__ == JIT_PYTORCH_VERSION:
                self.log.warn(
                    f"Installed PyTorch version {torch.__version__} is not TorchScript"
                    + f" supported in Modulus. Version {JIT_PYTORCH_VERSION} is officially supported."
                )

            self.aggregator = torch.jit.script(self.aggregator)
            if self.amp:
                torch._C._jit_set_autocast_mode(True)

        if len(list(self.aggregator.parameters())) > 0:
            self.log.debug("Adding loss aggregator param group. LBFGS will not work!")
            self.optimizer.add_param_group(
                {"params": list(self.aggregator.parameters())}
            )

        # create grad scalar for AMP
        # grad scaler is only available for float16 dtype on cuda device
        enable_scaler = self.amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler(enabled=enable_scaler)

        # make stop criterion
        if self.stop_criterion_metric is not None:
            self.stop_criterion = StopCriterion(
                self.stop_criterion_metric,
                self.stop_criterion_min_delta,
                self.stop_criterion_patience,
                self.stop_criterion_mode,
                self.stop_criterion_freq,
                self.stop_criterion_strict,
                self.cfg.training.rec_monitor_freq,
                self.cfg.training.rec_validation_freq,
            )

        # load network
        self.initial_step = self.load_network()

        # # make summary writer
        self.writer = SummaryWriter(
            log_dir=self.network_dir, purge_step=self.summary_freq + 1
        )
        self.summary_histograms = self.cfg["summary_histograms"]

        # write tensorboard config
        if self.manager.rank == 0:
            self.writer.add_text(
                "config", f"<pre>{str(OmegaConf.to_yaml(self.cfg))}</pre>"
            )

        # create profiler
        try:
            self.profile = self.cfg.profiler.profile
            self.profiler_start_step = self.cfg.profiler.start_step
            self.profiler_end_step = self.cfg.profiler.end_step
            if self.profiler_end_step < self.profiler_start_step:
                self.profile = False
        except:
            self.profile = False
            self.profiler_start_step = -1
            self.profiler_end_step = -1

        # Distributed barrier before starting the train loop
        if self.manager.distributed:
            dist.barrier(device_ids=[self.manager.local_rank])
        barrier_flag = False

        if self.manager.cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            t = time.time()

        # termination signal handler
        if sigterm_handler is None:
            self.sigterm_handler = lambda: False
        else:
            self.sigterm_handler = sigterm_handler

        # train loop
        with ExitStack() as stack:
            if self.profile:
                # Add NVTX context if in profile mode
                self.log.warning("Running in profiling mode")
                stack.enter_context(torch.autograd.profiler.emit_nvtx())

            for step in range(self.initial_step, self.max_steps + 1):

                if self.sigterm_handler():
                    if self.manager.rank == 0:
                        self.log.info(
                            f"Training terminated by the user at iteration {step}"
                        )
                    break

                if self.profile and step == self.profiler_start_step:
                    # Start profiling
                    self.log.info("Starting profiler at step {}".format(step))
                    profiler.start()

                if self.profile and step == self.profiler_end_step:
                    # Stop profiling
                    self.log.info("Stopping profiler at step {}".format(step))
                    profiler.stop()

                torch.cuda.nvtx.range_push("Training iteration")

                if self.cfg.cuda_graphs:
                    # If cuda graphs statically load it into defined allocations
                    self.load_data(static=True)

                    loss, losses = self._cuda_graph_training_step(step)
                else:
                    # Load all data for constraints
                    self.load_data()

                    self.global_optimizer_model.zero_grad(set_to_none=True)

                    # compute gradients
                    loss, losses = self.compute_gradients(
                        self.aggregator, self.global_optimizer_model, step
                    )

                    # take optimizer step
                    self.apply_gradients()

                    # take scheduler step
                    if self.movingAverageLoss is None or not(self.use_moving_average):
                        self.movingAverageLoss=float(loss)
                    else:
                        self.movingAverageLoss=self.movingAverageLoss*(1.-self.invPatience)+self.invPatience*float(loss)
                    self.scheduler.step(self.movingAverageLoss)

                # check for nans in loss
                if torch.isnan(loss):
                    self.log.error("loss went to Nans")
                    break

                self.step_str = f"[step: {step:10d}]"

                # write train loss / learning rate tensorboard summaries
                if step % self.summary_freq == 0:
                    if self.manager.rank == 0:

                        # add train loss scalars
                        for key, value in losses.items():
                            if TF_SUMMARY:
                                self.writer.add_scalar(
                                    "Train_/loss_L2" + str(key),
                                    value,
                                    step,
                                    new_style=True,
                                )
                            else:
                                self.writer.add_scalar(
                                    "Train/loss_" + str(key),
                                    value,
                                    step,
                                    new_style=True,
                                )
                        if TF_SUMMARY:
                            self.writer.add_scalar(
                                "Optimzer/loss", loss, step, new_style=True
                            )
                            self.writer.add_scalar(
                                "learning_rate/lr",
                                self.scheduler.get_last_lr()[0],  # TODO: handle list
                                step,
                                new_style=True,
                            )
                        else:
                            self.writer.add_scalar(
                                "Train/loss_aggregated", loss, step, new_style=True
                            )
                            self.writer.add_scalar(
                                "Train/learning_rate",
                                self.scheduler._last_lr[0],  # TODO: handle list
                                step,
                                new_style=True,
                            )

                    if self.manager.distributed:
                        barrier_flag = True

                # write train / inference / validation datasets to tensorboard and file
                if step % self.cfg.training.rec_constraint_freq == 0:
                    barrier_flag = True
                    self._record_constraints()

                if (step % self.cfg.training.rec_validation_freq == 0) and (
                    self.has_validators
                ):
                    barrier_flag = True
                    self._record_validators(step)

                if (step % self.cfg.training.rec_inference_freq == 0) and (
                    self.has_inferencers
                ):
                    barrier_flag = True
                    self._record_inferencers(step)

                if (step % self.cfg.training.rec_monitor_freq == 0) and (
                    self.has_monitors
                ):
                    barrier_flag = True
                    self._record_monitors(step)

                # save checkpoint
                if step % self.save_network_freq == 0:
                    # Get data parallel rank so all processes in the first model parallel group
                    # can save their checkpoint. In the case without model parallelism, data_parallel_rank
                    # should be the same as the process rank itself
                    data_parallel_rank = (
                        self.manager.group_rank("data_parallel")
                        if self.manager.distributed
                        else 0
                    )
                    if data_parallel_rank == 0:
                        self.save_checkpoint(step)
                        self.log.info(
                            f"{self.step_str} saved checkpoint to {add_hydra_run_path(self.network_dir)}"
                        )
                    if self.manager.distributed:
                        barrier_flag = True

                if self.manager.distributed and barrier_flag:
                    dist.barrier(device_ids=[self.manager.local_rank])
                    barrier_flag = False

                # print loss stats
                if step % self.print_stats_freq == 0:
                    # synchronize and get end time
                    if self.manager.cuda:
                        end_event.record()
                        end_event.synchronize()
                        elapsed_time = start_event.elapsed_time(
                            end_event
                        )  # in milliseconds
                    else:
                        t_end = time.time()
                        elapsed_time = (t_end - t) * 1.0e3  # in milliseconds

                    # Reduce loss across all GPUs
                    if self.manager.distributed:
                        dist.reduce(loss, 0, op=dist.ReduceOp.AVG)
                        elapsed_time = torch.tensor(elapsed_time).to(self.device)
                        dist.reduce(elapsed_time, 0, op=dist.ReduceOp.AVG)
                        elapsed_time = elapsed_time.cpu().numpy()[()]

                    # print statement
                    print_statement = (
                        f"{self.step_str} loss: {loss.cpu().detach().numpy():10.3e}"
                    )
                    if step >= self.initial_step + self.print_stats_freq:
                        print_statement += f", time/iteration: {elapsed_time/self.print_stats_freq:10.3e} ms"
                    if self.manager.rank == 0:
                        self.log.info(print_statement)

                    if self.manager.cuda:
                        start_event.record()
                    else:
                        t = time.time()

                # check stopping criterion
                stop_training = self._check_stopping_criterion(loss, losses, step)
                if stop_training:
                    if self.manager.rank == 0:
                        self.log.info(
                            f"{self.step_str} stopping criterion is met, finished training!"
                        )
                    break

                # check max steps
                if step >= self.max_steps:
                    if self.manager.rank == 0:
                        self.log.info(
                            f"{self.step_str} reached maximum training steps, finished training!"
                        )
                    break

                torch.cuda.nvtx.range_pop()
    def _cuda_graph_training_step(self, step: int):
        # Training step method for using cuda graphs
        # Warm up
        if (step - self.initial_step) < self.cfg.cuda_graph_warmup:
            if (step - self.initial_step) == 0:
                # Default stream for warm up
                self.warmup_stream = torch.cuda.Stream()

            self.warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.warmup_stream):
                # zero optimizer gradients
                self.global_optimizer_model.zero_grad(set_to_none=True)

                # # compute gradients
                self.loss_static, self.losses_static = self.compute_gradients(
                    self.aggregator, self.global_optimizer_model, step
                )
            torch.cuda.current_stream().wait_stream(self.warmup_stream)

            # take optimizer step
            self.apply_gradients()

            # take scheduler step
            if self.movingAverageLoss is None or not(self.use_moving_average):
                self.movingAverageLoss=float(self.loss_static)
            else:
                self.movingAverageLoss=self.movingAverageLoss*(1.-self.invPatience)+self.invPatience*float(self.loss_static)
            self.scheduler.step(self.movingAverageLoss)
        # Record graph
        elif (step - self.initial_step) == self.cfg.cuda_graph_warmup:
            torch.cuda.synchronize()
            if self.manager.distributed:
                dist.barrier(device_ids=[self.manager.local_rank])

            if self.cfg.cuda_graph_warmup < 11:
                self.log.warn(
                    f"Graph warm up length ({self.cfg.cuda_graph_warmup}) should be more than 11 steps, higher suggested"
                )
            self.log.info("Attempting cuda graph building, this may take a bit...")

            self.g = torch.cuda.CUDAGraph()
            self.global_optimizer_model.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.g):
                # compute gradients
                self.loss_static, self.losses_static = self.compute_gradients(
                    self.aggregator, self.global_optimizer_model, step
                )

            # take optimizer step
            # left out of graph for AMP compat, No perf difference
            self.apply_gradients()

            # take scheduler step
            if self.movingAverageLoss is None or not(self.use_moving_average):
                self.movingAverageLoss=float(self.loss_static)
            else:
                self.movingAverageLoss=self.movingAverageLoss*(1.-self.invPatience)+self.invPatience*float(self.loss_static)
            self.scheduler.step(self.movingAverageLoss)
        # Replay
        else:
            # Graph replay
            self.g.replay()
            # take optimizer step
            self.apply_gradients()

            if self.movingAverageLoss is None or not(self.use_moving_average):
                self.movingAverageLoss=float(self.loss_static)
            else:
                self.movingAverageLoss=self.movingAverageLoss*(1.-self.invPatience)+self.invPatience*float(self.loss_static)
            self.scheduler.step(self.movingAverageLoss)

        return self.loss_static, self.losses_static

