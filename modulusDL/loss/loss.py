"""
define losses
"""

from typing import Optional, Callable, List, Dict, Union, Tuple
from modulus.loss import Loss
from torch import Tensor
class PointwiseMinimiseLoss(Loss):
    """
    L-p loss function for pointwise data
    Computes the p-th order loss of each output tensor

    Parameters
    ----------
    ord : int
        Order of the loss. For example, `ord=2` would be the L2 loss.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _loss(
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        losses = {}
        for key, value in pred_outvar.items():
            l = lambda_weighting[key] * pred_outvar[key]
            if "area" in invar.keys():
                l *= invar["area"]
            losses[key] = l.sum()
        return losses

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        return PointwiseMinimiseLoss._loss(
            invar, pred_outvar, true_outvar, lambda_weighting, step
        )
class PointwiseSqMeanLoss(Loss):
    """
    L-p loss function for pointwise data
    Computes the p-th order loss of each output tensor

    Parameters
    ----------
    ord : int
        Order of the loss. For example, `ord=2` would be the L2 loss.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _loss(
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        losses = {}
        for key,value in pred_outvar.items():
            # print(key)
            # print(pred_outvar[key])
            l = lambda_weighting[key] * (pred_outvar[key] - true_outvar[key])**2.
            losses[key] = l.sum()
        
        return losses

    def forward(
        self,
        invar: Dict[str, Tensor],
        pred_outvar: Dict[str, Tensor],
        true_outvar: Dict[str, Tensor],
        lambda_weighting: Dict[str, Tensor],
        step: int,
    ) -> Dict[str, Tensor]:
        return PointwiseSqMeanLoss._loss(
            invar, pred_outvar, true_outvar, lambda_weighting, step
        )