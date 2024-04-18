'''
--------------------------------
    MODULUS MOTION TRACKING
--------------------------------

History:
    Date            Programmer      Description
    ----------      ----------      ----------------------------
    29/06/2023      S MU            v1.0.0 
'''
_version='1.0.0'
import os
import sys
sys.path.insert(0,"/examples")
#os.environ["CUDA_VISIBLE_DEVICES"]="2" 
import modulus
import numpy as np
from sympy import (
    Symbol,
    Function,
    Eq,
    Number,
    Abs,
    Max,
    Min,
    sqrt,
    pi,
    sin,
    cos,
    atan,
    atan2,
    acos,
    asin,
    sign,
    exp,
)
import pickle
from sympy import Symbol
from modulus.hydra import ModulusConfig
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry.parameterization import Parameterization
from modulusDL.models.arch import (
    CustomModuleArch, 
    InheritedInvertibleFullyConnectedArchCore,
    InheritedFullyConnectedWeightsGen,
    CustomZeroInputModuleArch,
    CustomTripleInputModuleArch,
    SubtractionArch)
from modulusDL.eq.pde import (
    Fmat,
    )
from modulusDL.solver.solver import Solver_NoRecordConstraints
from modulus.domain.constraint import PointwiseConstraint, PointwiseInteriorConstraint
from modulus.domain.inferencer import PointwiseInferencer, PointVTKInferencer
from modulus.domain.validator import PointwiseValidator
from modulus.key import Key
from modulus import quantity
from modulus.eq.non_dim import NonDimensionalizer
from modulus.utils.io.vtk import VTKPolyData
from modulus.geometry.tessellation import Tessellation
import torch
from motionSegmentation.BsplineFourier import Bspline, BsplineFourier
try:
    import trimesh
except:
    pass
class getuvw(torch.nn.Module):
    def __init__(self,omega,Nfourier):
        super().__init__()
        torchrangeomega=torch.tensor(omega*np.repeat(np.arange(1,Nfourier+1),3).reshape((1,-1)),dtype=torch.float)
        self.register_buffer("torchrangeomega", torchrangeomega, persistent=False)
        self.Nfourier=int(Nfourier)
        self.Nfourier2=int(Nfourier*2)
        self.Nfourier3=int(Nfourier*3)
    def forward(self,x):#Fcossin1,t,deltaT
        a0radian=x[...,-2:-1]*self.torchrangeomega
        a0_uvw=(torch.cat((torch.cos(a0radian),torch.sin(a0radian)),dim=-1)*x[...,:-2]).reshape(x.size(0),self.Nfourier2,3).sum(dim=-2)
        fwdradian=(x[...,-1:]+x[...,-2:-1])*self.torchrangeomega
        fwd_uvw=(torch.cat((torch.cos(fwdradian),torch.sin(fwdradian)),dim=-1)*x[...,:-2]).reshape(x.size(0),self.Nfourier2,3).sum(dim=-2)
        return fwd_uvw-a0_uvw
def extract_deform(b_coef_file_format,time_index_x,time_index_y,length_scale_in_mm,fileScale,weight=None,sampleratio=5.,savepath=None,separate=False,coords_input=None):
    if separate:
        invar_dict=[]
        outvar_dict=[]
    else:
        invar_dict={'x':np.zeros((0,1)),'y':np.zeros((0,1)),'z':np.zeros((0,1)),'t':np.zeros((0,1)),'Delta_t':np.zeros((0,1)),'weightsqrt':np.zeros((0,1))}
        outvar_dict={'u':np.zeros((0,1)),'v':np.zeros((0,1)),'w':np.zeros((0,1))}
    for n in range(len(time_index_x)):
        print(str(n)+" : from "+str(time_index_x[n])+" to "+str(time_index_y[n]))
        bspl=Bspline(coefFile=b_coef_file_format.format(time_index_x[n],time_index_y[n]),timeMap=[time_index_x[n],time_index_y[n]],fileScale=fileScale)
        origin_coord=bspl.origin
        spacing_coord=bspl.spacing
        Bcoef_shape=bspl.coef.shape
        if coords_input is None:
            coordsList=np.mgrid[origin_coord[0]:(origin_coord[0]+spacing_coord[0]*(Bcoef_shape[0]+0.01)):spacing_coord[0]/sampleratio,origin_coord[1]:(origin_coord[1]+spacing_coord[1]*(Bcoef_shape[1]+0.5)):spacing_coord[1]/sampleratio,origin_coord[2]:(origin_coord[2]+spacing_coord[2]*(Bcoef_shape[2]+0.5)):spacing_coord[2]/sampleratio].T.reshape((-1,3))
        else:
            coordsList=coords_input
        uvw=bspl.getVector(coordsList)
        if separate:
            invar_dict.append({})
            outvar_dict.append({})
            invar_dict[-1]['x']=coordsList[:,0:1]
            invar_dict[-1]['y']=coordsList[:,1:2] 
            invar_dict[-1]['z']=coordsList[:,2:3] 
            invar_dict[-1]['t']=coordsList[:,0:1]*0.+time_index_x[n] 
            invar_dict[-1]['Delta_t']=coordsList[:,0:1]*0.+time_index_y[n]-time_index_x[n] 
            outvar_dict[-1]['u']=uvw[:,0:1] 
            outvar_dict[-1]['v']=uvw[:,1:2] 
            outvar_dict[-1]['w']=uvw[:,2:3]
            if weight is None:
                invar_dict[-1]['weightsqrt']=coordsList[:,0:1]*0.+1.
                outvar_dict[-1]['elastix_u']=uvw[:,0:1] 
                outvar_dict[-1]['elastix_v']=uvw[:,1:2] 
                outvar_dict[-1]['elastix_w']=uvw[:,2:3]
            else:
                invar_dict[-1]['weightsqrt']=coordsList[:,0:1]*0.+weight[n]
                outvar_dict[-1]['elastix_u']=uvw[:,0:1] * weight[n]
                outvar_dict[-1]['elastix_v']=uvw[:,1:2] * weight[n]
                outvar_dict[-1]['elastix_w']=uvw[:,2:3] * weight[n]
        else:
            invar_dict['x']=np.concatenate((invar_dict['x'],coordsList[:,0:1]),axis=0)
            invar_dict['y']=np.concatenate((invar_dict['y'],coordsList[:,1:2]),axis=0)
            invar_dict['z']=np.concatenate((invar_dict['z'],coordsList[:,2:3]),axis=0)
            invar_dict['t']=np.concatenate((invar_dict['t'],coordsList[:,0:1]*0.+time_index_x[n]),axis=0)
            invar_dict['Delta_t']=np.concatenate((invar_dict['Delta_t'],coordsList[:,0:1]*0.+time_index_y[n]-time_index_x[n]),axis=0)
            outvar_dict['u']=np.concatenate((outvar_dict['u'],uvw[:,0:1]),axis=0)
            outvar_dict['v']=np.concatenate((outvar_dict['v'],uvw[:,1:2]),axis=0)
            outvar_dict['w']=np.concatenate((outvar_dict['w'],uvw[:,2:3]),axis=0)
            if weight is None:
                invar_dict['weightsqrt']=np.concatenate((invar_dict['weightsqrt'],coordsList[:,0:1]*0.+1.),axis=0)
                outvar_dict['elastix_u']=np.concatenate((outvar_dict['u'],uvw[:,0:1]),axis=0)
                outvar_dict['elastix_v']=np.concatenate((outvar_dict['v'],uvw[:,1:2]),axis=0)
                outvar_dict['elastix_w']=np.concatenate((outvar_dict['w'],uvw[:,2:3]),axis=0)
            else:
                invar_dict['weightsqrt']=np.concatenate((invar_dict['weightsqrt'],coordsList[:,0:1]*0.+weight[n]),axis=0)
                outvar_dict['elastix_u']=np.concatenate((outvar_dict['u'],uvw[:,0:1]),axis=0) * weight[n]
                outvar_dict['elastix_v']=np.concatenate((outvar_dict['v'],uvw[:,1:2]),axis=0) * weight[n]
                outvar_dict['elastix_w']=np.concatenate((outvar_dict['w'],uvw[:,2:3]),axis=0) * weight[n]
        if savepath is not None:
            np.savetxt(savepath+'t'+str(int(time_index_x[n]))+"to"+str(int(time_index_y[n]))+'.txt',np.concatenate((np.arange(coordsList.shape[0]).reshape((-1,1)),coordsList[:,0:3],uvw[:,0:3]),axis=-1),comments='#pointID ',header='x y z u v w')
    return invar_dict,outvar_dict
def get_BSF_deform(bsf_coef_file_format,time_index_x,time_index_y,length_scale_in_mm,fileScale,weight=None,sampleratio=5.,savepath=None,separate=False,coords_input=None):
    if separate:
        invar_dict=[]
        outvar_dict=[]
    else:
        invar_dict={'x':np.zeros((0,1)),'y':np.zeros((0,1)),'z':np.zeros((0,1)),'t':np.zeros((0,1)),'Delta_t':np.zeros((0,1)),'weightsqrt':np.zeros((0,1))}
        outvar_dict={'u':np.zeros((0,1)),'v':np.zeros((0,1)),'w':np.zeros((0,1))}
    for n in range(len(time_index_x)):
        print(str(n)+" : from "+str(time_index_x[n])+" to "+str(time_index_y[n]))
        bspl=Bspline(coefFile=bsf_coef_file_format.format(time_index_x[n],time_index_y[n]),timeMap=[time_index_x[n],time_index_y[n]],fileScale=fileScale)
        origin_coord=bspl.origin
        spacing_coord=bspl.spacing
        Bcoef_shape=bspl.coef.shape
        if coords_input is None:
            coordsList=np.mgrid[origin_coord[0]:(origin_coord[0]+spacing_coord[0]*(Bcoef_shape[0]+0.01)):spacing_coord[0]/sampleratio,origin_coord[1]:(origin_coord[1]+spacing_coord[1]*(Bcoef_shape[1]+0.5)):spacing_coord[1]/sampleratio,origin_coord[2]:(origin_coord[2]+spacing_coord[2]*(Bcoef_shape[2]+0.5)):spacing_coord[2]/sampleratio].T.reshape((-1,3))
        else:
            coordsList=coords_input
        uvw=bspl.getVector(coordsList)
        if separate:
            invar_dict.append({})
            outvar_dict.append({})
            invar_dict[-1]['x']=coordsList[:,0:1]
            invar_dict[-1]['y']=coordsList[:,1:2] 
            invar_dict[-1]['z']=coordsList[:,2:3] 
            invar_dict[-1]['t']=coordsList[:,0:1]*0.+time_index_x[n] 
            invar_dict[-1]['Delta_t']=coordsList[:,0:1]*0.+time_index_y[n]-time_index_x[n] 
            outvar_dict[-1]['u']=uvw[:,0:1] 
            outvar_dict[-1]['v']=uvw[:,1:2] 
            outvar_dict[-1]['w']=uvw[:,2:3]
            if weight is None:
                invar_dict[-1]['weightsqrt']=coordsList[:,0:1]*0.+1.
                outvar_dict[-1]['elastix_u']=uvw[:,0:1] 
                outvar_dict[-1]['elastix_v']=uvw[:,1:2] 
                outvar_dict[-1]['elastix_w']=uvw[:,2:3]
            else:
                invar_dict[-1]['weightsqrt']=coordsList[:,0:1]*0.+weight[n]
                outvar_dict[-1]['elastix_u']=uvw[:,0:1] * weight[n]
                outvar_dict[-1]['elastix_v']=uvw[:,1:2] * weight[n]
                outvar_dict[-1]['elastix_w']=uvw[:,2:3] * weight[n]
        else:
            invar_dict['x']=np.concatenate((invar_dict['x'],coordsList[:,0:1]),axis=0)
            invar_dict['y']=np.concatenate((invar_dict['y'],coordsList[:,1:2]),axis=0)
            invar_dict['z']=np.concatenate((invar_dict['z'],coordsList[:,2:3]),axis=0)
            invar_dict['t']=np.concatenate((invar_dict['t'],coordsList[:,0:1]*0.+time_index_x[n]),axis=0)
            invar_dict['Delta_t']=np.concatenate((invar_dict['Delta_t'],coordsList[:,0:1]*0.+time_index_y[n]-time_index_x[n]),axis=0)
            outvar_dict['u']=np.concatenate((outvar_dict['u'],uvw[:,0:1]),axis=0)
            outvar_dict['v']=np.concatenate((outvar_dict['v'],uvw[:,1:2]),axis=0)
            outvar_dict['w']=np.concatenate((outvar_dict['w'],uvw[:,2:3]),axis=0)
            if weight is None:
                invar_dict['weightsqrt']=np.concatenate((invar_dict['weightsqrt'],coordsList[:,0:1]*0.+1.),axis=0)
                outvar_dict['elastix_u']=np.concatenate((outvar_dict['u'],uvw[:,0:1]),axis=0)
                outvar_dict['elastix_v']=np.concatenate((outvar_dict['v'],uvw[:,1:2]),axis=0)
                outvar_dict['elastix_w']=np.concatenate((outvar_dict['w'],uvw[:,2:3]),axis=0)
            else:
                invar_dict['weightsqrt']=np.concatenate((invar_dict['weightsqrt'],coordsList[:,0:1]*0.+weight[n]),axis=0)
                outvar_dict['elastix_u']=np.concatenate((outvar_dict['u'],uvw[:,0:1]),axis=0) * weight[n]
                outvar_dict['elastix_v']=np.concatenate((outvar_dict['v'],uvw[:,1:2]),axis=0) * weight[n]
                outvar_dict['elastix_w']=np.concatenate((outvar_dict['w'],uvw[:,2:3]),axis=0) * weight[n]
        if savepath is not None:
            np.savetxt(savepath+'t'+str(int(time_index_x[n]))+"to"+str(int(time_index_y[n]))+'.txt',np.concatenate((np.arange(coordsList.shape[0]).reshape((-1,1)),coordsList[:,0:3],uvw[:,0:3]),axis=-1),comments='#pointID ',header='x y z u v w')
    return invar_dict,outvar_dict
class outofbounderror(torch.nn.Module):
    def __init__(self,deformation_limit):
        super().__init__()
        deformation_limit=torch.tensor(deformation_limit**2.)
        self.register_buffer("deformation_limit", deformation_limit, persistent=False)
    def forward(self,x):
        return torch.nn.functional.relu(x**2.-self.deformation_limit)
class sincostt2(torch.nn.Module):
    def __init__(self,period):
        super().__init__()
        self.freq=2.*np.pi/period
    def forward(self,x):
        t=x*self.freq
        t2=t[...,0:1]+t[...,1:2]
        result=torch.cat((torch.sin(t[...,0:1]),torch.cos(t[...,0:1]),torch.sin(t2),torch.cos(t2)),dim=-1)
        
        return result
class weight_mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x[...,0:3]*x[...,3:4]
class INNFoward(torch.nn.Module):
    def __init__(self,in_features,nr_layers,hidden_layers,hidden_layersize):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(InheritedInvertibleFullyConnectedArchCore(in_features=in_features,
                                                            nr_layers=nr_layers,
                                                            hidden_layersize = hidden_layersize,
                                                            hidden_layers = hidden_layers,
                                                            reducescale=0.01,
                                                            scaleend=1.,
                                                            extras=2
                                                            ))#0
    def forward(self,x, w_1,w_2):
        return self.layers[0](x, w_1,w_2)
class INNReverse(torch.nn.Module):
    def __init__(self,in_features,nr_layers,hidden_layers,hidden_layersize):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(InheritedInvertibleFullyConnectedArchCore(in_features=in_features,
                                                            nr_layers=nr_layers,
                                                            hidden_layersize = hidden_layersize,
                                                            hidden_layers = hidden_layers,
                                                            reducescale=0.01,
                                                            scaleend=1.,
                                                            extras=2
                                                            ))#0
    def forward(self,x, w_1,w_2):
        return self.layers[0].reverse(x, w_1,w_2)

class meandeterminant(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        F=torch.stack((x[...,0:3],x[...,3:6],x[...,6:9]), dim=x.dim()-1)
        det=torch.linalg.det(F).unsqueeze(-1)
        #mean_det=torch.mean(det,dim=-2,keepdim=True).expand(x[...,0:1].size())
        return det
class deviationdeterminant(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        mean_det=torch.mean(x,dim=-2,keepdim=True)
        return x-mean_det
@modulus.main(config_path="conf", config_name="config_motionTrackRegularization")
def run(cfg: ModulusConfig) -> None:
    ## ------------------------------------------------
    ## initialization
    ## ------------------------------------------------
    # case data from cfg
    casePath=os.path.realpath(__file__)
    outputPath=os.path.dirname(casePath)
    print(casePath)
    casePath=os.path.abspath(os.path.join(outputPath, os.pardir, os.pardir))
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cfg.custom.CUDA_DEVICE)
    b_coef_file_format=cfg.custom.b_coef_file_format
    if b_coef_file_format is None:
        raise Exception("Specify SimpleElastix Bspline result file formatted as 0:fixed time point (from),  1:moving time point (to)")
    time_point_num=cfg.custom.time_point_num
    if time_point_num is None:
        raise Exception("Specific the number of time points with time_point_num")
    training_folder=cfg.custom.training_folder
    stlt0=cfg.custom.stlt0
    if stlt0 is not None:
        if not(os.path.isfile(os.path.join(casePath,stlt0))):
            raise Exception("STL file not found in "+os.path.join(casePath,stlt0))
    input_all=cfg.custom.input_all
    fileScale=cfg.custom.fileScale
    length_scale_in_mm=cfg.custom.length_scale_in_mm
    num_of_block=cfg.custom.num_of_block
    hidden_layer_per_NN=cfg.custom.hidden_layer_per_NN
    result_nodes=cfg.custom.result_nodes
    num_of_block=int(num_of_block)
    hidden_layer_per_NN=int(hidden_layer_per_NN)
    result_nodes=int(result_nodes)
    sampleratio=cfg.custom.sample_ratio
    volume_anchor_file=cfg.custom.volume_anchor_file # stl_file, stl from time, to time, volume
    smooth_time_file=cfg.custom.smooth_time_file
    # load file
    time_index = np.loadtxt(os.path.join(casePath,training_folder,'time_index_x_y.txt'))
    time_index_x=time_index[:,0].astype(int)
    time_index_y=time_index[:,1].astype(int)
    if time_index.shape[-1]>2:
        weight=np.sqrt(time_index[:,2])
    else:
        weight=np.ones_like(time_index[:,0])
    if os.path.isfile(os.path.join(outputPath,'invar_dict.pickle')):
        print("loading previous dictionary")
        with open(os.path.join(outputPath,'invar_dict.pickle'), 'rb') as handle:
            invar_dict = pickle.load(handle)
        with open(os.path.join(outputPath,'outvar_dict.pickle'), 'rb') as handle:
            outvar_dict = pickle.load(handle)
    else:
        print("calculating dictionary on "+str(len(time_index_x))+" pairs")
        invar_dict,outvar_dict=extract_deform(os.path.join(casePath,training_folder,b_coef_file_format),time_index_x,time_index_y,length_scale_in_mm,fileScale,weight=weight,sampleratio=sampleratio)
        with open(os.path.join(outputPath,'invar_dict.pickle'), 'wb') as handle:
            pickle.dump(invar_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(outputPath,'outvar_dict.pickle'), 'wb') as handle:
            pickle.dump(outvar_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    for key in invar_dict:
            print(key,invar_dict[key].shape)
    for key in outvar_dict:
            print(key,outvar_dict[key].shape)
    # NN hyperparameter
    # case_layer_size_list = [1024,1024,1024]
    # total_modes = 512
    # # set para
    # caseID_param_values = np.array(b_coef)
    # caseID_param_keys=[Key("b_coef",size=b_coef.shape[1])]
    # param_ranges = {
    #     Symbol("caseID"): np.arange(caseID_param_values.shape[0]).reshape((-1,1)),
    # }
    
    ## ------------------------------------------------
    ## FourierNN
    ## ------------------------------------------------
    # physical quantities
    # cycle
    cycle = quantity(time_point_num, "s")
    length_scale = quantity(length_scale_in_mm, "mm") 
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=cycle
    ) # dimensionless scaling 
    
    # set geometry
    # bspline domain

    # image domain
    
    pr = Parameterization() 

    # construct image rectangle (if use fixed points,ignore it)
    
    # define the nerual network 
    getuvw_net =SubtractionArch(
        [Key("x_t"), Key("y_t"), Key("z_t")],#[Key("Delta_t")]
        [Key("x"), Key("y"), Key("z")],
        [Key("u"), Key("v"), Key("w")],
        )
    
    sincostt2_net =CustomModuleArch(
        [Key("t"),Key("Delta_t")],#[Key("Delta_t")]
        [Key('sint'),Key('cost'),Key('sint2'),Key('cost2')],
        module=sincostt2(time_point_num)
        )
    weightsgen_w1_module=InheritedFullyConnectedWeightsGen(in_features=1+2,layer_sizeList=[result_nodes]*hidden_layer_per_NN+[2],additional_pre_dimension=(2,3*num_of_block+1)) ## 102
    weightsgen_w1_net=CustomZeroInputModuleArch(
            output_keys=[Key("w1",size=weightsgen_w1_module.size)],
        module=weightsgen_w1_module
        )
    weightsgen_w2_module=InheritedFullyConnectedWeightsGen(in_features=2+2,layer_sizeList=[result_nodes]*hidden_layer_per_NN+[1],additional_pre_dimension=(2,3*num_of_block+1))
    weightsgen_w2_net=CustomZeroInputModuleArch(
            output_keys=[Key("w2",size=weightsgen_w2_module.size)],
        module=weightsgen_w2_module
        )
    result_fwd_net=CustomTripleInputModuleArch(
            input1_keys=[Key("x"), Key("y"), Key("z"),Key('sint'),Key('cost')],
            input2_keys=[Key("w1",size=weightsgen_w1_module.size)],
            input3_keys=[Key("w2",size=weightsgen_w2_module.size)],
            output_keys=[Key("arbitary_x"),Key("arbitary_y"),Key("arbitary_z")],
        module=INNFoward(3,3*num_of_block,hidden_layer_per_NN,result_nodes)
        )
    result_rev_net=CustomTripleInputModuleArch(
            input1_keys=[Key("arbitary_x"),Key("arbitary_y"),Key("arbitary_z"),Key('sint2'),Key('cost2')],
            input2_keys=[Key("w1",size=weightsgen_w1_module.size)],
            input3_keys=[Key("w2",size=weightsgen_w2_module.size)],
            output_keys=[Key("x_t"), Key("y_t"), Key("z_t")],
        module=INNReverse(3,3*num_of_block,hidden_layer_per_NN,result_nodes)
        )
    result_weight_net=CustomModuleArch(
            input_keys=[Key("u"), Key("v"), Key("w"),Key("weightsqrt")],
            output_keys=[Key('elastix_'+x) for x in ['u','v','w']],
        module=weight_mod()
        )
    baseStr=[]
    for n1 in ["u","v","w"]:
        for n2 in ["x","y","z"]:
            baseStr.append(n1+n2)
    Fmat_eq=Fmat(case_coord_strList=["x","y","z"],matrix_string='F',returnQR=False)
    determinant_net=CustomModuleArch(
        [Key("F_"+x) for x in baseStr],
        [Key("sum_det")],
        module=meandeterminant()
        )
    smooth_net=CustomModuleArch(
        [Key("sum_det")],
        [Key("deviation_det")],
        module=deviationdeterminant()
        )
    # define the loss 
    
    nodes = (
        # [caseID_net.make_node(name='caseID_net')]
        # +[modeweights_net.make_node(name='modeweights_net')]
        [result_fwd_net.make_node(name="result_fwd_net")]
        + [result_rev_net.make_node(name="result_rev_net")]
        + [result_weight_net.make_node(name="result_weight_net")]
        + [weightsgen_w1_net.make_node(name="weightsgen_w1_net")]
        + [weightsgen_w2_net.make_node(name="weightsgen_w2_net")]
        + [sincostt2_net.make_node(name="sincostt2_net")]
        + [getuvw_net.make_node(name="getuvw_net")]
        + [determinant_net.make_node(name="determinant_net")]
        + [smooth_net.make_node(name="smooth_net")]
        + Fmat_eq.make_nodes()
    ) 
    # for n in range(N_Fourier*3):
    #     nodes = nodes+([get_net_cos[n].make_node(name='get_net_cos_'+str(n))]
    #                     +[get_net_sin[n].make_node(name='get_net_sin_'+str(n))]
    #                     )

    # make domain
    domain = Domain()
    #x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
    # interior contraints (training data)
    interior = PointwiseConstraint.from_numpy(
        nodes=nodes,
        invar=invar_dict,
        outvar=outvar_dict,
        batch_size=cfg.batch_size.interior,
    )
        
    domain.add_constraint(interior, "interior")
    #volume constrain
    if volume_anchor_file is not None:
        if os.path.isfile(os.path.join(casePath,training_folder,volume_anchor_file.format(0))):
            with open(os.path.join(casePath,training_folder,volume_anchor_file.format(0)),'r') as f:
                volume_anchor_file_read=f.readlines()
                print("anchor",volume_anchor_file_read)
            while volume_anchor_file_read[0][-1:] in ['\r','\n']:
                volume_anchor_file_read[0]=volume_anchor_file_read[0][:-1]
            volume_anchor_stltimefrom=[float(volume_anchor_file_read[1])]
            volume_anchor_stlfilefrom=[os.path.join(casePath,volume_anchor_file_read[0].format(volume_anchor_stltimefrom[-1]))]
            volume_anchor_stltimeto=[float(volume_anchor_file_read[2])]
            volume_anchor_stlfileto=[os.path.join(casePath,volume_anchor_file_read[0].format(volume_anchor_stltimeto[-1]))]
            volume_anchor_volumeratio=[float(volume_anchor_file_read[3])]
            if len(volume_anchor_file_read)>4:
                volume_anchor_lambda=[float(volume_anchor_file_read[4])]
            else:
                volume_anchor_lambda=[1.]
            if volume_anchor_file.format(1)!=volume_anchor_file.format(0):
                read_file_num=1
                while os.path.isfile(os.path.join(casePath,training_folder,volume_anchor_file.format(read_file_num))):
                    with open(os.path.join(casePath,training_folder,volume_anchor_file.format(read_file_num)),'r') as f:
                        volume_anchor_file_read=f.readlines()
                        print("anchor",read_file_num,volume_anchor_file_read)
                    while volume_anchor_file_read[0][-1:] in ['\r','\n']:
                        volume_anchor_file_read[0]=volume_anchor_file_read[0][:-1]
                    volume_anchor_stltimefrom.append(float(volume_anchor_file_read[1]))
                    volume_anchor_stlfilefrom.append(os.path.join(casePath,volume_anchor_file_read[0].format(volume_anchor_stltimefrom[-1])))
                    volume_anchor_stltimeto.append(float(volume_anchor_file_read[2]))
                    volume_anchor_stlfileto.append(os.path.join(casePath,volume_anchor_file_read[0].format(volume_anchor_stltimeto[-1])))
                    volume_anchor_volumeratio.append(float(volume_anchor_file_read[3]))
                    if len(volume_anchor_file_read)>4:
                        volume_anchor_lambda.append(float(volume_anchor_file_read[4]))
                    else:
                        volume_anchor_lambda.append(1.)
                    read_file_num+=1
        else:
            raise Exception("Cannot find stl to load interior points.")
        interior_mesh_from=[]
        volume_constrain_from=[]
        interior_mesh_to=[]
        volume_constrain_to=[]
        if smooth_time_file is not None:
            smooth_time_file=np.loadtxt(os.path.join(casePath,training_folder,smooth_time_file)).reshape(-1)
            print("smooth time with anchor",smooth_time_file)
            volume_constrain_from_ref0=[]
            volume_constrain_to_ref0=[]
        for n in range(len(volume_anchor_volumeratio)):
            interior_mesh_from.append(Tessellation.from_stl(
                volume_anchor_stlfilefrom[n], airtight=True,
            ))
            volume_constrain_from.append(PointwiseInteriorConstraint(
                nodes=nodes,
                geometry=interior_mesh_from[n],
                outvar={'sum_det':volume_anchor_volumeratio[n]},
                lambda_weighting={'sum_det':volume_anchor_lambda[n]},
                parameterization=Parameterization({Symbol("t"): volume_anchor_stltimefrom[n],Symbol("Delta_t"): volume_anchor_stltimeto[n]-volume_anchor_stltimefrom[n],}),
                batch_size=cfg.batch_size.anchor,
                batch_per_epoch= 1000,
                fixed_dataset = True,
                quasirandom =True,
            ))
            domain.add_constraint(volume_constrain_from[-1], "volume"+str(n)+"_from")
            interior_mesh_to.append(Tessellation.from_stl(
                volume_anchor_stlfileto[n], airtight=True,
            ))
            volume_constrain_to.append(PointwiseInteriorConstraint(
                nodes=nodes,
                geometry=interior_mesh_to[n],
                outvar={'sum_det':1./volume_anchor_volumeratio[n]},
                lambda_weighting={'sum_det':volume_anchor_lambda[n]},
                parameterization=Parameterization({Symbol("t"): volume_anchor_stltimeto[n],Symbol("Delta_t"): volume_anchor_stltimefrom[n]-volume_anchor_stltimeto[n],}),
                batch_size=cfg.batch_size.anchor,
                batch_per_epoch= 1000,
                fixed_dataset = True,
                quasirandom =True,
            ))
            domain.add_constraint(volume_constrain_to[-1], "volume"+str(n)+"_to")
            if smooth_time_file is not None:
                for tref in smooth_time_file:
                    volume_constrain_from_ref0.append(PointwiseInteriorConstraint(
                        nodes=nodes,
                        geometry=interior_mesh_from[n],
                        outvar={'deviation_det':0.},
                        lambda_weighting={'deviation_det':volume_anchor_lambda[n]},
                        parameterization=Parameterization({Symbol("t"): volume_anchor_stltimefrom[n],Symbol("Delta_t"): tref-volume_anchor_stltimefrom[n],}),
                        batch_size=cfg.batch_size.anchor,
                        batch_per_epoch= 1000,
                        fixed_dataset = True,
                        quasirandom =True,
                    ))
                    domain.add_constraint(volume_constrain_from_ref0[-1], "volume"+str(n)+"_from_ref"+str(tref))
                    volume_constrain_to_ref0.append(PointwiseInteriorConstraint(
                        nodes=nodes,
                        geometry=interior_mesh_to[n],
                        outvar={'deviation_det':0.},
                        lambda_weighting={'deviation_det':volume_anchor_lambda[n]},
                        parameterization=Parameterization({Symbol("t"): volume_anchor_stltimeto[n],Symbol("Delta_t"): tref-volume_anchor_stltimeto[n],}),
                        batch_size=cfg.batch_size.anchor,
                        batch_per_epoch= 1000,
                        fixed_dataset = True,
                        quasirandom =True,
                    ))
                    domain.add_constraint(volume_constrain_to_ref0[-1], "volume"+str(n)+"_to_ref"+str(tref))
    # inferencer (training set)
    openfoam_inferencer=[]   
    if os.path.isfile(os.path.join(outputPath,'invar_dict_validator.pickle')):
        print("loading previous validator dictionary")
        with open(os.path.join(outputPath,'invar_dict_validator.pickle'), 'rb') as handle:
            invar_dict_validator = pickle.load(handle)
        with open(os.path.join(outputPath,'outvar_dict_validator.pickle'), 'rb') as handle:
            outvar_dict_validator = pickle.load(handle)
    else:
        invar_dict_validator,outvar_dict_validator=extract_deform(os.path.join(casePath,training_folder,b_coef_file_format),time_index_x,time_index_y,length_scale_in_mm,fileScale,weight=weight,sampleratio=1.,savepath=os.path.join(outputPath,'elastix_'),separate=True)
        with open(os.path.join(outputPath,'invar_dict_validator.pickle'), 'wb') as handle:
            pickle.dump(invar_dict_validator, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(outputPath,'outvar_dict_validator.pickle'), 'wb') as handle:
            pickle.dump(outvar_dict_validator, handle, protocol=pickle.HIGHEST_PROTOCOL)
    coordsList=np.concatenate((invar_dict_validator[0]['x'],invar_dict_validator[0]['y'],invar_dict_validator[0]['z']),axis=-1)
    if stlt0 is not None:
        if stlt0[-4:]=='.stl':
            mesh=trimesh.load_mesh(os.path.join(casePath,stlt0)) 
            meshcoords=np.array(mesh.vertices)[:,0:3]/length_scale_in_mm
        elif stlt0[-4:]=='.vtk':
            stl0_filname=stlt0[:-4]
            stlt0=VTKPolyData.init_from_obj(casePath,stl0_filname)
            print("WARNING: VTK data unable to scale. Please ensure data is pre-scaled.")
            if os.path.isfile(os.path.join(casePath,stl0_filname)):
                raise Exception("VTK not implemented.")
                stlt0_invar_dict,stlt0_outvar_dict=get_BSF_deform(os.path.join(casePath,training_folder,b_coef_file_format),time_index_x,time_index_y,length_scale_in_mm,fileScale,weight=weight,sampleratio=sampleratio)
                ####!!!!
        else:
            meshcoords=np.loadtxt(os.path.join(casePath,stlt0))[:,0:3]/length_scale_in_mm
    if isinstance(input_all,str):
        if input_all[-7:]=='.pickle':
            with open(os.path.join(casePath,input_all), 'rb') as handle:
                input_all = pickle.load(handle)
        elif os.path.isfile(os.path.join(casePath,input_all+'.pickle')):
            with open(os.path.join(casePath,input_all+'.pickle'), 'rb') as handle:
                input_all = pickle.load(handle)
        else:
            if input_all[-4:]=='.txt':
                try:
                    input_all_temp=np.loadtxt(input_all)
                    if len(input_all_temp.shape)>1:
                        if input_all_temp.shape[1]>3:
                            input_all=[]
                            for n in range(int(time_point_num)):
                                if np.sum(input_all_temp[:,3]==n)>0:
                                    input_all.append(input_all_temp[input_all_temp[:,3]==n])
                                else:
                                    input_all.append(None)
                        else:
                            input_all=input_all_temp
                    else:
                        raise Exception("Unable to use numpy loadtxt on file. Trying BsplineFourier.")
                except:
                    bsf=BsplineFourier(input_all)
                    input_all=[np.concatenate((coordsList,coordsList[0:1]*0.,coordsList[0:3]*0.),axis=-1)]
                    for n in range(1,int(time_point_num)):
                        input_all.append(np.concatenate((coordsList,coordsList[0:1]*0.,bsf(np.concatenate((coordsList,coordsList[0:1]*0.+n),axis=-1))),axis=-1))
            else:
                input_all_temp=np.load(input_all)
                if input_all_temp.shape[1]>3:
                    input_all=[]
                    for n in range(int(time_point_num)):
                        if np.sum(input_all_temp[:,3]==n)>0:
                            input_all.append(input_all_temp[input_all_temp[:,3]==n])
                        else:
                            input_all.append(None)
                else:
                    input_all=input_all_temp
            with open(os.path.join(casePath,input_all+'.pickle'), 'wb') as handle:
                pickle.dump(input_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    openfoam_inferencer2=[]
    openfoam_inferencer3=[]
    for n in range(int(time_point_num)):
        if input_all is not None:
            if isinstance(input_all,list):
                if input_all[n] is not None:
                    openfoam_invar_numpy={"x":input_all[n][...,0:1],
                                        "y":input_all[n][...,1:2],
                                        "z":input_all[n][...,2:3],
                                        "t":0.*input_all[n][...,0:1],
                                        "Delta_t":n+0.*input_all[n][...,0:1],
                                        }
                    for nn in range(4,input_all[n].shape[1]):
                        openfoam_invar_numpy["input"+str(nn)]=input_all[n][...,nn:nn+1]
                    openfoam_inferencer.append(PointwiseInferencer(
                        nodes=nodes, invar=openfoam_invar_numpy, output_names=['u','v','w']
                    ))
                    domain.add_inferencer(openfoam_inferencer[-1], "inf_data"+str(n))
            else:
                openfoam_invar_numpy={"x":input_all[...,0:1],
                                    "y":input_all[...,1:2],
                                    "z":input_all[...,2:3],
                                    "t":0.*input_all[...,0:1],
                                    "Delta_t":n+0.*input_all[...,0:1],
                                    }
                openfoam_inferencer.append(PointwiseInferencer(
                    nodes=nodes, invar=openfoam_invar_numpy, output_names=['u','v','w']
                ))
                domain.add_inferencer(openfoam_inferencer[-1], "inf_data"+str(n))
        openfoam_invar_numpy2={"x":coordsList[...,0:1],
                            "y":coordsList[...,1:2],
                            "z":coordsList[...,2:3],
                            "t":0.*coordsList[...,0:1],
                            "Delta_t":n+0.*coordsList[...,0:1],
                            }
        openfoam_inferencer2.append(PointwiseInferencer(
            nodes=nodes, invar=openfoam_invar_numpy2, output_names=['u','v','w']
        ))
        domain.add_inferencer(openfoam_inferencer2[-1], "motion_data"+str(n))
        openfoam_invar_numpy2={"x":coordsList[...,0:1],
                            "y":coordsList[...,1:2],
                            "z":coordsList[...,2:3],
                            "t":n+0.*coordsList[...,0:1],
                            "Delta_t":1+0.*coordsList[...,0:1],
                            }
        openfoam_inferencer2.append(PointwiseInferencer(
            nodes=nodes, invar=openfoam_invar_numpy2, output_names=['u','v','w']
        ))
        domain.add_inferencer(openfoam_inferencer2[-1], "march_data"+str(n))
        if stlt0 is not None:
            if isinstance(stlt0,VTKPolyData):
                temp_points=0.*stlt0.get_points(dims=[0])
                openfoam_invar_numpy3={"t":temp_points,
                                    "Delta_t":n+temp_points,
                                    }
                openfoam_inferencer3.append(PointVTKInferencer(
                    vtk_obj=stlt0, nodes=nodes, input_vtk_map={'x':['x'],'y':['y'],'z':['z']},invar =openfoam_invar_numpy3, output_names=['u','v','w']
                ))
                domain.add_inferencer(openfoam_inferencer3[-1], "mesh_data"+str(n))
            else:
                openfoam_invar_numpy3={"x":meshcoords[...,0:1],
                                    "y":meshcoords[...,1:2],
                                    "z":meshcoords[...,2:3],
                                    "t":0.*meshcoords[...,0:1],
                                    "Delta_t":n+0.*meshcoords[...,0:1],
                                    }
                openfoam_inferencer3.append(PointwiseInferencer(
                    nodes=nodes, invar=openfoam_invar_numpy3, output_names=['u','v','w']
                ))
                domain.add_inferencer(openfoam_inferencer3[-1], "mesh_data"+str(n))
    
    openfoam_validator=[]
    for n in range(len(time_index_x)):
        openfoam_validator.append(PointwiseValidator(
            nodes=nodes,
            invar=invar_dict_validator[n],
            true_outvar=outvar_dict_validator[n],
        ))
        domain.add_validator(openfoam_validator[-1],"compare_t"+str(time_index_x[n])+"to"+str(time_index_y[n]))
        if time_index_y[n]-time_index_x[n]==1:
            domain.add_validator(openfoam_validator[-1],"compare_plus1to_t"+str(time_index_x[n]))



    # make solver
    slv = Solver_NoRecordConstraints(cfg, domain)
    # start solver
    slv.solve()
    
if __name__ == "__main__":
    run()



