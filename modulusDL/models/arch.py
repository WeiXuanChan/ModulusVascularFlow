# -*- coding: utf-8 -*-
"""
################################################
MIT License
Copyright (c) 2021 L. C. Lee
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
################################################
File: ModulusModel_GeometricalModes.py
Description: Geometrical-Modes Model Architecture For Nvidia Modulus

History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         10Mar2023           - Created
"""
_version='1.0.0'
from typing import Optional, Dict, Tuple, Union, List
from modulus.key import Key

import numpy as np
import itertools
import torch
import torch.nn as nn
from torch import Tensor

from modulus.models.layers import Activation, FCLayer, Conv1dFCLayer, get_activation_fn
from modulus.models.arch import Arch
from modulus.models.fully_connected import FullyConnectedArchCore
from .layers import inheritedFCLayer, singleInputInheritedFCLayer, cosSinLayer , cosLayer , sinLayer , PILayer

###FUNCTIONS frequently used
def getTN(layersizes):
    '''
    Function to calculate the number of weights and bias inside a fully connected network
    '''
    count=0
    for n in range(len(layersizes)-1):
        count+=(layersizes[n]+1)*layersizes[n+1]
    return count
def createKey(keystr,startnumber,endnumber):
    '''
    Function to create Key List with running ID
    '''
    result=[]
    for n in range(startnumber,endnumber):
        result.append(Key(keystr+str(n)))
    return result
###WeightsGen
class InheritedFullyConnectedWeightsGen(nn.Module):
    '''
    Architecture core for fully connected neural network with different number of nodes per layer.
    This architecture allows weights and bias to be input as secondary input in dual input
    '''
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        bias: bool = True,
        weight_norm: bool = True,
        additional_pre_dimension=(1,),
    ) -> None:
        super().__init__()
        self.in_features=in_features
        if isinstance(layer_sizeList,int):
            layer_sizeList=[layer_sizeList]
        weights=[]
        for n in range(np.prod(additional_pre_dimension)):
            temp_tensor=torch.empty((layer_sizeList[0],in_features))
            nn.init.xavier_uniform_(temp_tensor)
            temp_weight = temp_tensor.clone().detach()
            #temp_weight=torch.tensor(temp_tensor)
            if bias:
                temp_tensor=torch.empty((layer_sizeList[0],1))
                nn.init.constant_(temp_tensor, 0)
                temp_weight=torch.cat((temp_weight,temp_tensor),-1)
            if weight_norm:
                temp_tensor=torch.empty((layer_sizeList[0],1))
                nn.init.constant_(temp_tensor, 1)
                temp_weight=torch.cat((temp_weight,temp_tensor),-1)
            flatten_weights=temp_weight.reshape(-1)
            for n in range(1,len(layer_sizeList)):
                temp_tensor=torch.empty((layer_sizeList[n],layer_sizeList[n-1]))
                nn.init.xavier_uniform_(temp_tensor)
                temp_weight = temp_tensor.clone().detach()
                # temp_weight=torch.tensor(temp_tensor)
                if bias:
                    temp_tensor=torch.empty((layer_sizeList[n],1))
                    nn.init.constant_(temp_tensor, 0)
                    temp_weight=torch.cat((temp_weight,temp_tensor),-1)
                if weight_norm and n<(len(layer_sizeList)-1): ## len
                    temp_tensor=torch.empty((layer_sizeList[n],1))
                    nn.init.constant_(temp_tensor, 1)
                    temp_weight=torch.cat((temp_weight,temp_tensor),-1)
                flatten_weights=torch.cat((flatten_weights,temp_weight.reshape(-1),torch.tensor([0]))) ## tensor[0]
            weights.append(flatten_weights)
        stacked_weights = torch.stack(weights, dim=0) # list to tensor
        #domainNN_size_torch = torch.tensor(domainNN_size,dtype=torch.long)
        #self.register_buffer("domainNN_size", domainNN_size_torch, persistent=False)
        self.weight_g = nn.Parameter(stacked_weights.reshape(additional_pre_dimension+(-1,))) ## tensor weight
        self.size=int(self.weight_g.size(-1))
    def forward(self) -> Tensor:
        return self.weight_g
    
###ArchCore
class AdditionArchCore(nn.Module):
    '''
    Addition architecture core
    '''
    def __init__(
        self
    ) -> None:
        super().__init__()
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        return x+o
class SubtractionArchCore(nn.Module):
    '''
    Subtraction architecture core
    '''
    def __init__(
        self
    ) -> None:
        super().__init__()
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        return x-o
class MultiplicationArchCore(nn.Module):
    '''
    Element multiplication architecture core
    '''
    def __init__(
        self
    ) -> None:
        super().__init__()
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        return x*o
class MatrixMultiplicationArchCore(nn.Module):
    '''
    Matrix multiplication (operates on the last two dimension only) architecture core
    '''
    def __init__(
        self
    ) -> None:
        super().__init__()
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        return torch.matmul(x,o)
class SumMultiplicationArchCore(nn.Module):
    '''
    Linear layer architecture core with summation of element multiplication
    '''
    def __init__(
        self
    ) -> None:
        super().__init__()
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        return torch.sum(x*o,-1,keepdim=True)
class CaseIDArchCore(nn.Module):
    '''
    Architecture core to output case Identification matrix which tag each sample point to a case index
    '''
    def __init__(
        self,
        case_num: int,
    ) -> None:
        super().__init__()
        if case_num<=0:
            raise Exception("case number must be a positive integer.")
        else:
            caseID_range = torch.tensor(np.arange(case_num).reshape((1,-1)),dtype=torch.long)
            self.register_buffer("caseID_range", caseID_range, persistent=False)
            self.unit_function=Unit_function.apply
    def forward(self, x: Tensor) -> Tensor:
        caseID=x.detach()
        caseID_unit=self.unit_function(caseID.expand((*caseID.size()[:-1],self.caseID_range.size(-1)))-self.caseID_range)#training_points,case_num
        return caseID_unit
class ParametricInsertArchCore(nn.Module):
    '''
    Architecture core to output trainable variables
    '''
    def __init__(
        self,
        out_features: int = 512,
        set_to:float =1.0
    ) -> None:
        super().__init__()
        self.out_features=out_features
        self.weight_g = nn.Parameter(torch.empty((1,out_features)))
        self.reset_parameters(set_to)

    def reset_parameters(self,set_to=1.0) -> None:
        torch.nn.init.constant_(self.weight_g, set_to)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight_g.expand(x.size()[:-1]+(-1,))

    def extra_repr(self) -> str:
        return "out_features={}".format(
            self.out_features
        )
class CosSinArchCore(nn.Module):
    '''
    Sine and Cosine architecture core which performs sin or cosine operation to inputs
    '''
    def __init__(
        self,
        in_features: int = 512,
        layer_size: int = 512,
        out_features: int = 512,
        nr_layers: int = 0,
        activation_fn: Activation = Activation.IDENTITY,
        cos_layer: bool=True,
        sin_layer: bool=True
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        layer_in_features = in_features
        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                nr_layers - len(activation_fn)
            )
        if cos_layer and sin_layer:
            CSLayer=cosSinLayer
        elif cos_layer:
            CSLayer=cosLayer
        elif sin_layer:
            CSLayer=sinLayer
        else:
            raise Exception("Either sin_layer or/and ccos_layer must be True")
        for i in range(nr_layers):
            self.layers.append(
                CSLayer(
                    layer_in_features,
                    layer_size,
                    activation_fn[i],
                )
            )
            layer_in_features = layer_size

        self.final_layer = CSLayer(
            in_features=layer_in_features,
            out_features=out_features,
            activation_fn=Activation.IDENTITY,
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.final_layer(x)
        return x
class BackgroundGraphNNFullyConnectedFlexiLayerSizeArchCore(nn.Module):
    '''
    Fully connected architecture core with background GNN convolution
    '''
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        skip_layer: int = 1,
    ) -> None:
        super().__init__()

        self.skip_layer = skip_layer
        if layer_sizeList is None:
            self.layers=[]
            self.final_layer=torch.nn.Identity()
        else:
            if isinstance(layer_sizeList,int):
                layer_sizeList=[layer_sizeList]
            nr_layers=len(layer_sizeList)-1
            # Allows for regular linear layers to be swapped for 1D Convs
            # Useful for channel operations in FNO/Transformers
    
            if adaptive_activations:
                activation_par = nn.Parameter(torch.ones(1))
            else:
                activation_par = None
    
            if not isinstance(activation_fn, list):
                activation_fn = [activation_fn] * nr_layers
            if len(activation_fn) < nr_layers:
                activation_fn = activation_fn + [activation_fn[-1]] * (
                    nr_layers - len(activation_fn)
                )
    
            self.layers = nn.ModuleList()
    
            layer_in_features = in_features
            for i in range(nr_layers):
                self.layers.append(
                    FCLayer(
                        layer_in_features,
                        layer_sizeList[i],
                        activation_fn[i],
                        weight_norm,
                        activation_par,
                    )
                )
                layer_in_features = layer_sizeList[i]
    
            self.final_layer = FCLayer(
                in_features=layer_in_features,
                out_features=layer_sizeList[-1],
                activation_fn=Activation.IDENTITY,
                weight_norm=False,
                activation_par=None,
            )
    def forward(self,x: Tensor,bgGNN: Tensor,bgGNNSampleadjacency: Tensor,bgGNNBackgroundadjacency: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            if self.skip_layer <= i :
                x=torch.matmul(bgGNNSampleadjacency[...,1:],bgGNN)+x*bgGNNSampleadjacency[...,0:1]
                bgGNN=torch.matmul(bgGNNBackgroundadjacency[...,1:],bgGNN)+bgGNN*bgGNNBackgroundadjacency[...,0:1]
            x = layer(x)
            bgGNN=layer(bgGNN)
        if self.skip_layer <= len(self.layers) :
            x=torch.matmul(bgGNNSampleadjacency[...,1:],bgGNN)+x*bgGNNSampleadjacency[...,0:1]
            bgGNN=torch.matmul(bgGNNBackgroundadjacency[...,1:],bgGNN)+bgGNN*bgGNNBackgroundadjacency[...,0:1]
        x = self.final_layer(x)
        return x
class BackgroundGNNAdjacencyArchCore(nn.Module):
    '''
    Architecture core to calculate edge weight for background GNN into adjacency matrix
    '''
    def __init__(
        self,
        dist_upper_limit: float,
        p_norm:float =2.,
        activation_fn=None,
    ) -> None:
        super().__init__()
        self.p_norm=p_norm
        dist_upper_limit=torch.tensor(dist_upper_limit)
        self.register_buffer("dist_upper_limit", dist_upper_limit, persistent=False)
        self.activation_fn=nn.ModuleList()
        if activation_fn is None:
            self.activation_fn.append(torch.nn.Identity())
        else:
            self.activation_fn.append(activation_fn)
    def forward(self,x: Tensor, bg_GNN_coordinates:Tensor) -> Tensor:
        return self.activation_fn[0](torch.nn.functional.relu(1.-torch.cdist(x, bg_GNN_coordinates, p=self.p_norm)/self.dist_upper_limit))
def BackgroundGNN_NormalizeAdjacencyNumpy(unnormalized_sampleadjacency,unnormalized_bgadjacency,bgweight=None):
    '''
    Numpy version to normalize background GNN Adjacency matrix
    '''
    if bgweight is None:
        one_plus_sum_col=1.+np.sum(unnormalized_sampleadjacency,axis=-1,keepdims=True)#m,1
        D_col=1./np.sqrt(one_plus_sum_col*(unnormalized_sampleadjacency+np.sum(unnormalized_bgadjacency,axis=-2,keepdims=True)))#m,n
        one_plus_sum_col=1./one_plus_sum_col#m,1
        return np.concatenate((one_plus_sum_col,D_col*unnormalized_sampleadjacency),axis=-1)
    else:
        one_plus_sum_col=np.sum(unnormalized_sampleadjacency,axis=-1,keepdims=True)#m,1
        D_col=unnormalized_sampleadjacency*np.sqrt(bgweight)/np.sqrt(one_plus_sum_col*(unnormalized_sampleadjacency+np.sum(unnormalized_bgadjacency,axis=-2,keepdims=True)))#m,n
        return np.concatenate((np.zeros_like(one_plus_sum_col)+(1.-bgweight),D_col),axis=-1)
class BackgroundGNN_NormalizeAdjacencyArchCore(nn.Module):
    '''
    Architecture core to normalize background GNN Adjacency matrix
    '''
    def __init__(
        self,
        bgweight=None,
    ) -> None:
        super().__init__()
        if bgweight is None:
            self.bgweight=None
        else:
            bgweight=torch.tensor(bgweight)
            self.register_buffer("bgweight", bgweight, persistent=False)
    def forward(self,unnormalized_sampleadjacency:Tensor, unnormalized_bgadjacency: Tensor) -> Tensor:
        if self.bgweight is None:
            one_plus_sum_col=(1.+torch.sum(unnormalized_sampleadjacency,dim=-1,keepdim=True))#m,1
            D_col=1./torch.sqrt(one_plus_sum_col*(unnormalized_sampleadjacency+torch.sum(unnormalized_bgadjacency,dim=-2,keepdim=True)))#m,n
            one_plus_sum_col=1./one_plus_sum_col#m,1
            return torch.cat((one_plus_sum_col,D_col*unnormalized_sampleadjacency),dim=-1)
        else:
            #the 1. ->(1.-self.bgweight)*torch.sum(unnormalized_sampleadjacency,dim=-1,keepdim=True)
            one_plus_sum_col=torch.sum(unnormalized_sampleadjacency,dim=-1,keepdim=True)#m,1
            D_col=unnormalized_sampleadjacency*torch.sqrt(self.bgweight)/torch.sqrt(one_plus_sum_col*(unnormalized_sampleadjacency+torch.sum(unnormalized_bgadjacency,dim=-2,keepdim=True)))#m,n
            return torch.cat(((1.-self.bgweight).expand(D_col.size(0),1),D_col),dim=-1)
class FBWindowCore(nn.Module):
    '''
    Architecture core to calculate finite basis window
    '''
    def __init__(
        self,
        start,
        end,
        overlap,
        tol: float= 1e-10,
    ) -> None:
        super().__init__()
        if np.any(overlap <= 0) : raise Exception("ERROR overlapRatio ="+repr(overlap))
        scale=torch.tensor((np.log(1/tol-1)/overlap).reshape((1,-1)),dtype=torch.float)
        self.register_buffer("scale", scale, persistent=False)
        if len(start)>1:
            dmul=torch.tensor(range(1,len(start)),dtype=torch.long)
            self.register_buffer("dmul", dmul, persistent=False)
        else:
            self.dmul=None
        tol=torch.tensor(tol,dtype=torch.float)
        self.register_buffer("tol", tol, persistent=False)
        if np.any(start>=end): raise Exception("ERROR: xmin "+repr(start)+" >= xmax "+repr(end))
        start=torch.tensor(start.reshape((1,-1)),dtype=torch.float)
        self.register_buffer("start", start, persistent=False)
        end=torch.tensor(end.reshape((1,-1)),dtype=torch.float)
        self.register_buffer("end", end, persistent=False)
    def forward(self,x):
        ytop=torch.clamp(torch.sigmoid((self.end-x)*self.scale),min=self.tol)
        ybtm=torch.clamp(torch.sigmoid((x-self.start)*self.scale),min=self.tol)
        y_all=torch.clamp(ybtm*ytop-self.tol,min=0.)
        y=y_all[...,0:1]
        if self.dmul is not None:
            for n in self.dmul:
                y=y*y_all[...,n:n+1]
        return y
class FBLevelDivide(nn.Module):
    '''
    Architecture core to normalize of Finite Basis levels
    '''
    def __init__(
        self,
        nlevel,
    ) -> None:
        super().__init__()
        nlevel=torch.tensor(nlevel)
        self.register_buffer("nlevel", nlevel, persistent=False)
    def forward(self,x):
        return x/self.nlevel
class FullyConnectedFlexiLayerSizeArchCore(nn.Module):
    '''
    Architecture core for fully connected neural network with different number of nodes per layer
    '''
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        conv_layers: bool = False,
    ) -> None:
        super().__init__()

        self.skip_connections = skip_connections
        if isinstance(layer_sizeList,int):
            layer_sizeList=[layer_sizeList]
        nr_layers=len(layer_sizeList)-1
        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers
        if conv_layers:
            fc_layer = Conv1dFCLayer
        else:
            fc_layer = FCLayer

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                nr_layers - len(activation_fn)
            )

        self.layers = nn.ModuleList()

        layer_in_features = in_features
        for i in range(nr_layers):
            self.layers.append(
                fc_layer(
                    layer_in_features,
                    layer_sizeList[i],
                    activation_fn[i],
                    weight_norm,
                    activation_par,
                )
            )
            layer_in_features = layer_sizeList[i]

        self.final_layer = fc_layer(
            in_features=layer_in_features,
            out_features=layer_sizeList[-1],
            activation_fn=Activation.IDENTITY,
            weight_norm=False,
            activation_par=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x)
        return x

    def get_weight_list(self):
        weights = [layer.conv.weight for layer in self.layers] + [
            self.final_layer.conv.weight
        ]
        biases = [layer.conv.bias for layer in self.layers] + [
            self.final_layer.conv.bias
        ]
        return weights, biases
def PInetgetTN(in_features,layer_size,out_features,nr_layers,layer_size_reduction):
    layer_sizeList=[layer_size]
    submod_number=[layer_size_reduction]
    layer_trainable=[getTN([layer_size,out_features])]
    for n in range(nr_layers):
        layer_sizeList.append(int(layer_sizeList[-1]/layer_size_reduction))
        layer_trainable.append(getTN([layer_sizeList[-1],layer_sizeList[-2]])*submod_number[-1])
        submod_number.append(int(submod_number[-1]*layer_size_reduction))
    layer_trainable.append(getTN([in_features,layer_sizeList[-1]])*submod_number[-1])
    return sum(layer_trainable)
    
class PIConnectedArchCore(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        layer_size: int = 512,
        out_features: int = 512,
        nr_layers: int = 6,
        layer_size_reduction: int = 2,
        activation_fn: Activation = Activation.IDENTITY,
        skip_activation: int = -1,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        reducenorm:float = 0.2

    ) -> None:
        super().__init__()

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None
        #calculate size of each layer
        layer_sizeList=[layer_size]
        submod_number=[layer_size_reduction]
        for n in range(nr_layers):
            layer_sizeList.append(int(layer_sizeList[-1]/layer_size_reduction))
            submod_number.append(int(submod_number[-1]*layer_size_reduction))
        layer_sizeList=layer_sizeList[::-1]
        submod_number=submod_number[::-1]
        temp_layers = []
        for m in range(submod_number[0]):
            temp_layers.append(FCLayer(in_features,
                                    layer_sizeList[0],
                                    activation_fn=activation_fn,
                                    weight_norm=weight_norm,
                                    activation_par=activation_par))
        skip_activation_count=1
        for n in range(1,nr_layers+1):
            new_layers = []
            if skip_activation_count==skip_activation:
                skip_activation_count=0
                activation_fn_temp=Activation.IDENTITY
            else:
                skip_activation_count+=1
                activation_fn_temp=activation_fn
            for m in range(submod_number[n]):
                new_layers.append(PILayer(layer_sizeList[n-1],
                                           layer_sizeList[n],
                                           temp_layers[int(layer_size_reduction*m):int(layer_size_reduction*(m+1))],
                                           activation_fn=activation_fn_temp,
                                           weight_norm = weight_norm,
                                           activation_par = activation_par,
                                           reducenorm=reducenorm))
            temp_layers=new_layers
        self.layers= nn.ModuleList()
        self.layers.append(PILayer(layer_size,
                                    out_features,
                                    temp_layers,
                                    activation_fn=Activation.IDENTITY,
                                    weight_norm = weight_norm,
                                    activation_par = activation_par,
                                    reducenorm=reducenorm))
    def forward(self, x: Tensor) -> Tensor:
        x = self.layers[0](x)
        return x

class FullyConnectedFlexiLayerSizeFBWindowArchCore(nn.Module):
    '''
    Architecture core for fully connected neural network with different number of nodes per layer.
    Output multiplied by the window fuction.
    '''
    def __init__(
        self,
        startList: List[Union[float]],
        endList: List[Union[float]],
        overlapRatio:float,
        levels_num:int,
        tol: float= 1e-10,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        conv_layers: bool = False,
    ) -> None:
        super().__init__()
        self.FCNmoduleList = nn.ModuleList()
        self.window_moduleList = nn.ModuleList()
        dim=len(startList)
        discrete=np.linspace(startList,endList,num=levels_num+1)
        space_temp=(discrete[1]-discrete[0])
        overlapList=overlapRatio*space_temp
        discrete[0]=discrete[0]-space_temp
        discrete[-1]=discrete[-1]+space_temp
        windows_permute= np.unique(list(itertools.permutations(list(range(levels_num))*dim,dim)),axis=0)
        self.levels_num=levels_num
        for leveln in windows_permute:
            self.FCNmoduleList.append(FullyConnectedFlexiLayerSizeArchCore(in_features= in_features,
                                                                        layer_sizeList = layer_sizeList,
                                                                        skip_connections=skip_connections,
                                                                        activation_fn = activation_fn,
                                                                        adaptive_activations = adaptive_activations,
                                                                        weight_norm = weight_norm,
                                                                        conv_layers = conv_layers))
            temp_start_end=np.array([[discrete[x,xn],discrete[x+1,xn]] for xn,x in enumerate(leveln)])
            self.window_moduleList.append(FBWindowCore(temp_start_end[:,0],
                                                       temp_start_end[:,1],
                                                       overlapList,
                                                       tol= tol))
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        y=self.FCNmoduleList[0](x)*self.window_moduleList[0](o)
        for i in range(1,self.levels_num):
            y=y+self.FCNmoduleList[i](x)*self.window_moduleList[i](o)
        return y
            
class InheritedFullyConnectedFlexiLayerSizeCombinedArchCore(nn.Module):
    '''
    Architecture core for fully connected neural network with different number of nodes per layer.
    This architecture allows weights and bias to be input as secondary input in dual input
    '''
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        caseNN_in_features: int = 512,
        caseNN_layer_sizeList: Union[int,List[int],None] = None,
        caseNN_skip_connections: bool = False,
        caseNN_activation_fn: Activation = Activation.SILU,
        caseNN_adaptive_activations: bool = False,
        caseNN_weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_features=in_features
        self.caseNN_in_features=caseNN_in_features
        self.skip_connections = skip_connections
        self.caseNN_skip_connections = caseNN_skip_connections
        if isinstance(layer_sizeList,int):
            layer_sizeList=[layer_sizeList]
        nr_layers=len(layer_sizeList)-1
        domainNN_size=[0,(in_features+1)*layer_sizeList[0]]
        domainNN_weight_size=[in_features*layer_sizeList[0]]
        domainNN_bias_size=[layer_sizeList[0]]
        for n in range(1,len(layer_sizeList)):
            domainNN_weight_size.append(layer_sizeList[n-1]*layer_sizeList[n])
            domainNN_bias_size.append(layer_sizeList[n])
            domainNN_size.append((layer_sizeList[n-1]+1)*layer_sizeList[n])
        self.domainNN_cumulative_size=list(np.cumsum(domainNN_size))
        #domainNN_size_torch = torch.tensor(domainNN_size,dtype=torch.long)
        #self.register_buffer("domainNN_size", domainNN_size_torch, persistent=False)
        
        if caseNN_layer_sizeList is None:
            caseNN_layer_sizeList=[]
        elif isinstance(caseNN_layer_sizeList,int):
            caseNN_layer_sizeList=[caseNN_layer_sizeList]
        caseNN_layer_sizeList=caseNN_layer_sizeList+[sum(domainNN_size)]
        caseNN_nr_layers=len(caseNN_layer_sizeList)-1
        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers

        if caseNN_activation_fn:
            caseNN_activation_par = nn.Parameter(torch.ones(1))
        else:
            caseNN_activation_par = None

        if not isinstance(caseNN_activation_fn, list):
            caseNN_activation_fn = [caseNN_activation_fn] * caseNN_nr_layers
        if len(caseNN_activation_fn) < caseNN_nr_layers:
            caseNN_activation_fn = caseNN_activation_fn + [caseNN_activation_fn[-1]] * (
                caseNN_nr_layers - len(caseNN_activation_fn)
            )

        self.caseNN_layers = nn.ModuleList()

        layer_in_features = caseNN_in_features
        for i in range(caseNN_nr_layers):
            self.caseNN_layers.append(
                FCLayer(
                    layer_in_features,
                    caseNN_layer_sizeList[i],
                    caseNN_activation_fn[i],
                    caseNN_weight_norm,
                    caseNN_activation_par,
                )
            )
            layer_in_features = caseNN_layer_sizeList[i]
            
        self.caseNN_final_layer_weight = nn.ModuleList()
        self.caseNN_final_layer_bias = nn.ModuleList()
        for i in range(len(layer_sizeList)):
            self.caseNN_final_layer_weight.append( FCLayer(
                in_features=layer_in_features,
                out_features=domainNN_weight_size[i],
                activation_fn=Activation.IDENTITY,
                weight_norm=False,
                activation_par=None,
            ))
            self.caseNN_final_layer_bias.append( FCLayer(
                in_features=layer_in_features,
                out_features=domainNN_bias_size[i],
                activation_fn=Activation.IDENTITY,
                weight_norm=False,
                activation_par=None,
            ))
        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                nr_layers - len(activation_fn)
            )
        self.layers = nn.ModuleList()
        layer_in_features = in_features
        for i in range(nr_layers):
            self.layers.append(
                inheritedFCLayer(
                    layer_in_features,
                    layer_sizeList[i],
                    activation_fn[i],
                )
            )
            layer_in_features = layer_sizeList[i]
        self.final_layer = inheritedFCLayer(
            in_features=layer_in_features,
            out_features=layer_sizeList[-1],
            activation_fn=Activation.IDENTITY,
        )
        
    def forward(self, x: Tensor,o: Tensor) -> Tensor:#wrong!!!
        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.caseNN_layers):
            o = layer(o)
            if self.caseNN_skip_connections and i % 2 == 0:
                if x_skip is not None:
                    o, x_skip = o + x_skip, o
                else:
                    x_skip = o

        o = self.caseNN_final_layer(o)
        for i, layer in enumerate(self.layers):
            x = layer(x.unsqueeze(-3),o[...,self.domainNN_cumulative_size[i]:self.domainNN_cumulative_size[i+1]])
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x,o[...,self.domainNN_cumulative_size[-2]:self.domainNN_cumulative_size[-1]])
        return x.view((-1,)+x.size()[2:x.dim()])           
class InheritedFullyConnectedFlexiLayerSizeArchCore(nn.Module):
    '''
    Architecture core for fully connected neural network with different number of nodes per layer.
    This architecture allows weights and bias to be input as secondary input in dual input
    '''
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        bias: bool = True,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()
        self.in_features=in_features
        self.skip_connections = skip_connections
        if isinstance(layer_sizeList,int):
            layer_sizeList=[layer_sizeList]
        nr_layers=len(layer_sizeList)-1
        domainNN_size=[0]
        domainNN_weight_size=[in_features*layer_sizeList[0]]
        if bias:
            domainNN_bias_size=[layer_sizeList[0]]
        else:
            domainNN_bias_size=[0]
        if weight_norm:
            domainNN_weightsNorm_size=[layer_sizeList[0]]
        else:
            domainNN_weightsNorm_size=[0]
        domainNN_size.append(domainNN_weight_size[-1]+domainNN_bias_size[-1]+domainNN_weightsNorm_size[-1])
        for n in range(1,len(layer_sizeList)):
            domainNN_weight_size.append(layer_sizeList[n-1]*layer_sizeList[n])
            if bias:
                domainNN_bias_size.append(layer_sizeList[n])
            else:
                domainNN_bias_size.append(0)
            if weight_norm and n<(len(layer_sizeList)-1):
                domainNN_weightsNorm_size.append(layer_sizeList[n])
            else:
                domainNN_weightsNorm_size.append(0)
            domainNN_size.append(domainNN_weight_size[-1]+domainNN_bias_size[-1]+domainNN_weightsNorm_size[-1])
        self.domainNN_cumulative_size=list(np.cumsum(domainNN_size))
        #domainNN_size_torch = torch.tensor(domainNN_size,dtype=torch.long)
        #self.register_buffer("domainNN_size", domainNN_size_torch, persistent=False)
        
        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                nr_layers - len(activation_fn)
            )
        self.layers = nn.ModuleList()
        layer_in_features = in_features
        for i in range(nr_layers):
            self.layers.append(
                inheritedFCLayer(
                    layer_in_features,
                    layer_sizeList[i],
                    activation_fn[i],
                    bias=bias,
                    weight_norm=weight_norm,
                )
            )
            layer_in_features = layer_sizeList[i]
        self.final_layer = inheritedFCLayer(
            in_features=layer_in_features,
            out_features=layer_sizeList[-1],
            activation_fn=Activation.IDENTITY,
            bias=bias,
            weight_norm=False,
        )
        self.addsize=bias+weight_norm
        self.final_addsize=bias+0
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        x=x.unsqueeze(-1)
        for i, layer in enumerate(self.layers):
            # x = layer(x,o[...,self.domainNN_cumulative_size[i]:self.domainNN_cumulative_size[i+1]].reshape(o.size()[:-1]+[-1]+x.size(-1)))
            x = layer(x, o[..., self.domainNN_cumulative_size[i]:self.domainNN_cumulative_size[i+1]].reshape(o.size()[:-1] + (-1,) + (x.size(-2)+self.addsize,)))
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x
        x = self.final_layer(x,o[...,self.domainNN_cumulative_size[-2]:self.domainNN_cumulative_size[-1]].reshape(o.size()[:-1] + (-1,) + (x.size(-2)+self.final_addsize,)))#.reshape(o.size()[:-1]+[-1]+x.size(-1)))
        return x.squeeze(-1)
class SingleInputInheritedFullyConnectedFlexiLayerSizeCombinedArchCore(InheritedFullyConnectedFlexiLayerSizeCombinedArchCore):
    '''
    Architecture core for fully connected neural network with different number of nodes per layer.
    Both Hyoernetwork and domain network combined into a single class
    '''
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        caseNN_in_features: int = 512,
        caseNN_layer_sizeList: Union[int,List[int],None] = None,
        caseNN_skip_connections: bool = False,
        caseNN_activation_fn: Activation = Activation.SILU,
        caseNN_adaptive_activations: bool = False,
        caseNN_weight_norm: bool = True,
        with_caseID: int = 0,
    ) -> None:
        super().__init__(in_features = in_features,
                         layer_sizeList=layer_sizeList,
                         skip_connections=skip_connections,
                         activation_fn=activation_fn,
                         caseNN_in_features=caseNN_in_features,
                         caseNN_layer_sizeList=caseNN_layer_sizeList,
                         caseNN_skip_connections=caseNN_skip_connections,
                         caseNN_activation_fn=caseNN_activation_fn,
                         caseNN_adaptive_activations=caseNN_adaptive_activations,
                         caseNN_weight_norm=caseNN_weight_norm,)
        if with_caseID:
            self.CaseIDArch = nn.ModuleList()
            self.CaseIDArch.append(CaseIDArchCore(with_caseID))
            self.consolidative_matmul=Consolidative_matmul.apply
            self.with_caseID=True
        else:
            self.with_caseID=False
    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        if self.with_caseID:
            caseID=x[...,(self.in_features+self.caseNN_in_features):(self.in_features+self.caseNN_in_features+1)].detach()
            caseID_unit=torch.transpose(self.CaseIDArch[0](caseID),-1,-2)#case_num,training_points
            
            case_inputs=self.consolidative_matmul(caseID_unit,x[...,self.in_features:(self.in_features+self.caseNN_in_features)])
        else:
            case_inputs=x[...,self.in_features:(self.in_features+self.caseNN_in_features)]
        
        for i, layer in enumerate(self.caseNN_layers):
            case_inputs = layer(case_inputs)
            if self.caseNN_skip_connections and i % 2 == 0:
                if x_skip is not None:
                    case_inputs, x_skip = case_inputs + x_skip, case_inputs
                else:
                    x_skip = case_inputs

        domain_inputs=x[...,:self.in_features].unsqueeze(-2)
        domain_inputs_size=domain_inputs.size(0)
        for i, layer in enumerate(self.layers):
            weights_case=self.caseNN_final_layer_weight[i](case_inputs)
            bias_case=self.caseNN_final_layer_bias[i](case_inputs)
            if self.with_caseID:
                weights_case=torch.matmul(torch.transpose(caseID_unit,-1,-2),weights_case)
                bias_case=torch.matmul(torch.transpose(caseID_unit,-1,-2),bias_case)
            weights_case=weights_case.view((domain_inputs_size,layer.in_features,layer.out_features))
            bias_case=bias_case.unsqueeze(-2)
            domain_inputs = layer(domain_inputs,weights_case,bias_case)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    domain_inputs, x_skip = domain_inputs + x_skip, domain_inputs
                else:
                    x_skip = domain_inputs
        weights_case=self.caseNN_final_layer_weight[-1](case_inputs)
        bias_case=self.caseNN_final_layer_bias[-1](case_inputs)
        if self.with_caseID:
            weights_case=torch.matmul(torch.transpose(caseID_unit,-1,-2),weights_case)
            bias_case=torch.matmul(torch.transpose(caseID_unit,-1,-2),bias_case)
        weights_case=weights_case.view((domain_inputs_size,self.final_layer.in_features,self.final_layer.out_features))
        bias_case=bias_case.unsqueeze(-2)
        domain_inputs = self.final_layer(domain_inputs,weights_case,bias_case)
        return domain_inputs.squeeze(-2)
class PointNetMaxPoolArchCore(nn.Module):
    '''
    Architecture core for pointnet with max pool function
    '''
    def __init__(
        self,
        in_features: int = 512,
        with_caseID: int = 0,
    ) -> None:
        super().__init__()
        # Output forces to positive
        self.CaseIDArch = nn.ModuleList()
        self.CaseIDArch.append(CaseIDArchCore(with_caseID))
    def forward(self, x: Tensor) -> Tensor:
        caseID=x[...,self.in_features:(self.in_features+1)].detach()
        caseID_unit=self.CaseIDArch[0](caseID)#training_points,case_num
        case_max, max_indices=torch.max(torch.transpose(caseID_unit,-1,-2).unsqueeze(-1)*x[...,0:self.in_features].unsqueeze(0),-2)#case_num,features
        return torch.matmul(caseID_unit,case_max)
class InvertibleFullyConnectedArchCore(nn.Module):
    '''
    Architecture core for invertible fully connected neural network with different number of nodes per layer
    following DOI: arXiv:1808.04730
    '''
    def __init__(
        self,
        in_features: int = 512,
        nr_layers: int = 6,
        activation_fn: Activation = Activation.TANH,
        hidden_layersize: int = 512,
        hidden_layers: int = 6,
        hidden_activation_fn: Activation = Activation.SILU,
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        rotate:int=1,
        reducescale=0.01,
        scaleend=1.0
    ) -> None:
        super().__init__()

        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers
        self.s1 = nn.ModuleList()
        self.s2 = nn.ModuleList()
        self.t1 = nn.ModuleList()
        self.t2 = nn.ModuleList()

        for i in range(nr_layers+1):
            features1=int(in_features/2)
            features2=in_features-features1
            self.s2.append(
                FullyConnectedArchCore(
                    in_features=features2,
                    layer_size=hidden_layersize,
                    out_features=features1,
                    nr_layers=hidden_layers,
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    adaptive_activations=adaptive_activations,
                    weight_norm=weight_norm,
                )
            )
            self.t2.append(
                FullyConnectedArchCore(
                    in_features=features2,
                    layer_size=hidden_layersize,
                    out_features=features1,
                    nr_layers=hidden_layers,
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    adaptive_activations=adaptive_activations,
                    weight_norm=weight_norm,
                )
            )
            self.s1.append(
                FullyConnectedArchCore(
                    in_features=features1,
                    layer_size=hidden_layersize,
                    out_features=features2,
                    nr_layers=hidden_layers,
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    adaptive_activations=adaptive_activations,
                    weight_norm=weight_norm,
                )
            )
            self.t1.append(
                FullyConnectedArchCore(
                    in_features=features1,
                    layer_size=hidden_layersize,
                    out_features=features2,
                    nr_layers=hidden_layers,
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    adaptive_activations=adaptive_activations,
                    weight_norm=weight_norm,
                )
            )
        self.nr_layers=nr_layers
        self.initsplit=int(in_features/2)
        self.rotate=rotate
        self.activation1_fn=get_activation_fn(activation_fn,out_features=features2)
        self.activation2_fn=get_activation_fn(activation_fn,out_features=features1)
        self.reducescale=reducescale
        self.scaleend=scaleend
    def forward(self, x: Tensor) -> Tensor:
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        for i in range(self.nr_layers):
            x1 = x1*torch.exp(self.activation2_fn(self.s2[i](x2)*self.reducescale))+self.t2[i](x2)*self.reducescale
            x2 = x2*torch.exp(self.activation1_fn(self.s1[i](x1)*self.reducescale))+self.t1[i](x1)*self.reducescale
            if self.rotate > 0:
                x1_temp=torch.cat((x1[...,self.rotate:],x2[...,:self.rotate]),-1)
                x2=torch.cat((x2[...,self.rotate:],x1[...,:self.rotate]),-1)
                x1=x1_temp
        x1 = x1*torch.exp(self.activation2_fn(self.s2[-1](x2)*self.reducescale))*self.scaleend+self.t2[-1](x2)*self.reducescale
        x2 = x2*torch.exp(self.activation1_fn(self.s1[-1](x1)*self.reducescale))*self.scaleend+self.t1[-1](x1)*self.reducescale
        return torch.cat((x1,x2),-1)
    def reverse(self, x: Tensor) -> Tensor:
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        x2 = (x2-self.t1[-1](x1)*self.reducescale)/self.scaleend*torch.exp(-self.activation1_fn(self.s1[-1](x1)*self.reducescale))
        x1 = (x1-self.t2[-1](x2)*self.reducescale)/self.scaleend*torch.exp(-self.activation2_fn(self.s2[-1](x2)*self.reducescale))
        if self.rotate > 0:
            x1_temp=torch.cat((x2[...,-self.rotate:],x1[...,:-self.rotate]),-1)
            x2=torch.cat((x1[...,-self.rotate:],x2[...,:-self.rotate]),-1)
            x1=x1_temp
        for i in range(1,self.nr_layers):
            x2 = (x2-self.t1[-i-1](x1)*self.reducescale)*torch.exp(-self.activation1_fn(self.s1[-i-1](x1)*self.reducescale))
            x1 = (x1-self.t2[-i-1](x2)*self.reducescale)*torch.exp(-self.activation2_fn(self.s2[-i-1](x2)*self.reducescale))
            if self.rotate > 0:
                x1_temp=torch.cat((x2[...,-self.rotate:],x1[...,:-self.rotate]),-1)
                x2=torch.cat((x1[...,-self.rotate:],x2[...,:-self.rotate]),-1)
                x1=x1_temp
        x2 = (x2-self.t1[0](x1)*self.reducescale)*torch.exp(-self.activation1_fn(self.s1[0](x1)*self.reducescale))
        x1 = (x1-self.t2[0](x2)*self.reducescale)*torch.exp(-self.activation2_fn(self.s2[0](x2)*self.reducescale))
        return torch.cat((x1,x2),-1)
class InheritedInvertibleFullyConnectedArchCore(nn.Module):
    '''
    Architecture core for invertible fully connected neural network with different number of nodes per layer
    following DOI: arXiv:1808.04730
    '''
    def __init__(
        self,
        in_features: int = 512,
        nr_layers: int = 6,
        activation_fn: Activation = Activation.TANH,
        hidden_layersize: int = 512,
        hidden_layers: int = 6,
        hidden_activation_fn: Activation = Activation.SILU,
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        rotate:int=1,
        reducescale=0.01,
        scaleend=1.0,
        extras=0
    ) -> None:
        super().__init__()

        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers
        self.s1 = nn.ModuleList()
        self.s2 = nn.ModuleList()
        self.t1 = nn.ModuleList()
        self.t2 = nn.ModuleList()

        for i in range(nr_layers+1):
            features1=int(in_features/2)
            features2=in_features-features1
            self.s2.append(
                InheritedFullyConnectedFlexiLayerSizeArchCore(
                    in_features=features2+extras,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features1],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.t2.append(
                InheritedFullyConnectedFlexiLayerSizeArchCore(
                    in_features=features2+extras,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features1],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.s1.append(
                InheritedFullyConnectedFlexiLayerSizeArchCore(
                    in_features=features1+extras,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.t1.append(
                InheritedFullyConnectedFlexiLayerSizeArchCore(
                    in_features=features1+extras,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            
        self.nr_layers=nr_layers
        self.initsplit=int(in_features/2)
        self.rotate=rotate
        self.activation1_fn=get_activation_fn(activation_fn,out_features=features2)
        self.activation2_fn=get_activation_fn(activation_fn,out_features=features1)
        self.reducescale=reducescale
        self.scaleend=scaleend
        self.extras=extras
    def forward(self, x: Tensor, w_1:Tensor,w_2:Tensor) -> Tensor:
        #w_1[w_s1,w_t1]
        if self.extras>0:
            x1=x[...,:self.initsplit]
            x2=x[...,self.initsplit:-self.extras]
            extra=x[...,-self.extras:]
        else:
            x1=x[...,:self.initsplit]
            x2=x[...,self.initsplit:]
            extra=x[...,:0]
        for i in range(self.nr_layers):
            x2e=torch.cat((x2,extra),-1)
            x1 = x1*torch.exp(self.activation2_fn(self.s2[i](x2e,w_2[0,i])*self.reducescale))+self.t2[i](x2e,w_2[1,i])*self.reducescale
            x1e=torch.cat((x1,extra),-1)
            x2 = x2*torch.exp(self.activation1_fn(self.s1[i](x1e,w_1[0,i])*self.reducescale))+self.t1[i](x1e,w_1[1,i])*self.reducescale
            if self.rotate > 0:
                x1_temp=torch.cat((x1[...,self.rotate:],x2[...,:self.rotate]),-1)
                x2=torch.cat((x2[...,self.rotate:],x1[...,:self.rotate]),-1)
                x1=x1_temp
        x2e=torch.cat((x2,extra),-1)
        x1 = x1*torch.exp(self.activation2_fn(self.s2[-1](x2e,w_2[0,-1])*self.reducescale))*self.scaleend+self.t2[-1](x2e,w_2[1,-1])*self.reducescale
        x1e=torch.cat((x1,extra),-1)
        x2 = x2*torch.exp(self.activation1_fn(self.s1[-1](x1e,w_1[0,-1])*self.reducescale))*self.scaleend+self.t1[-1](x1e,w_1[1,-1])*self.reducescale
        return torch.cat((x1,x2),-1)
    def reverse(self, x: Tensor, w_1:Tensor,w_2:Tensor) -> Tensor:
        if self.extras>0:
            x1=x[...,:self.initsplit]
            x2=x[...,self.initsplit:-self.extras]
            extra=x[...,-self.extras:]
        else:
            x1=x[...,:self.initsplit]
            x2=x[...,self.initsplit:]
            extra=x[...,:0]
        x1e=torch.cat((x1,extra),-1)
        x2 = (x2-self.t1[-1](x1e,w_1[1,-1])*self.reducescale)/self.scaleend*torch.exp(-self.activation1_fn(self.s1[-1](x1e,w_1[0,-1])*self.reducescale))
        x2e=torch.cat((x2,extra),-1)
        x1 = (x1-self.t2[-1](x2e,w_2[1,-1])*self.reducescale)/self.scaleend*torch.exp(-self.activation2_fn(self.s2[-1](x2e,w_2[0,-1])*self.reducescale))
        if self.rotate > 0:
            x1_temp=torch.cat((x2[...,-self.rotate:],x1[...,:-self.rotate]),-1)
            x2=torch.cat((x1[...,-self.rotate:],x2[...,:-self.rotate]),-1)
            x1=x1_temp
        for i in range(1,self.nr_layers):
            x1e=torch.cat((x1,extra),-1)
            x2 = (x2-self.t1[-i-1](x1e,w_1[1,-i-1])*self.reducescale)*torch.exp(-self.activation1_fn(self.s1[-i-1](x1e,w_1[0,-i-1])*self.reducescale))
            x2e=torch.cat((x2,extra),-1)
            x1 = (x1-self.t2[-i-1](x2e,w_2[1,-i-1])*self.reducescale)*torch.exp(-self.activation2_fn(self.s2[-i-1](x2e,w_2[0,-i-1])*self.reducescale))
            if self.rotate > 0:
                x1_temp=torch.cat((x2[...,-self.rotate:],x1[...,:-self.rotate]),-1)
                x2=torch.cat((x1[...,-self.rotate:],x2[...,:-self.rotate]),-1)
                x1=x1_temp
        x1e=torch.cat((x1,extra),-1)
        x2 = (x2-self.t1[0](x1e,w_1[1,0])*self.reducescale)*torch.exp(-self.activation1_fn(self.s1[0](x1e,w_1[0,0])*self.reducescale))
        x2e=torch.cat((x2,extra),-1)
        x1 = (x1-self.t2[0](x2e,w_2[1,0])*self.reducescale)*torch.exp(-self.activation2_fn(self.s2[0](x2e,w_2[0,0])*self.reducescale))
        return torch.cat((x1,x2),-1)
class InvertibleExtrasFullyConnectedArchCore(nn.Module):
    '''
    Architecture core for invertible fully connected neural network with different number of nodes per layer
    following DOI: arXiv:1808.04730
    '''
    def __init__(
        self,
        in_features: int = 512,
        nr_layers: int = 6,
        activation_fn: Activation = Activation.TANH,
        hidden_layersize: int = 512,
        hidden_layers: int = 6,
        hidden_activation_fn: Activation = Activation.SILU,
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        rotate:int=1,
        reducescale=0.01,
        extras=1
    ) -> None:
        super().__init__()

        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers
        self.s1 = nn.ModuleList()
        self.s2 = nn.ModuleList()
        self.t1 = nn.ModuleList()
        self.t2 = nn.ModuleList()

        for i in range(nr_layers+1):
            features1=int(in_features/2)
            features2=in_features-features1
            self.s2.append(
                FullyConnectedArchCore(
                    in_features=features2+extras,
                    layer_size=hidden_layersize,
                    out_features=features1,
                    nr_layers=hidden_layers,
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    adaptive_activations=adaptive_activations,
                    weight_norm=weight_norm,
                )
            )
            self.t2.append(
                FullyConnectedArchCore(
                    in_features=features2+extras,
                    layer_size=hidden_layersize,
                    out_features=features1,
                    nr_layers=hidden_layers,
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    adaptive_activations=adaptive_activations,
                    weight_norm=weight_norm,
                )
            )
            self.s1.append(
                FullyConnectedArchCore(
                    in_features=features1+extras,
                    layer_size=hidden_layersize,
                    out_features=features2,
                    nr_layers=hidden_layers,
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    adaptive_activations=adaptive_activations,
                    weight_norm=weight_norm,
                )
            )
            self.t1.append(
                FullyConnectedArchCore(
                    in_features=features1+extras,
                    layer_size=hidden_layersize,
                    out_features=features2,
                    nr_layers=hidden_layers,
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    adaptive_activations=adaptive_activations,
                    weight_norm=weight_norm,
                )
            )
        self.nr_layers=nr_layers
        self.initsplit=int(in_features/2)
        self.rotate=rotate
        self.activation1_fn=get_activation_fn(activation_fn,out_features=features2)
        self.activation2_fn=get_activation_fn(activation_fn,out_features=features1)
        self.reducescale=reducescale
    def forward(self, x: Tensor,o: Tensor) -> Tensor:
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        for i in range(self.nr_layers):
            x1 = x1*torch.exp(self.activation2_fn(self.s2[i](torch.cat((x2,o),-1))*self.reducescale))+self.t2[i](torch.cat((x2,o),-1))*self.reducescale
            x2 = x2*torch.exp(self.activation1_fn(self.s1[i](torch.cat((x1,o),-1))*self.reducescale))+self.t1[i](torch.cat((x1,o),-1))*self.reducescale
            if self.rotate > 0:
                x1_temp=torch.cat((x1[...,self.rotate:],x2[...,:self.rotate]),-1)
                x2=torch.cat((x2[...,self.rotate:],x1[...,:self.rotate]),-1)
                x1=x1_temp
        x1 = x1*torch.exp(self.activation2_fn(self.s2[-1](torch.cat((x2,o),-1))*self.reducescale))+self.t2[-1](torch.cat((x2,o),-1))*self.reducescale
        x2 = x2*torch.exp(self.activation1_fn(self.s1[-1](torch.cat((x1,o),-1))*self.reducescale))+self.t1[-1](torch.cat((x1,o),-1))*self.reducescale
        return torch.cat((x1,x2),-1)
    def reverse(self, x: Tensor,o: Tensor) -> Tensor:
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        for i in range(self.nr_layers):
            x2 = (x2-self.t1[-i-1](torch.cat((x1,o),-1))*self.reducescale)*torch.exp(-self.activation1_fn(self.s1[-i-1](torch.cat((x1,o),-1))*self.reducescale))
            x1 = (x1-self.t2[-i-1](torch.cat((x2,o),-1))*self.reducescale)*torch.exp(-self.activation2_fn(self.s2[-i-1](torch.cat((x2,o),-1))*self.reducescale))
            if self.rotate > 0:
                x1_temp=torch.cat((x2[...,-self.rotate:],x1[...,:-self.rotate]),-1)
                x2=torch.cat((x1[...,-self.rotate:],x2[...,:-self.rotate]),-1)
                x1=x1_temp
        x2 = (x2-self.t1[0](torch.cat((x1,o),-1))*self.reducescale)*torch.exp(-self.activation1_fn(self.s1[0](torch.cat((x1,o),-1))*self.reducescale))
        x1 = (x1-self.t2[0](torch.cat((x2,o),-1))*self.reducescale)*torch.exp(-self.activation2_fn(self.s2[0](torch.cat((x2,o),-1))*self.reducescale))
        return torch.cat((x1,x2),-1)
class Fixed0FourierApply(nn.Module):
    '''
    Architecture core for invertible fully connected neural network with different number of nodes per layer
    following DOI: arXiv:1808.04730
    '''
    def __init__(
        self,
        varNumber,
        omega=1.,
        FourierNumber=4,
    ) -> None:
        super().__init__()
        self.varNumber=varNumber
        torchrangeomega=torch.tensor(omega*np.repeat(np.arange(1,FourierNumber+1),varNumber).reshape((1,-1)),dtype=torch.float)
        self.register_buffer("torchrangeomega", torchrangeomega, persistent=False)
        self.Nfourier=int(FourierNumber)
        self.Nfourier2=int(FourierNumber*2)
    def forward(self,x,o):#Fcossin1, o:deltaT
        a0radian=o*self.torchrangeomega
        a0_uvw=(torch.cat((torch.cos(a0radian),torch.sin(a0radian)),dim=-1)*x).reshape(x.size(0),self.Nfourier2,self.varNumber).sum(dim=-2)
        fwd_uvw=x[...,:self.Nfourier*self.varNumber].reshape(x.size(0),self.Nfourier,self.varNumber).sum(dim=-2)
        return a0_uvw-fwd_uvw
class Fixed0FourierInvertibleFullyConnectedArchCore(nn.Module):
    '''
    Architecture core for invertible fully connected neural network with different number of nodes per layer
    following DOI: arXiv:1808.04730
    '''
    def __init__(
        self,
        in_features: int = 512,
        nr_layers: int = 6,
        activation_fn: Activation = Activation.TANH,
        hidden_layersize: int = 512,
        hidden_layers: int = 6,
        hidden_activation_fn: Activation = Activation.SILU,
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        rotate:int=1,
        reducescale=0.01,
        omega=1.,
        FourierNumber=4,
    ) -> None:
        super().__init__()

        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers
        features1=int(in_features/2)
        features2=in_features-features1
        self.Fixed0Fourier1=Fixed0FourierApply(features2,omega,FourierNumber)
        self.Fixed0Fourier2=Fixed0FourierApply(features1,omega,FourierNumber)
        self.s1 = nn.ModuleList()
        self.s2 = nn.ModuleList()
        self.t1 = nn.ModuleList()
        self.t2 = nn.ModuleList()
        for i in range(nr_layers+1):
            self.s2.append(
                FullyConnectedFlexiLayerSizeArchCore(
                    in_features=features2+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features1*FourierNumber*2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.t2.append(
                FullyConnectedFlexiLayerSizeArchCore(
                    in_features=features2+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features1*FourierNumber*2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.s1.append(
                FullyConnectedFlexiLayerSizeArchCore(
                    in_features=features1+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features2*FourierNumber*2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.t1.append(
                FullyConnectedFlexiLayerSizeArchCore(
                    in_features=features1+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features2*FourierNumber*2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
        self.nr_layers=nr_layers
        self.initsplit=int(in_features/2)
        self.rotate=rotate
        self.activation1_fn=get_activation_fn(activation_fn,out_features=features2)
        self.activation2_fn=get_activation_fn(activation_fn,out_features=features1)
        self.reducescale=reducescale
        omega=torch.tensor(omega,dtype=torch.float)
        self.register_buffer("omega", omega, persistent=False)
    def forward(self, x: Tensor,o: Tensor) -> Tensor: #o:t,deltaT
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        sint=torch.sin(self.omega*o[...,0:1])
        cost=torch.cos(self.omega*o[...,0:1])
        for i in range(self.nr_layers):
            x1 = x1*torch.exp(self.Fixed0Fourier2(self.activation2_fn(self.s2[i](torch.cat((x2,sint,cost),-1))*self.reducescale),o[...,1:2]))+self.Fixed0Fourier2(self.t2[i](torch.cat((x2,sint,cost),-1))*self.reducescale,o[...,1:2])
            x2 = x2*torch.exp(self.Fixed0Fourier1(self.activation1_fn(self.s1[i](torch.cat((x1,sint,cost),-1))*self.reducescale),o[...,1:2]))+self.Fixed0Fourier1(self.t1[i](torch.cat((x1,sint,cost),-1))*self.reducescale,o[...,1:2])
            if self.rotate > 0:
                x1_temp=torch.cat((x1[...,self.rotate:],x2[...,:self.rotate]),-1)
                x2=torch.cat((x2[...,self.rotate:],x1[...,:self.rotate]),-1)
                x1=x1_temp
        x1 = x1*torch.exp(self.Fixed0Fourier2(self.activation2_fn(self.s2[-1](torch.cat((x2,sint,cost),-1))*self.reducescale),o[...,1:2]))+self.Fixed0Fourier2(self.t2[-1](torch.cat((x2,sint,cost),-1))*self.reducescale,o[...,1:2])
        x2 = x2*torch.exp(self.Fixed0Fourier1(self.activation1_fn(self.s1[-1](torch.cat((x1,sint,cost),-1))*self.reducescale),o[...,1:2]))+self.Fixed0Fourier1(self.t1[-1](torch.cat((x1,sint,cost),-1))*self.reducescale,o[...,1:2])
        return torch.cat((x1,x2),-1)
    def reverse(self, x: Tensor,o: Tensor) -> Tensor:
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        sint=torch.sin(self.omega*o[...,0:1])
        cost=torch.cos(self.omega*o[...,0:1])
        for i in range(self.nr_layers):
            x2 = (x2-self.Fixed0Fourier1(self.t1[-i-1](torch.cat((x1,sint,cost),-1))*self.reducescale,o[...,1:2]))*torch.exp(-self.Fixed0Fourier1(self.activation1_fn(self.s1[-i-1](torch.cat((x1,sint,cost),-1))*self.reducescale),o[...,1:2]))
            x1 = (x1-self.Fixed0Fourier2(self.t2[-i-1](torch.cat((x2,sint,cost),-1))*self.reducescale,o[...,1:2]))*torch.exp(-self.Fixed0Fourier2(self.activation2_fn(self.s2[-i-1](torch.cat((x2,sint,cost),-1))*self.reducescale),o[...,1:2]))
            if self.rotate > 0:
                x1_temp=torch.cat((x2[...,-self.rotate:],x1[...,:-self.rotate]),-1)
                x2=torch.cat((x1[...,-self.rotate:],x2[...,:-self.rotate]),-1)
                x1=x1_temp
        x2 = (x2-self.Fixed0Fourier1(self.t1[0](torch.cat((x1,sint,cost),-1))*self.reducescale,o[...,1:2]))*torch.exp(-self.Fixed0Fourier1(self.activation1_fn(self.s1[0](torch.cat((x1,sint,cost),-1))*self.reducescale),o[...,1:2]))
        x1 = (x1-self.Fixed0Fourier2(self.t2[0](torch.cat((x2,sint,cost),-1))*self.reducescale,o[...,1:2]))*torch.exp(-self.Fixed0Fourier2(self.activation2_fn(self.s2[0](torch.cat((x2,sint,cost),-1))*self.reducescale),o[...,1:2]))
        return torch.cat((x1,x2),-1)
class FourierInvertibleFullyConnectedArchCore(nn.Module):
    '''
    Architecture core for invertible fully connected neural network with different number of nodes per layer
    following DOI: arXiv:1808.04730
    '''
    def __init__(
        self,
        in_features: int = 512,
        nr_layers: int = 6,
        activation_fn: Activation = Activation.TANH,
        hidden_layersize: int = 512,
        hidden_layers: int = 6,
        hidden_activation_fn: Activation = Activation.SILU,
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        rotate:int=1,
        reducescale=0.01,
        omega=1.,
    ) -> None:
        super().__init__()

        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers
        features1=int(in_features/2)
        features2=in_features-features1
        omega=torch.tensor(omega,dtype=torch.float)
        self.register_buffer("omega", omega, persistent=False)
        self.s1 = nn.ModuleList()
        self.s2 = nn.ModuleList()
        self.t1 = nn.ModuleList()
        self.t2 = nn.ModuleList()
        for i in range(nr_layers+1):
            self.s2.append(
                FullyConnectedFlexiLayerSizeArchCore(
                    in_features=features2+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features1],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.t2.append(
                FullyConnectedFlexiLayerSizeArchCore(
                    in_features=features2+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features1],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.s1.append(
                FullyConnectedFlexiLayerSizeArchCore(
                    in_features=features1+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.t1.append(
                FullyConnectedFlexiLayerSizeArchCore(
                    in_features=features1+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
        self.nr_layers=nr_layers
        self.initsplit=int(in_features/2)
        self.rotate=rotate
        self.activation1_fn=get_activation_fn(activation_fn,out_features=features2)
        self.activation2_fn=get_activation_fn(activation_fn,out_features=features1)
        self.reducescale=reducescale
    def forward(self, x: Tensor,o: Tensor) -> Tensor: #o:t
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        sint=torch.sin(self.omega*o[...,0:1])
        cost=torch.cos(self.omega*o[...,0:1])
        for i in range(self.nr_layers):
            x1 = x1*torch.exp(self.activation2_fn(self.s2[i](torch.cat((x2,sint,cost),-1))*self.reducescale))+self.t2[i](torch.cat((x2,sint,cost),-1))*self.reducescale
            x2 = x2*torch.exp(self.activation1_fn(self.s1[i](torch.cat((x1,sint,cost),-1))*self.reducescale))+self.t1[i](torch.cat((x1,sint,cost),-1))*self.reducescale
            if self.rotate > 0:
                x1_temp=torch.cat((x1[...,self.rotate:],x2[...,:self.rotate]),-1)
                x2=torch.cat((x2[...,self.rotate:],x1[...,:self.rotate]),-1)
                x1=x1_temp
        x1 = x1*torch.exp(self.activation2_fn(self.s2[-1](torch.cat((x2,sint,cost),-1))*self.reducescale))+self.t2[-1](torch.cat((x2,sint,cost),-1))*self.reducescale
        x2 = x2*torch.exp(self.activation1_fn(self.s1[-1](torch.cat((x1,sint,cost),-1))*self.reducescale))+self.t1[-1](torch.cat((x1,sint,cost),-1))*self.reducescale
        return torch.cat((x1,x2),-1)
    def reverse(self, x: Tensor,o: Tensor) -> Tensor:
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        sint=torch.sin(self.omega*o[...,0:1])
        cost=torch.cos(self.omega*o[...,0:1])
        for i in range(self.nr_layers):
            x2 = (x2-self.t1[-i-1](torch.cat((x1,sint,cost),-1))*self.reducescale)*torch.exp(-self.activation1_fn(self.s1[-i-1](torch.cat((x1,sint,cost),-1))*self.reducescale))
            x1 = (x1-self.t2[-i-1](torch.cat((x2,sint,cost),-1))*self.reducescale)*torch.exp(-self.activation2_fn(self.s2[-i-1](torch.cat((x2,sint,cost),-1))*self.reducescale))
            if self.rotate > 0:
                x1_temp=torch.cat((x2[...,-self.rotate:],x1[...,:-self.rotate]),-1)
                x2=torch.cat((x1[...,-self.rotate:],x2[...,:-self.rotate]),-1)
                x1=x1_temp
        x2 = (x2-self.t1[0](torch.cat((x1,sint,cost),-1))*self.reducescale)*torch.exp(-self.activation1_fn(self.s1[0](torch.cat((x1,sint,cost),-1))*self.reducescale))
        x1 = (x1-self.t2[0](torch.cat((x2,sint,cost),-1))*self.reducescale)*torch.exp(-self.activation2_fn(self.s2[0](torch.cat((x2,sint,cost),-1))*self.reducescale))
        return torch.cat((x1,x2),-1)
class InheritedFixed0FourierInvertibleFullyConnectedArchCore(nn.Module):
    '''
    Architecture core for invertible fully connected neural network with different number of nodes per layer
    following DOI: arXiv:1808.04730
    '''
    def __init__(
        self,
        in_features: int = 512,
        nr_layers: int = 6,
        activation_fn: Activation = Activation.TANH,
        hidden_layersize: int = 512,
        hidden_layers: int = 6,
        hidden_activation_fn: Activation = Activation.SILU,
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        rotate:int=1,
        reducescale=0.01,
        omega=1.,
        FourierNumber=4,
    ) -> None:
        super().__init__()

        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers
        features1=int(in_features/2)
        features2=in_features-features1
        self.Fixed0Fourier1=Fixed0FourierApply(features2,omega,FourierNumber)
        self.Fixed0Fourier2=Fixed0FourierApply(features1,omega,FourierNumber)
        self.s1 = nn.ModuleList()
        self.s2 = nn.ModuleList()
        self.t1 = nn.ModuleList()
        self.t2 = nn.ModuleList()
        for i in range(nr_layers+1):
            self.s2.append(
                InheritedFullyConnectedFlexiLayerSizeArchCore(
                    in_features=features2+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features1*FourierNumber*2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.t2.append(
                InheritedFullyConnectedFlexiLayerSizeArchCore(
                    in_features=features2+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features1*FourierNumber*2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.s1.append(
                InheritedFullyConnectedFlexiLayerSizeArchCore(
                    in_features=features1+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features2*FourierNumber*2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
            self.t1.append(
                InheritedFullyConnectedFlexiLayerSizeArchCore(
                    in_features=features1+2,
                    layer_sizeList=[hidden_layersize]*hidden_layers+[features2*FourierNumber*2],
                    skip_connections=skip_connections,
                    activation_fn=hidden_activation_fn,
                    weight_norm=weight_norm,
                )
            )
        self.nr_layers=nr_layers
        self.initsplit=int(in_features/2)
        self.rotate=rotate
        self.activation1_fn=get_activation_fn(activation_fn,out_features=features2)
        self.activation2_fn=get_activation_fn(activation_fn,out_features=features1)
        self.reducescale=reducescale
        omega=torch.tensor(omega,dtype=torch.float)
        self.register_buffer("omega", omega, persistent=False)
    def forward(self, x: Tensor,o: Tensor) -> Tensor: #o:t,deltaT
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        sint=torch.sin(self.omega*o[...,0:1])
        cost=torch.cos(self.omega*o[...,0:1])
        for i in range(self.nr_layers):
            x1 = x1*torch.exp(self.Fixed0Fourier2(self.activation2_fn(self.s2[i](torch.cat((x2,sint,cost),-1))*self.reducescale),o[...,1:2]))+self.Fixed0Fourier2(self.t2[i](torch.cat((x2,sint,cost),-1))*self.reducescale,o[...,1:2])
            x2 = x2*torch.exp(self.Fixed0Fourier1(self.activation1_fn(self.s1[i](torch.cat((x1,sint,cost),-1))*self.reducescale),o[...,1:2]))+self.Fixed0Fourier1(self.t1[i](torch.cat((x1,sint,cost),-1))*self.reducescale,o[...,1:2])
            if self.rotate > 0:
                x1_temp=torch.cat((x1[...,self.rotate:],x2[...,:self.rotate]),-1)
                x2=torch.cat((x2[...,self.rotate:],x1[...,:self.rotate]),-1)
                x1=x1_temp
        x1 = x1*torch.exp(self.Fixed0Fourier2(self.activation2_fn(self.s2[-1](torch.cat((x2,sint,cost),-1))*self.reducescale),o[...,1:2]))+self.Fixed0Fourier2(self.t2[-1](torch.cat((x2,sint,cost),-1))*self.reducescale,o[...,1:2])
        x2 = x2*torch.exp(self.Fixed0Fourier1(self.activation1_fn(self.s1[-1](torch.cat((x1,sint,cost),-1))*self.reducescale),o[...,1:2]))+self.Fixed0Fourier1(self.t1[-1](torch.cat((x1,sint,cost),-1))*self.reducescale,o[...,1:2])
        return torch.cat((x1,x2),-1)
    def reverse(self, x: Tensor,o: Tensor,t1weights: Tensor) -> Tensor:##!!inherited not coded
        x1=x[...,:self.initsplit]
        x2=x[...,self.initsplit:]
        sint=torch.sin(self.omega*o[...,0:1])
        cost=torch.cos(self.omega*o[...,0:1])
        for i in range(self.nr_layers):
            x2 = (x2-self.Fixed0Fourier1(self.t1[-i-1](torch.cat((x1,sint,cost),-1))*self.reducescale,o[...,1:2]))*torch.exp(-self.Fixed0Fourier1(self.activation1_fn(self.s1[-i-1](torch.cat((x1,sint,cost),-1))*self.reducescale),o[...,1:2]))
            x1 = (x1-self.Fixed0Fourier2(self.t2[-i-1](torch.cat((x2,sint,cost),-1))*self.reducescale,o[...,1:2]))*torch.exp(-self.Fixed0Fourier2(self.activation2_fn(self.s2[-i-1](torch.cat((x2,sint,cost),-1))*self.reducescale),o[...,1:2]))
            if self.rotate > 0:
                x1_temp=torch.cat((x2[...,-self.rotate:],x1[...,:-self.rotate]),-1)
                x2=torch.cat((x1[...,-self.rotate:],x2[...,:-self.rotate]),-1)
                x1=x1_temp
        x2 = (x2-self.Fixed0Fourier1(self.t1[0](torch.cat((x1,sint,cost),-1))*self.reducescale,o[...,1:2]))*torch.exp(-self.Fixed0Fourier1(self.activation1_fn(self.s1[0](torch.cat((x1,sint,cost),-1))*self.reducescale),o[...,1:2]))
        x1 = (x1-self.Fixed0Fourier2(self.t2[0](torch.cat((x2,sint,cost),-1))*self.reducescale,o[...,1:2]))*torch.exp(-self.Fixed0Fourier2(self.activation2_fn(self.s2[0](torch.cat((x2,sint,cost),-1))*self.reducescale),o[...,1:2]))
        return torch.cat((x1,x2),-1)
class Intensity3DGradientArchCore(nn.Module):
    '''
    Architecture core for Intensity3DGradient of image
    '''
    def __init__(
        self,
        image,
        extend = 5,
        multiplier: float=1.#with one it is +ve direction - -ve direction
    ) -> None:
        super().__init__()
        if not(isinstance(extend,list)):
            extend=[extend,extend,extend]
        extend=extend[::-1]
        image=torch.tensor(image[np.newaxis,np.newaxis,...],dtype=torch.float)
        self.register_buffer("image", image, persistent=False)
        imageshape=torch.tensor(image.shape,dtype=torch.int)
        self.register_buffer("imageshape", imageshape, persistent=False)
        extend=(np.mgrid[(-extend[0]+0.5):extend[0],(-extend[1]+0.5):extend[1],(-extend[2]+0.5):extend[2]].T/np.array(image.shape).reshape((1,1,1,-1))*2.).reshape((1,1,1,-1,3))
        xplus=(extend[...,0:1]>=0).astype(float)*2.-1.
        yplus=(extend[...,1:2]>=0).astype(float)*2.-1.
        zplus=(extend[...,2:3]>=0).astype(float)*2.-1.
        extend=torch.tensor(extend,dtype=torch.float)
        self.register_buffer("extend", extend, persistent=False)
        xplus=torch.tensor(xplus,dtype=torch.float)
        self.register_buffer("xplus", xplus, persistent=False)
        yplus=torch.tensor(yplus,dtype=torch.float)
        self.register_buffer("yplus", xplus, persistent=False)
        zplus=torch.tensor(zplus,dtype=torch.float)
        self.register_buffer("zplus", zplus, persistent=False)
        multiplier=torch.tensor(multiplier,dtype=torch.float)
        self.register_buffer("multiplier", multiplier, persistent=False)
    def forward(self, x: Tensor, o: Tensor) -> Tensor:
        grid=x.unsqueeze(1).unsqueeze(0).unsqueeze(0)+self.extend
        surrounding=torch.nn.functional.grid_sample(self.image, grid, mode='bilinear', padding_mode='zeros', align_corners=None)
        surrounding_x=torch.sum(surrounding*self.xplus,dim=(3,4))*self.multiplier
        surrounding_y=torch.sum(surrounding*self.yplus,dim=(3,4))*self.multiplier
        surrounding_z=torch.sum(surrounding*self.zplus,dim=(3,4))*self.multiplier
        xyz_grad=torch.stack([surrounding_x[0,0],surrounding_y[0,0],surrounding_z[0,0]],dim=-1)
        return xyz_grad*o
class MaxPoolArchCore(nn.Module):
    '''
    Architecture core for max pool function
    '''
    def __init__(
        self,
        pooldim: int = 1,
        keepdim: bool = False,
        keepview: bool = False
    ) -> None:
        super().__init__()
        self.pooldim=pooldim
        self.keepdim=keepdim
        self.keepview=keepview
    def forward(self, x: Tensor) -> Tensor:
        input_max, max_indices=torch.max(x,self.pooldim,keepdim=self.keepdim)
        if self.keepview and self.keepdim:
            input_max=input_max.expand(x.size())
        return input_max
class MaxFeatureIndicesArchCore(nn.Module):
    '''
    Architecture core for extracting max pool point index
    '''
    def __init__(
        self,
        pooldim: int = 1,
        k: int = 1,
    ) -> None:
        super().__init__()
        self.pooldim=pooldim
        self.k=k
    def forward(self, x: Tensor) -> Tensor:
        input_max, max_indices=torch.topk(x,self.k,dim=self.pooldim, largest=True)
        input_min, min_indices=torch.topk(x,self.k,dim=self.pooldim ,largest=False)
        indices=torch.flatten(torch.cat((max_indices,min_indices),dim=1))
        return indices
class MinPoolArchCore(nn.Module):
    '''
    Architecture core for min pool function
    '''
    def __init__(
        self,
        pooldim: int = 1,
        keepdim: bool = False,
        keepview: bool = False
    ) -> None:
        super().__init__()
        self.pooldim=pooldim
        self.keepdim=keepdim
        self.keepview=keepview
    def forward(self, x: Tensor) -> Tensor:
        input_min, min_indices=torch.min(x,self.pooldim,keepdim=self.keepdim)
        if self.keepview and self.keepdim:
            input_min=input_min.expand(x.size())
        return input_min
class MeanPoolArchCore(nn.Module):
    '''
    Architecture core for mean pool function
    '''
    def __init__(
        self,
        pooldim: int = 1,
        keepdim: bool = False,
        keepview: bool = False
    ) -> None:
        super().__init__()
        self.pooldim=pooldim
        self.keepdim=keepdim
        self.keepview=keepview
    def forward(self, x: Tensor) -> Tensor:
        input_mean=torch.mean(x,self.pooldim,keepdim=self.keepdim)
        if self.keepview and self.keepdim:
            input_mean=input_mean.expand(x.size())
        return input_mean
class SumPoolArchCore(nn.Module):
    '''
    Architecture core for sum pool function
    '''
    def __init__(
        self,
        pooldim: int = 1,
        keepdim: bool = False,
        keepview: bool = False
    ) -> None:
        super().__init__()
        self.pooldim=pooldim
        self.keepdim=keepdim
        self.keepview=keepview
    def forward(self, x: Tensor) -> Tensor:
        input_mean=torch.sum(x,self.pooldim,keepdim=self.keepdim)
        if self.keepview and self.keepdim:
            input_mean=input_mean.expand(x.size())
        return input_mean
class HypernetFullyConnectedFlexiLayerSizeArchCore(nn.Module):
    '''
    Architecture core Hypernetwork
    '''
    def __init__(
        self,
        in_features: int = 512,
        layer_sizeList: Union[int,List[int],None] = None,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        with_caseID: int = 0,
    ) -> None:
        super().__init__()
        self.in_features=in_features
        self.skip_connections = skip_connections
        if isinstance(layer_sizeList,int):
            layer_sizeList=[layer_sizeList]
        nr_layers=len(layer_sizeList)-1
        # Allows for regular linear layers to be swapped for 1D Convs
        # Useful for channel operations in FNO/Transformers

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * nr_layers
        if len(activation_fn) < nr_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                nr_layers - len(activation_fn)
            )

        self.layers = nn.ModuleList()

        layer_in_features = in_features
        for i in range(nr_layers):
            self.layers.append(
                FCLayer(
                    layer_in_features,
                    layer_sizeList[i],
                    activation_fn[i],
                    weight_norm,
                    activation_par,
                )
            )
            layer_in_features = layer_sizeList[i]

        self.final_layer = FCLayer(
            in_features=layer_in_features,
            out_features=layer_sizeList[-1],
            activation_fn=Activation.IDENTITY,
            weight_norm=False,
            activation_par=None,
        )
        if with_caseID:
            self.CaseIDArch = nn.ModuleList()
            self.CaseIDArch.append(CaseIDArchCore(with_caseID))
            self.consolidative_matmul=Consolidative_matmul.apply
            self.with_caseID=True
        else:
            self.with_caseID=False
    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        if self.with_caseID:
            caseID=x[...,self.in_features:(self.in_features+1)].detach()
            caseID_unit=torch.transpose(self.CaseIDArch[0](caseID),-1,-2)#case_num,training_points
            case_inputs=self.consolidative_matmul(caseID_unit,x[...,0:self.in_features])
        else:
            case_inputs=x[...,0:self.in_features]
        
        for i, layer in enumerate(self.layers):
            case_inputs = layer(case_inputs)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    case_inputs, x_skip = case_inputs + x_skip, case_inputs
                else:
                    x_skip = case_inputs
        case_inputs=self.final_layer(case_inputs)
        if self.with_caseID:
            case_inputs=torch.matmul(torch.transpose(caseID_unit,-1,-2),case_inputs)
        
        return case_inputs

class CaseIDtoFeatureArchCore(nn.Module):
    '''
    Architecture core to store feature array based on case index
    '''
    def __init__(
        self,
        feature_array
    ) -> None:
        super().__init__()
        self.out_features=feature_array.shape[-1]
        self.total_case=feature_array.shape[0]
        feature_array = torch.tensor(feature_array,dtype=torch.float)##!!!QUICK FIXED
        self.register_buffer("feature_array", feature_array, persistent=False)
        self.unit_function=Unit_function.apply
        case_range = torch.tensor(np.arange(self.total_case).reshape((1,-1)))
        self.register_buffer("case_range", case_range, persistent=False)
    def forward(self, x: Tensor) -> Tensor:
        caseMatrix=self.unit_function(x-self.case_range)
        return torch.matmul(caseMatrix,self.feature_array)

    def extra_repr(self) -> str:
        return "out_features={}".format(
            self.out_features
        )
class FixedFeatureArchCore(nn.Module):
    '''
    Architecture core to store feature array
    '''
    def __init__(
        self,
        feature_array
    ) -> None:
        super().__init__()
        self.out_features=feature_array.shape[-1]
        self.total_case=feature_array.shape[0]
        feature_array = torch.tensor(feature_array,dtype=torch.float)##!!!QUICK FIXED
        self.register_buffer("feature_array", feature_array, persistent=False)
    def forward(self) -> Tensor:
        return self.feature_array

    def extra_repr(self) -> str:
        return "out_features={}".format(
            self.out_features
        )
def multinomial(sampleweights,samplenumber):
    x=torch.cumsum(sampleweights,dim=1)
    x=x/x[:,-1:]
    
    return torch.randint(0,sampleweights.size(1),(sampleweights.size(0),samplenumber),device=sampleweights.device)
class NominalSampledFixedFeatureArchCore(nn.Module):
    '''
    Architecture core to store and sample feature array and keep some as a fixed sample
    '''
    def __init__(
        self,
        feature_array,
    ) -> None:
        super().__init__()
        #assume feature_array has rank3
        self.out_features=feature_array.shape[-1]
        self.total_case=feature_array.shape[0]
        feature_array = torch.tensor(feature_array,dtype=torch.float)##!!!QUICK FIXED
        self.register_buffer("feature_array", feature_array, persistent=False)
        fixed_ind=torch.randint(0,self.feature_array.size(1),(self.feature_array.size(0),self.out_features))
        fixed_ind=torch.cat([torch.index_select(feature_array[row:row+1],1,yind) for row, yind in enumerate(fixed_ind)]).detach()
        self.register_buffer("fixed_ind", fixed_ind, persistent=True)
    def forward(self) -> Tensor:
        return self.fixed_ind
    def allpts(self):
        return self.feature_array
    def updateSampleWeight(self,indfrom_samples):
        self.fixed_ind=torch.cat([torch.index_select(self.feature_array[row:row+1],1,yind) for row, yind in enumerate(indfrom_samples)]).detach()
    def extra_repr(self) -> str:
        return "out_features={}".format(
            self.out_features
        )
    
###autograd functions
class Consolidative_matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,select_mat, x):
        ctx.save_for_backward(select_mat)
        select_sum=torch.sum(select_mat, -1, keepdim=True)
        select_sum=(torch.nn.functional.relu(select_sum-1.)+1.).detach()
        return torch.matmul(select_mat,x)/select_sum
    @staticmethod
    def backward(ctx, grad_out):
        select_mat, = ctx.saved_tensors
        return grad_out*0. , torch.matmul(torch.transpose(select_mat,-1,-2),grad_out)
class Unit_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y=torch.ones_like(x)
        y[x<=-0.5]=0.
        y[x>=0.5]=0.
        return y.detach()
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out*0.
def Heaviside_function(input ,values):
    y=torch.zeros_like(input)
    y[input>0.]=1.
    y[input==0.]=values
    return y
class BsplineBasis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        B0 = Heaviside_function(x-1., 0.)*Heaviside_function(-x+2., 0.)*(2.-x)**3./6.
        B1 = Heaviside_function(x, 1.)*Heaviside_function(-x+1., 1.)*(3.*x**3.-6.*x**2.+4.)/6. 
        B2 = Heaviside_function(x+1., 0.)*Heaviside_function(-x, 1.)*(-3.*x**3.-6.*x**2.+4.)/6. 
        B3 = Heaviside_function(x+2., 0.)*Heaviside_function(-x-1., 0.)*(x+2.)**3./6.
        ctx.save_for_backward(x)
        return B0+B1+B2+B3
    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        B0=Heaviside_function(x-1., 0.)*Heaviside_function(-x+2., 0.)*(2.-x)**2./-2.
        B1=Heaviside_function(x, 1.)*Heaviside_function(-x+1., 1.)*(3.*x**2.-4.*x**2.)/2. 
        B2=Heaviside_function(x+1., 0.)*Heaviside_function(-x, 1.)*(-3.*x**2.-4.*x**2.)/2. 
        B3=Heaviside_function(x+2., 0.)*Heaviside_function(-x-1., 0.)*(x+2.)**2./2.
        return grad_out*(B0+B1+B2+B3)
class Bridge_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x,xmin,xmax):
        y=torch.ones_like(x)
        y[x<=xmin]=0.
        y[x>xmax]=0.
        return y
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out*0.,grad_out*0.,grad_out*0.
#Bridge=Bridge_function.apply
def Bridge(x,xmin,xmax):
    y=torch.ones_like(x)
    y[x<=xmin]=0.
    y[x>xmax]=0.
    return y
#Bspline1D=BsplineBasis.apply
def Bspline1D(x):
    B0 = Bridge(x, 1.,2.)*(2.-x)**3./6.
    B1 = Bridge(x, 0.,1.)*(3.*x**3.-6.*x**2.+4.)/6. 
    B2 = Bridge(x, -1.,0.)*(-3.*x**3.-6.*x**2.+4.)/6. 
    B3 = Bridge(x, -2.,-1.)*(x+2.)**3./6.
    return B0+B1+B2+B3
class Bspline2DArchCore(nn.Module):
    def __init__(
        self,
        origin:List[float],
        spacing:List[float],
        nodes_shape:List[int],
        fix_coef=None
    ) -> None:
        super().__init__()
        origin=torch.tensor(np.array(origin).reshape((1,-1)))
        self.register_buffer("origin", origin, persistent=False)
        spacing=torch.tensor(np.array(spacing).reshape((1,-1)))
        self.register_buffer("spacing", spacing, persistent=False)
        self.nodes_shape=nodes_shape
        nodes_coord=np.mgrid[0:self.nodes_shape[0],0:self.nodes_shape[1]]
        shift_coord_x=torch.tensor(nodes_coord[0].reshape((1,-1)))
        self.register_buffer("shift_coord_x", shift_coord_x, persistent=False)
        shift_coord_y=torch.tensor(nodes_coord[1].reshape((1,-1)))
        self.register_buffer("shift_coord_y", shift_coord_y, persistent=False)
        if fix_coef is None:
            self.weight_u = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.weight_v = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.reset_parameters()
        elif len(fix_coef)==len(nodes_shape):
            weight_u=torch.tensor(np.array(fix_coef[0]).reshape((1,-1)))
            self.register_buffer("weight_u", weight_u, persistent=False)
            weight_v=torch.tensor(np.array(fix_coef[1]).reshape((1,-1)))
            self.register_buffer("weight_v", weight_v, persistent=False)
        else:
            weight_u=torch.tensor(np.array(fix_coef[...,0]).reshape((1,-1)))
            self.register_buffer("weight_u", weight_u, persistent=False)
            weight_v=torch.tensor(np.array(fix_coef[...,1]).reshape((1,-1)))
            self.register_buffer("weight_v", weight_v, persistent=False)
            
    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight_u)
        torch.nn.init.xavier_uniform_(self.weight_v)

    def forward(self, x: Tensor) -> Tensor:
        BsplineCoord=(x-self.origin)/self.spacing
        shifted_weight=Bspline1D(BsplineCoord[...,0:1]-self.shift_coord_x)*Bspline1D(BsplineCoord[...,1:2]-self.shift_coord_y)
        u=torch.sum(shifted_weight*self.weight_u,-1, keepdim=True)
        v=torch.sum(shifted_weight*self.weight_v,-1, keepdim=True)
        return torch.cat((u,v),-1)
class ParametricBspline2DArchCore(nn.Module):
    def __init__(
        self,
        origin:List[float],
        spacing:List[float],
        nodes_shape:List[int],
        separateCoef:bool = False
    ) -> None:
        super().__init__()
        origin=torch.tensor(np.array(origin).reshape((1,-1)))
        self.register_buffer("origin", origin, persistent=False)
        spacing=torch.tensor(np.array(spacing).reshape((1,-1)))
        self.register_buffer("spacing", spacing, persistent=False)
        self.nodes_shape=nodes_shape
        nodes_coord=np.mgrid[0:self.nodes_shape[0],0:self.nodes_shape[1]]
        shift_coord_x=torch.tensor(nodes_coord[0].reshape((1,-1)))
        self.register_buffer("shift_coord_x", shift_coord_x, persistent=False)
        shift_coord_y=torch.tensor(nodes_coord[1].reshape((1,-1)))
        self.register_buffer("shift_coord_y", shift_coord_y, persistent=False)
        total=int(np.prod(nodes_shape))*2
        if separateCoef:
            self.u_start=0
            self.u_stop=int(total/2)
            self.u_skip=1
            self.v_start=int(total/2)
            self.v_stop=total
            self.v_skip=1
        else:
            self.u_start=0
            self.u_stop=total
            self.u_skip=2
            self.v_start=1
            self.v_stop=total
            self.v_skip=2
        '''
        if separateCoef:
            u_start=torch.tensor(0,dtype=torch.long)
            self.register_buffer("u_start", u_start, persistent=False)
            u_stop=torch.tensor(int(total/2),dtype=torch.long)
            self.register_buffer("u_stop", u_stop, persistent=False)
            u_skip=torch.tensor(1,dtype=torch.long)
            self.register_buffer("u_skip", u_skip, persistent=False)
            v_start=torch.tensor(int(total/2),dtype=torch.long)
            self.register_buffer("v_start", v_start, persistent=False)
            v_stop=torch.tensor(total,dtype=torch.long)
            self.register_buffer("v_stop", v_stop, persistent=False)
            v_skip=torch.tensor(1,dtype=torch.long)
            self.register_buffer("v_skip", v_skip, persistent=False)
        else:
            u_start=torch.tensor(0,dtype=torch.long)
            self.register_buffer("u_start", u_start, persistent=False)
            u_stop=torch.tensor(total,dtype=torch.long)
            self.register_buffer("u_stop", u_stop, persistent=False)
            u_skip=torch.tensor(2,dtype=torch.long)
            self.register_buffer("u_skip", u_skip, persistent=False)
            v_start=torch.tensor(1,dtype=torch.long)
            self.register_buffer("v_start", v_start, persistent=False)
            v_stop=torch.tensor(total,dtype=torch.long)
            self.register_buffer("v_stop", v_stop, persistent=False)
            v_skip=torch.tensor(2,dtype=torch.long)
            self.register_buffer("v_skip", v_skip, persistent=False)
        '''
    def forward(self, x: Tensor, o: Tensor) -> Tensor:
        BsplineCoord=(x-self.origin)/self.spacing
        shifted_weight=Bspline1D(BsplineCoord[...,0:1]-self.shift_coord_x)*Bspline1D(BsplineCoord[...,1:2]-self.shift_coord_y)
        u=torch.sum(shifted_weight*o[...,self.u_start:self.u_stop:self.u_skip],-1, keepdim=True)
        v=torch.sum(shifted_weight*o[...,self.v_start:self.v_stop:self.v_skip],-1, keepdim=True)
        return torch.cat((u,v),-1)
class Bspline3DArchCore(nn.Module):
    def __init__(
        self,
        origin:List[float],
        spacing:List[float],
        nodes_shape:List[int],
        fix_coef=None
    ) -> None:
        super().__init__()
        origin=torch.tensor(np.array(origin).reshape((1,-1)))
        self.register_buffer("origin", origin, persistent=False)
        spacing=torch.tensor(np.array(spacing).reshape((1,-1)))
        self.register_buffer("spacing", spacing, persistent=False)
        self.nodes_shape=nodes_shape
        nodes_coord=np.mgrid[0:self.nodes_shape[0],0:self.nodes_shape[1],0:self.nodes_shape[2]]
        shift_coord_x=torch.tensor(nodes_coord[0].reshape((1,-1)))
        self.register_buffer("shift_coord_x", shift_coord_x, persistent=False)
        shift_coord_y=torch.tensor(nodes_coord[1].reshape((1,-1)))
        self.register_buffer("shift_coord_y", shift_coord_y, persistent=False)
        shift_coord_z=torch.tensor(nodes_coord[2].reshape((1,-1)))
        self.register_buffer("shift_coord_z", shift_coord_z, persistent=False)
        if fix_coef is None:
            self.weight_u = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.weight_v = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.weight_w = nn.Parameter(torch.empty((1,np.prod(nodes_shape))))
            self.reset_parameters()
        elif len(fix_coef)==len(nodes_shape):
            weight_u=torch.tensor(np.array(fix_coef[0]).reshape((1,-1)))
            self.register_buffer("weight_u", weight_u, persistent=False)
            weight_v=torch.tensor(np.array(fix_coef[1]).reshape((1,-1)))
            self.register_buffer("weight_v", weight_v, persistent=False)
            weight_w=torch.tensor(np.array(fix_coef[2]).reshape((1,-1)))
            self.register_buffer("weight_w", weight_w, persistent=False)
        else:
            weight_u=torch.tensor(np.array(fix_coef[...,0]).reshape((1,-1)))
            self.register_buffer("weight_u", weight_u, persistent=False)
            weight_v=torch.tensor(np.array(fix_coef[...,1]).reshape((1,-1)))
            self.register_buffer("weight_v", weight_v, persistent=False)
            weight_w=torch.tensor(np.array(fix_coef[...,2]).reshape((1,-1)))
            self.register_buffer("weight_w", weight_w, persistent=False)
            
    def reset_parameters(self) -> None:
        torch.nn.init.constant_(self.weight_u, 1.0)
        torch.nn.init.constant_(self.weight_v, 1.0)
        torch.nn.init.constant_(self.weight_w, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        BsplineCoord=(x-self.origin)/self.spacing
        shifted_weight=Bspline1D(BsplineCoord[...,0:1]-self.shift_coord_x)*Bspline1D(BsplineCoord[...,1:2]-self.shift_coord_y)*Bspline1D(BsplineCoord[...,2:3]-self.shift_coord_z)
        u=torch.sum(shifted_weight*self.weight_u,-1, keepdim=True)
        v=torch.sum(shifted_weight*self.weight_v,-1, keepdim=True)
        w=torch.sum(shifted_weight*self.weight_w,-1, keepdim=True)
        return torch.cat((u,v,w)-1)
class ParametricBspline3DArchCore(nn.Module):
    def __init__(
        self,
        origin:List[float],
        spacing:List[float],
        nodes_shape:List[int],
        separateCoef:bool = False
    ) -> None:
        super().__init__()
        origin=torch.tensor(np.array(origin).reshape((1,-1)))
        self.register_buffer("origin", origin, persistent=False)
        spacing=torch.tensor(np.array(spacing).reshape((1,-1)))
        self.register_buffer("spacing", spacing, persistent=False)
        self.nodes_shape=nodes_shape
        nodes_coord=np.mgrid[0:self.nodes_shape[0],0:self.nodes_shape[1],0:self.nodes_shape[2]]
        shift_coord_x=torch.tensor(nodes_coord[0].reshape((1,-1)))
        self.register_buffer("shift_coord_x", shift_coord_x, persistent=False)
        shift_coord_y=torch.tensor(nodes_coord[1].reshape((1,-1)))
        self.register_buffer("shift_coord_y", shift_coord_y, persistent=False)
        shift_coord_z=torch.tensor(nodes_coord[2].reshape((1,-1)))
        self.register_buffer("shift_coord_z", shift_coord_z, persistent=False)
        total=int(np.prod(nodes_shape))*3
        if separateCoef:
            u_start=torch.tensor(0,dtype=torch.long)
            self.register_buffer("u_start", u_start, persistent=False)
            u_stop=torch.tensor(int(total/3),dtype=torch.long)
            self.register_buffer("u_stop", u_stop, persistent=False)
            u_skip=torch.tensor(1,dtype=torch.long)
            self.register_buffer("u_skip", u_skip, persistent=False)
            v_start=torch.tensor(int(total/3),dtype=torch.long)
            self.register_buffer("v_start", v_start, persistent=False)
            v_stop=torch.tensor(int(total*2/3),dtype=torch.long)
            self.register_buffer("v_stop", v_stop, persistent=False)
            v_skip=torch.tensor(1,dtype=torch.long)
            self.register_buffer("v_skip", v_skip, persistent=False)
            w_start=torch.tensor(int(total*2/3),dtype=torch.long)
            self.register_buffer("w_start", w_start, persistent=False)
            w_stop=torch.tensor(total,dtype=torch.long)
            self.register_buffer("w_stop", w_stop, persistent=False)
            w_skip=torch.tensor(1,dtype=torch.long)
            self.register_buffer("w_skip", w_skip, persistent=False)
        else:
            u_start=torch.tensor(0,dtype=torch.long)
            self.register_buffer("u_start", u_start, persistent=False)
            u_stop=torch.tensor(total,dtype=torch.long)
            self.register_buffer("u_stop", u_stop, persistent=False)
            u_skip=torch.tensor(3,dtype=torch.long)
            self.register_buffer("u_skip", u_skip, persistent=False)
            v_start=torch.tensor(1,dtype=torch.long)
            self.register_buffer("v_start", v_start, persistent=False)
            v_stop=torch.tensor(total,dtype=torch.long)
            self.register_buffer("v_stop", v_stop, persistent=False)
            v_skip=torch.tensor(3,dtype=torch.long)
            self.register_buffer("v_skip", v_skip, persistent=False)
            w_start=torch.tensor(2,dtype=torch.long)
            self.register_buffer("w_start", w_start, persistent=False)
            w_stop=torch.tensor(total,dtype=torch.long)
            self.register_buffer("w_stop", w_stop, persistent=False)
            w_skip=torch.tensor(3,dtype=torch.long)
            self.register_buffer("w_skip", w_skip, persistent=False)

    def forward(self, x: Tensor, o: Tensor) -> Tensor:
        BsplineCoord=(x-self.origin)/self.spacing
        shifted_weight=Bspline1D(BsplineCoord[...,0:1]-self.shift_coord_x)*Bspline1D(BsplineCoord[...,1:2]-self.shift_coord_y)*Bspline1D(BsplineCoord[...,2:3]-self.shift_coord_z)
        u=torch.sum(shifted_weight*o[...,self.u_start:self.u_stop:self.u_skip],-1, keepdim=True)
        v=torch.sum(shifted_weight*o[...,self.v_start:self.v_stop:self.v_skip],-1, keepdim=True)
        w=torch.sum(shifted_weight*o[...,self.w_start:self.w_stop:self.w_skip],-1, keepdim=True)
        return torch.cat((u,v,w)-1)
    
###CustomModuleArch
class CustomModuleArch(Arch):
    def __init__(
        self,
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=input_keys,
                output_keys=output_keys,
                detach_keys=detach_keys,
                periodicity=None,
            )
        self.module=nn.ModuleList()
        if module is None:
            self.module.append(torch.nn.Identity())
        else:
            self.module.append(module)
        
    def _tensor_forward(self, x: Tensor) -> Tensor:
        x = self.process_input(
            x,
            self.input_scales_tensor,
            periodicity=self.periodicity,
            input_dict=self.input_key_dict,
            dim=-1,
        )
        x = self.module[0](x)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        x = self.module[0](x)
        x = self.process_output(x, self.output_scales_tensor)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
class CustomDualInputModuleArch(Arch):
    def __init__(
        self,
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=input1_keys+input2_keys,
                output_keys=output_keys,
                detach_keys=detach_keys,
                periodicity=None,
            )
        self.input1_key_dict = {str(var): var.size for var in input1_keys}
        self.total_input1_size=sum(self.input1_key_dict.values())
        self.total_input_size=sum(self.input_key_dict.values())
        self.input2_key_dict = {str(var): var.size for var in input2_keys}
        self.module=nn.ModuleList()
        if module is None:
            raise Exception("module cannot be None.")
        else:
            self.module.append(module)
        
    def _tensor_forward(self, x: Tensor,o: Tensor) -> Tensor:
        if self.input_scales_tensor is None:
            input_scales_tensor1=None
            input_scales_tensor2=None
        else:
            input_scales_tensor1=self.input_scales_tensor[...,0:self.total_input1_size]
            input_scales_tensor2=self.input_scales_tensor[...,self.total_input1_size:self.total_input_size]
        x = self.process_input(
            x,
            input_scales_tensor1,
            periodicity=self.periodicity,
            input_dict=self.input1_key_dict,
            dim=-1,
        )
        o = self.process_input(
            o,
            input_scales_tensor2,
            periodicity=self.periodicity,
            input_dict=self.input2_key_dict,
            dim=-1,
        )
        x = self.module[0](x,o)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        o = self.concat_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x,o)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        o = self.prepare_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        x = self.module[0](x,o)
        x = self.process_output(x, self.output_scales_tensor)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
class CustomDualInputDualOutputModuleArch(Arch):#!!!!
    def __init__(
        self,
        input1_keys: List[Key],
        input2_keys: List[Key],
        output1_keys: List[Key],
        output2_keys: List[Key],
        detach_keys: List[Key] = [],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=input1_keys+input2_keys,
                output_keys=output1_keys+output2_keys,
                detach_keys=detach_keys,
                periodicity=None,
            )
        self.input1_key_dict = {str(var): var.size for var in input1_keys}
        self.total_input1_size=sum(self.input1_key_dict.values())
        self.total_input_size=sum(self.input_key_dict.values())
        self.input2_key_dict = {str(var): var.size for var in input2_keys}
        self.module=nn.ModuleList()
        if module is None:
            raise Exception("module cannot be None.")
        else:
            self.module.append(module)
        self.output1_key_dict = {str(var): var.size for var in output1_keys}
        self.total_output1_size=sum(self.output1_key_dict.values())
        self.total_output_size=sum(self.output_key_dict.values())
        self.output2_key_dict = {str(var): var.size for var in output2_keys}
    def _tensor_forward(self, x: Tensor,o: Tensor) -> Tensor:
        if self.input_scales_tensor is None:
            input_scales_tensor1=None
            input_scales_tensor2=None
        else:
            input_scales_tensor1=self.input_scales_tensor[...,0:self.total_input1_size]
            input_scales_tensor2=self.input_scales_tensor[...,self.total_input1_size:self.total_input_size]
        x = self.process_input(
            x,
            input_scales_tensor1,
            periodicity=self.periodicity,
            input_dict=self.input1_key_dict,
            dim=-1,
        )
        o = self.process_input(
            o,
            input_scales_tensor2,
            periodicity=self.periodicity,
            input_dict=self.input2_key_dict,
            dim=-1,
        )
        x,y2 = self.module[0](x,o)
        if self.output_scales_tensor is None:
            output_scales_tensor1=None
            output_scales_tensor2=None
        else:
            output_scales_tensor1=self.output_scales_tensor[...,0:self.total_output1_size]
            output_scales_tensor2=self.output_scales_tensor[...,self.total_output1_size:self.total_output_size]
        x = self.process_output(x, output_scales_tensor1)
        y2 = self.process_output(y2, output_scales_tensor2)
        return x,y2

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        o = self.concat_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y,y2 = self._tensor_forward(x,o)
        returnDict=self.split_output(y, self.output1_key_dict, dim=-1)
        returnDict.update(self.split_output(y2, self.output2_key_dict, dim=-1))
        return returnDict

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        o = self.prepare_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        x,y2 = self.module[0](x,o)
        if self.output_scales_tensor is None:
            output_scales_tensor1=None
            output_scales_tensor2=None
        else:
            output_scales_tensor1=self.output_scales_tensor[...,0:self.total_output1_size]
            output_scales_tensor2=self.output_scales_tensor[...,self.total_output1_size:self.total_output_size]
        y = self.process_output(x, output_scales_tensor1)
        y2 = self.process_output(y2, output_scales_tensor2)
        returnDict=self.prepare_output(y, self.output1_key_dict, dim=-1, output_scales=self.output_scales)
        returnDict.update(self.prepare_output(y2, self.output2_key_dict, dim=-1, output_scales=self.output_scales))
        
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
class CustomTripleInputModuleArch(Arch):
    
    def __init__(
        self,
        input1_keys: List[Key],
        input2_keys: List[Key],
        input3_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=input1_keys+input2_keys+input3_keys,
                output_keys=output_keys,
                detach_keys=detach_keys,
                periodicity=None,
            )
        self.input1_key_dict = {str(var): var.size for var in input1_keys}
        self.total_input1_size=sum(self.input1_key_dict.values())
        self.total_input_size=sum(self.input_key_dict.values())
        self.input2_key_dict = {str(var): var.size for var in input2_keys}
        self.total_input2_size=self.total_input1_size+sum(self.input2_key_dict.values())
        self.input3_key_dict = {str(var): var.size for var in input3_keys}
        self.module=nn.ModuleList()
        if module is None:
            raise Exception("module cannot be None.")
        else:
            self.module.append(module)
        
    def _tensor_forward(self, x: Tensor,o: Tensor,o2: Tensor) -> Tensor:
        if self.input_scales_tensor is None:
            input_scales_tensor1=None
            input_scales_tensor2=None
            input_scales_tensor3=None
        else:
            input_scales_tensor1=self.input_scales_tensor[...,0:self.total_input1_size]
            input_scales_tensor2=self.input_scales_tensor[...,self.total_input1_size:self.total_input2_size]
            input_scales_tensor3=self.input_scales_tensor[...,self.total_input2_size:self.total_input_size]
        x = self.process_input(
            x,
            input_scales_tensor1,
            periodicity=self.periodicity,
            input_dict=self.input1_key_dict,
            dim=-1,
        )
        o = self.process_input(
            o,
            input_scales_tensor2,
            periodicity=self.periodicity,
            input_dict=self.input2_key_dict,
            dim=-1,
        )
        o2 = self.process_input(
            o2,
            input_scales_tensor3,
            periodicity=self.periodicity,
            input_dict=self.input3_key_dict,
            dim=-1,
        )
        x = self.module[0](x,o,o2)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        o = self.concat_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        o2 = self.concat_input(
            in_vars,
            self.input3_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x,o,o2)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        o = self.prepare_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        o2 = self.prepare_input(
            in_vars,
            self.input3_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        x = self.module[0](x,o,o2)
        x = self.process_output(x, self.output_scales_tensor)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
class CustomQuadInputModuleArch(Arch):
    
    def __init__(
        self,
        input1_keys: List[Key],
        input2_keys: List[Key],
        input3_keys: List[Key],
        input4_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=input1_keys+input2_keys+input3_keys+input4_keys,
                output_keys=output_keys,
                detach_keys=detach_keys,
                periodicity=None,
            )
        self.input1_key_dict = {str(var): var.size for var in input1_keys}
        self.total_input1_size=sum(self.input1_key_dict.values())
        self.total_input_size=sum(self.input_key_dict.values())
        self.input2_key_dict = {str(var): var.size for var in input2_keys}
        self.total_input2_size=self.total_input1_size+sum(self.input2_key_dict.values())
        self.input3_key_dict = {str(var): var.size for var in input3_keys}
        self.total_input3_size=self.total_input2_size+sum(self.input3_key_dict.values())
        self.input4_key_dict = {str(var): var.size for var in input4_keys}
        self.module=nn.ModuleList()
        if module is None:
            raise Exception("module cannot be None.")
        else:
            self.module.append(module)
        
    def _tensor_forward(self, x: Tensor,o: Tensor,o2: Tensor,o3: Tensor) -> Tensor:
        if self.input_scales_tensor is None:
            input_scales_tensor1=None
            input_scales_tensor2=None
            input_scales_tensor3=None
            input_scales_tensor4=None
        else:
            input_scales_tensor1=self.input_scales_tensor[...,0:self.total_input1_size]
            input_scales_tensor2=self.input_scales_tensor[...,self.total_input1_size:self.total_input2_size]
            input_scales_tensor3=self.input_scales_tensor[...,self.total_input2_size:self.total_input_size]
            input_scales_tensor4=self.input_scales_tensor[...,self.total_input3_size:self.total_input_size]
        x = self.process_input(
            x,
            input_scales_tensor1,
            periodicity=self.periodicity,
            input_dict=self.input1_key_dict,
            dim=-1,
        )
        o = self.process_input(
            o,
            input_scales_tensor2,
            periodicity=self.periodicity,
            input_dict=self.input2_key_dict,
            dim=-1,
        )
        o2 = self.process_input(
            o2,
            input_scales_tensor3,
            periodicity=self.periodicity,
            input_dict=self.input3_key_dict,
            dim=-1,
        )
        o3 = self.process_input(
            o3,
            input_scales_tensor4,
            periodicity=self.periodicity,
            input_dict=self.input4_key_dict,
            dim=-1,
        )
        x = self.module[0](x,o,o2,o3)
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        x = self.concat_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        o = self.concat_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        o2 = self.concat_input(
            in_vars,
            self.input3_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        o3 = self.concat_input(
            in_vars,
            self.input4_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
        )
        y = self._tensor_forward(x,o,o2,o3)
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.prepare_input(
            in_vars,
            self.input1_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        o = self.prepare_input(
            in_vars,
            self.input2_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        o2 = self.prepare_input(
            in_vars,
            self.input3_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        o3 = self.prepare_input(
            in_vars,
            self.input4_key_dict.keys(),
            detach_dict=self.detach_key_dict,
            dim=-1,
            input_scales=self.input_scales,
            periodicity=self.periodicity,
        )
        x = self.module[0](x,o,o2,o3)
        x = self.process_output(x, self.output_scales_tensor)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )
class CustomZeroInputModuleArch(Arch):
    """Fully Connected Neural Network.

    Parameters
    ----------
    input_keys : List[Key]
        Input key list.
    output_keys : List[Key]
        Output key list.
    detach_keys : List[Key], optional
        List of keys to detach gradients, by default []
    domain_layer_sizeList : List[None,List[int]], optional
        List of Layer size for every hidden layer of the model including the output, by default 512,512,512,512,512,512,len(output_keys)
    domain_inputKeyList : List[boolList[Key]], optional
        Key to send to each NN of the domain cluster, by default [True]*len(domain_layer_sizeList)
    modes_decoding : torch.nn.Module, optional
        module to run before output of architecture, number of output must match number of output_keys, by default torch.nn.Identity()
    gEncoding_layer_sizeList: List[None,List[int]], optional
        List of Layer size for every hidden layer of the geometry encoding including the output, by default None
    gEncoding_inputKeyList: List[boolList[Key]], optional
        Key to send to each NN of the geometry encoding, by default [True]*len(gEncoding_layer_sizeList)
    gfEncoding_outputKey: Union[None,List[Key]] =None,
        Key to output from geometry and functional encoding, by default None
    functionalEncoding : torch.nn.Module, optional
        module to run between geometry encoding and domain cluster,  number of output must match number of gfEncoding_outputKey, by default None
    activation_fn : Activation, optional
        Activation function used by network, by default :obj:`Activation.SILU`
    periodicity : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary of tuples that allows making model give periodic predictions on
        the given bounds in tuple.
    skip_connections : bool, optional
        Apply skip connections every 2 hidden layers, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default True
    adaptive_activations : bool, optional
        Use an adaptive activation functions, by default False

    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size]`
    - Output variable tensor shape: :math:`[N, size]`

    Example
    -------
    Fully-connected model (2 -> 64 -> 64 -> 2)

    >>> arch = .geometrical_modes.GeometricalModesArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    domain_layer_sizeList = [64,64,2])
    >>> model = arch.make_node()
    >>> input = {"x": torch.randn(64, 2)}
    >>> output = model.evaluate(input)
    Note
    ----
    For information regarding adaptive activations please refer to
    https://arxiv.org/abs/1906.01170.
    """
    def __init__(
        self,
        output_keys: List[Key],
        module=None,
    ) -> None:
        super().__init__(
                input_keys=[],
                output_keys=output_keys,
                detach_keys=[],
                periodicity=None,
            )
        self.module=nn.ModuleList()
        if module is None:
            raise Exception("module cannot be None.")
        else:
            self.module.append(module)
        
    def _tensor_forward(self) -> Tensor:
        x = self.module[0]()
        x = self.process_output(x, self.output_scales_tensor)
        return x

    def forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        y = self._tensor_forward()
        return self.split_output(y, self.output_key_dict, dim=-1)

    def _dict_forward(self, in_vars: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        This is the original forward function, left here for the correctness test.
        """
        x = self.module[0]()
        x = self.process_output(x, self.output_scales_tensor)
        return self.prepare_output(
            x, self.output_key_dict, dim=-1, output_scales=self.output_scales
        )

###Arch
def FullyConnectedFlexiLayerSizeArch(
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        conv_layers: bool = False,
    ):
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomModuleArch(input_keys,
                            output_keys,
                            detach_keys,
                            module=FullyConnectedFlexiLayerSizeArchCore(in_features=sum([x.size for x in input_keys]),
                                                                        layer_sizeList=layer_sizeList,
                                                                        skip_connections=skip_connections,
                                                                        activation_fn=activation_fn,
                                                                        adaptive_activations=adaptive_activations,
                                                                        weight_norm=weight_norm,
                                                                        conv_layers=conv_layers
                                                                        )
                            )
def PIConnectedArch(
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 6,
        layer_size_reduction: int = 2,
        activation_fn: Activation = Activation.SILU,
        skip_activation: int = -1,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        reducenorm: float = 0.2
    ):
    return CustomModuleArch(input_keys,
                            output_keys,
                            detach_keys,
                            module=PIConnectedArchCore(in_features = sum([x.size for x in input_keys]),
                                                        layer_size = layer_size,
                                                        out_features = sum([x.size for x in output_keys]),
                                                        nr_layers = nr_layers,
                                                        layer_size_reduction = layer_size_reduction,
                                                        activation_fn = activation_fn,
                                                        skip_activation = skip_activation,
                                                        adaptive_activations = adaptive_activations,
                                                        weight_norm = weight_norm,
                                                        reducenorm=reducenorm,
                                                        )
                            )
def InheritedFullyConnectedFlexiLayerSizeArch(
        input_keys: List[Key],
        output_keys: List[Key],
        case_input_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        caseNN_layer_sizeList: Union[int,List[int],None] = None,
        caseNN_skip_connections: bool = False,
        caseNN_activation_fn: Activation = Activation.SILU,
        caseNN_adaptive_activations: bool = False,
        caseNN_weight_norm: bool = True,
        with_caseID: bool = True,
    ):
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomDualInputModuleArch(input_keys,
                            case_input_keys,
                            output_keys,
                            detach_keys,
                            module=InheritedFullyConnectedFlexiLayerSizeCombinedArchCore(in_features=sum([x.size for x in input_keys]),
                                                                        layer_sizeList=layer_sizeList,
                                                                        skip_connections=skip_connections,
                                                                        activation_fn=activation_fn,
                                                                        caseNN_in_features=sum([x.size for x in case_input_keys]),
                                                                        caseNN_layer_sizeList=caseNN_layer_sizeList,
                                                                        caseNN_skip_connections=caseNN_skip_connections,
                                                                        caseNN_activation_fn=caseNN_activation_fn,
                                                                        caseNN_adaptive_activations=caseNN_adaptive_activations,
                                                                        caseNN_weight_norm=caseNN_weight_norm,
                                                                        )
                            )
def FullyConnectedFlexiLayerSizeFBWindowArch(
        startList: List[Union[float]],
        endList: List[Union[float]],
        overlapRatio:float,
        levels_num:int,
        input_keys: List[Key],
        output_keys: List[Key],
        FB_keys: List[Key],
        detach_keys: List[Key] = [],
        tol: float= 1e-10,
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        conv_layers: bool = False,
    ):
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomDualInputModuleArch(input_keys,
                                     FB_keys,
                                     output_keys,
                                     detach_keys,
                                     module=FullyConnectedFlexiLayerSizeFBWindowArchCore(startList=startList,
                                                                                        endList=endList,
                                                                                        overlapRatio=overlapRatio,
                                                                                        levels_num=levels_num,
                                                                                        tol=tol,
                                                                                        in_features=sum([x.size for x in input_keys]),
                                                                                        layer_sizeList=layer_sizeList,
                                                                                        skip_connections=skip_connections,
                                                                                        activation_fn=activation_fn,
                                                                                        adaptive_activations=adaptive_activations,
                                                                                        weight_norm=weight_norm,
                                                                                        conv_layers=conv_layers
                                                                                        )
                                    )        
def BackgroundGNNAdjacencyArch(
        input_key:List[Key],
        output_keys: List[Key],
        bg_GNN_coordinates: Tensor,
        dist_upper_limit: float,
        p_norm:float =2.
    ):
    return CustomModuleArch([input_key],
                            output_keys,
                            [],
                            module=BackgroundGNNAdjacencyArchCore(bg_GNN_coordinates=bg_GNN_coordinates,
                                                                  dist_upper_limit=dist_upper_limit,
                                                                  p_norm=p_norm)
                            )
def BackgroundGraphNNFullyConnectedFlexiLayerSizeArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        input3_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_sizeList: Union[int,List[int]] = 512,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        skip_layer: int = 1,
    ):
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomTripleInputModuleArch(input1_keys,
                                       input2_keys,
                                       input3_keys,
                                       output_keys,
                                       detach_keys,
                                       module=BackgroundGraphNNFullyConnectedFlexiLayerSizeArchCore(in_features=sum([x.size for x in input1_keys]),
                                                                                    layer_sizeList=layer_sizeList,
                                                                                    activation_fn=activation_fn,
                                                                                    adaptive_activations=adaptive_activations,
                                                                                    weight_norm=weight_norm,
                                                                                    skip_layer = skip_layer,
                                                                                    )
                                       )
def AdditionArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
    ):
    if sum([x.size for x in input1_keys])!=sum([x.size for x in input2_keys]):
        raise Exception("input1_keys and input2_keys in AdditionArch has to be the same total size")
    return CustomDualInputModuleArch(input1_keys,
                            input2_keys,
                            output_keys,
                            detach_keys,
                            module=AdditionArchCore()
                            )
def SubtractionArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
    ):
    if sum([x.size for x in input1_keys])!=sum([x.size for x in input2_keys]):
        raise Exception("input1_keys and input2_keys in AdditionArch has to be the same total size")
    return CustomDualInputModuleArch(input1_keys,
                            input2_keys,
                            output_keys,
                            detach_keys,
                            module=SubtractionArchCore()
                            )
def MultiplicationArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
    ):
    size1=sum([x.size for x in input1_keys])
    size2=sum([x.size for x in input2_keys])
    if size1!=size2 and size1!=1 and size2!=1:
        raise Exception("input1_keys and input2_keys in AdditionArch has to be the same total size")
    return CustomDualInputModuleArch(input1_keys,
                            input2_keys,
                            output_keys,
                            detach_keys,
                            module=MultiplicationArchCore()
                            )
def MatrixMultiplicationArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
    ):
    return CustomDualInputModuleArch(input1_keys,
                            input2_keys,
                            output_keys,
                            detach_keys,
                            module=MatrixMultiplicationArchCore()
                            )
def SumMultiplicationArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
    ):
    size1=sum([x.size for x in input1_keys])
    size2=sum([x.size for x in input2_keys])
    if size1!=size2 and size1!=1 and size2!=1:
        raise Exception("input1_keys and input2_keys in AdditionArch has to be the same total size")
    return CustomDualInputModuleArch(input1_keys,
                            input2_keys,
                            output_keys,
                            detach_keys,
                            module=SumMultiplicationArchCore()
                            )
def SingleInputInheritedFullyConnectedFlexiLayerSizeArch(
        input_keys: List[Key],
        output_keys: List[Key],
        case_input_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_sizeList: Union[int,List[int]] = 512,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        caseNN_layer_sizeList: Union[int,List[int],None] = None,
        caseNN_skip_connections: bool = False,
        caseNN_activation_fn: Activation = Activation.SILU,
        caseNN_adaptive_activations: bool = False,
        caseNN_weight_norm: bool = True,
        with_caseID: int = 0,
    ):
    adjust_caseNN_in_features=0
    if with_caseID:
        adjust_caseNN_in_features=-1
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomModuleArch(input_keys+case_input_keys,
                            output_keys,
                            detach_keys,
                            module=SingleInputInheritedFullyConnectedFlexiLayerSizeCombinedArchCore(in_features=sum([x.size for x in input_keys]),
                                                                        layer_sizeList=layer_sizeList,
                                                                        skip_connections=skip_connections,
                                                                        activation_fn=activation_fn,
                                                                        caseNN_in_features=sum([x.size for x in case_input_keys])+adjust_caseNN_in_features,
                                                                        caseNN_layer_sizeList=caseNN_layer_sizeList,
                                                                        caseNN_skip_connections=caseNN_skip_connections,
                                                                        caseNN_activation_fn=caseNN_activation_fn,
                                                                        caseNN_adaptive_activations=caseNN_adaptive_activations,
                                                                        caseNN_weight_norm=caseNN_weight_norm,
                                                                        with_caseID=with_caseID
                                                                        )
                            )
def HypernetFullyConnectedFlexiLayerSizeArch(
        input_keys: List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_sizeList: Union[int,List[int],None] = None,
        skip_connections: bool = False,
        activation_fn: Activation = Activation.SILU,
        adaptive_activations: bool = False,
        weight_norm: bool = True,
        with_caseID: int = 0,
    ):
    adjust_caseNN_in_features=0
    if with_caseID:
        adjust_caseNN_in_features=-1
    if isinstance(layer_sizeList,int):
        if layer_sizeList<=0:
            layer_sizeList=[]
        else:
            layer_sizeList=[layer_sizeList]
    layer_sizeList=layer_sizeList+[sum([x.size for x in output_keys])]
    return CustomModuleArch(input_keys,
                            output_keys,
                            detach_keys,
                            module=HypernetFullyConnectedFlexiLayerSizeArchCore(in_features=sum([x.size for x in input_keys])+adjust_caseNN_in_features,
                                                                        layer_sizeList=layer_sizeList,
                                                                        skip_connections=skip_connections,
                                                                        activation_fn=activation_fn,
                                                                        adaptive_activations=adaptive_activations,
                                                                        weight_norm=weight_norm,
                                                                        with_caseID=with_caseID
                                                                        )
                            )
def CaseIDArch(
        input_key:Key,
        output_keys: List[Key],
        caseNum,
    ):
    return CustomModuleArch([input_key],
                            output_keys,
                            [],
                            module=CaseIDArchCore(caseNum,
                                                    )
                            )
def CosSinArch(
        input_keys:List[Key],
        output_keys: List[Key],
        detach_keys: List[Key] = [],
        layer_size: int = 512,
        nr_layers: int = 0,
        activation_fn: Activation = Activation.IDENTITY,
        cos_layer: bool = True,
        sin_layer: bool = True,
    ):
    return CustomModuleArch(input_keys,
                            output_keys,
                            detach_keys,
                            module=CosSinArchCore(in_features=sum([x.size for x in input_keys]),
                                                  layer_size=layer_size,
                                                  out_features=sum([x.size for x in output_keys]),
                                                  nr_layers = nr_layers,
                                                  activation_fn = activation_fn,
                                                  cos_layer=cos_layer,
                                                  sin_layer=sin_layer
                                                    )
                            )
def CaseIDtoFeatureArch(
        input_key:Key,
        output_keys: List[Key],
        feature_array,
    ):
    return CustomModuleArch([input_key],
                            output_keys,
                            [],
                            module=CaseIDtoFeatureArchCore(feature_array,
                                                    )
                            )
def FixedFeatureArch(
        output_keys: List[Key],
        feature_array,
    ):
    return CustomZeroInputModuleArch(output_keys,
                            module=FixedFeatureArchCore(feature_array,
                                                    )
                            )
def MaxPoolArch(
        input_keys: List[Key],
        output_keys: List[Key],
        pooldim=1,
        keepdim=False,
        keepview=True,
    ):
    return CustomModuleArch(input_keys,
                            output_keys,
                            [],
                            module=MaxPoolArchCore(pooldim,
                                                   keepdim,
                                                   keepview)
                            )
def MinPoolArch(
        input_keys: List[Key],
        output_keys: List[Key],
        pooldim=1,
        keepdim=False,
        keepview=True,
    ):
    return CustomModuleArch(input_keys,
                            output_keys,
                            [],
                            module=MinPoolArchCore(pooldim,
                                                   keepdim,
                                                   keepview)
                            )
def MeanPoolArch(
        input_keys: List[Key],
        output_keys: List[Key],
        pooldim=1,
        keepdim=False,
        keepview=True,
    ):
    return CustomModuleArch(input_keys,
                            output_keys,
                            [],
                            module=MeanPoolArchCore(pooldim,
                                                   keepdim,
                                                   keepview)
                            )
def SumPoolArch(
        input_keys: List[Key],
        output_keys: List[Key],
        pooldim=1,
        keepdim=False,
        keepview=True,
    ):
    return CustomModuleArch(input_keys,
                            output_keys,
                            [],
                            module=SumPoolArchCore(pooldim,
                                                   keepdim,
                                                   keepview)
                            )
    
def ParametricInsertArch(
        output_keys: List[Key],
        input_keys=None
    ):
    if input_keys is None:
        input_keys=[Key('x')]
    return CustomModuleArch(input_keys,
                            output_keys,
                            [],
                            module=ParametricInsertArchCore(out_features=sum([x.size for x in output_keys]),
                                                    )
                            )
def BsplineArch(
        input_keys: List[Key],
        output_keys: List[Key],
        origin:List[float],
        spacing:List[float],
        nodes_shape:List[int],
        fix_coef=None,
        detach_keys: List[Key] = [],
    ):
    if len(origin)==2:
        module=Bspline2DArchCore(origin,
                                    spacing,
                                    nodes_shape,
                                    fix_coef
                                    )
    elif len(origin)==3:
        module=Bspline3DArchCore(origin,
                                    spacing,
                                    nodes_shape,
                                    fix_coef
                                    )
    else:
        raise Exception("Dimension "+str(len(origin))+" not implemented")
    return CustomModuleArch(input_keys,
                            output_keys,
                            detach_keys,
                            module=module
                            )
def ParametricBsplineArch(
        input1_keys: List[Key],
        input2_keys: List[Key],
        output_keys: List[Key],
        origin:List[float],
        spacing:List[float],
        nodes_shape:List[int],
        separateCoef:bool=False,
        detach_keys: List[Key] = [],
    ):
    
    if len(origin)==2:
        module=ParametricBspline2DArchCore(origin,
                                            spacing,
                                            nodes_shape,
                                            separateCoef
                                            )
    elif len(origin)==3:
        module=ParametricBspline3DArchCore(origin,
                                            spacing,
                                            nodes_shape,
                                            separateCoef
                                            )
    else:
        raise Exception("Dimension "+str(len(origin))+" not implemented")
    return CustomDualInputModuleArch(input1_keys,
                                        input2_keys,
                                        output_keys,
                                        detach_keys,
                                        module=module
                                        )




try:
    from torch_geometric.nn.conv.gcn_conv import GCNConv
    class GCNConvFullyConnectedFlexiLayerSizeArchCore(nn.Module):###!!!!! NOt done yet
        def __init__(
            self,
            in_features: int = 512,
            layer_sizeList: Union[int,List[int]] = 512,
            activation_fn: Activation = Activation.SILU,
            weight_norm: bool = True,
        ) -> None:
            super().__init__()
    
            if isinstance(layer_sizeList,int):
                layer_sizeList=[layer_sizeList]
            nr_layers=len(layer_sizeList)-1
            # Allows for regular linear layers to be swapped for 1D Convs
            # Useful for channel operations in FNO/Transformers
    
    
            if not isinstance(activation_fn, list):
                activation_fn = [activation_fn] * nr_layers
            if len(activation_fn) < nr_layers:
                activation_fn = activation_fn + [activation_fn[-1]] * (
                    nr_layers - len(activation_fn)
                )

            self.activation_fn = nn.ModuleList()
            self.layers = nn.ModuleList()
    
            layer_in_features = in_features
            for i in range(nr_layers):
                self.layers.append(
                    GCNConv(
                        layer_in_features,
                        layer_sizeList[i],
                        normalize=weight_norm,
                    )
                )
                layer_in_features = layer_sizeList[i]
                self.activation_fn.append( get_activation_fn(
                activation_fn[i], out_features=layer_sizeList[i]
            ))
    
            self.final_layer = GCNConv(
                in_features=layer_in_features,
                out_features=layer_sizeList[-1],
                normalize=weight_norm,
            )
    
        def forward(self, x: Tensor,euclidean_feature: Tensor) -> Tensor:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                x= self.activation_fn[i](x)
    
            x = self.final_layer(x)
            return x
    
        def get_weight_list(self):
            weights = [layer.conv.weight for layer in self.layers] + [
                self.final_layer.conv.weight
            ]
            biases = [layer.conv.bias for layer in self.layers] + [
                self.final_layer.conv.bias
            ]
            return weights, biases
except:
    pass
