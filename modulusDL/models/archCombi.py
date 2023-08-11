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
import math
import itertools
import torch
import torch.nn as nn
from torch import Tensor

from modulus.models.fully_connected import FullyConnectedArch
from .arch import (
    CustomModuleArch,
    FullyConnectedFlexiLayerSizeArch, 
    CaseIDtoFeatureArch,
    CaseIDArch,
    FixedFeatureArch,
    MaxPoolArch,
    MatrixMultiplicationArch,
    SumMultiplicationArch,
    MultiplicationArch,
    CosSinArch,
    MinPoolArch,
    MeanPoolArch,
    SubtractionArch,
    SumPoolArch,
    CustomDualInputModuleArch,
    getTN,
)
###FUNCTIONS frequently used
'''
class FBwindow(nn.Module):
    def __init__(
        self,
        start,
        end,
        overlap,
        tol: float= 1e-10,
    ) -> None:
        super().__init__()
        if np.any(overlap <= 0) : raise Exception("ERROR overlapRatio ="+repr(overlap))
        scale=torch.tensor(np.log(1/tol-1)/overlap,dtype=torch.float)
        self.register_buffer("scale", scale, persistent=False)
        tol=torch.tensor(tol,dtype=torch.float)
        self.register_buffer("tol", tol, persistent=False)
        if np.any(start>=end): raise Exception("ERROR: xmin "+repr(start)+" >= xmax "+repr(end))
        start=torch.tensor(start,dtype=torch.float)
        self.register_buffer("start", start, persistent=False)
        end=torch.tensor(end,dtype=torch.float)
        self.register_buffer("end", end, persistent=False)
        zero=torch.tensor(0.,dtype=torch.float)
        self.register_buffer("zero", zero, persistent=False)
    def forward(self,x):
        #ytop=torch.clamp(torch.sigmoid((self.end-x)*self.scale),min=self.tol)
        #ybtm=torch.clamp(torch.sigmoid((x-self.start)*self.scale),min=self.tol)
        #y=torch.clamp(ybtm*ytop-self.tol,min=self.zero)
        y=torch.sigmoid((self.end-x)*self.scale)*torch.sigmoid((x-self.start)*self.scale)
        return torch.prod(y,dim=-1,keepdim=True)
'''
class FBwindow(nn.Module):
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
class FBleveldivide(nn.Module):
    def __init__(
        self,
        nlevel,
    ) -> None:
        super().__init__()
        nlevel=torch.tensor(nlevel)
        self.register_buffer("nlevel", nlevel, persistent=False)
    def forward(self,x):
        return x/self.nlevel

def FBarch_getWeight_number(
        input_keys,
        output_keys,
        FB_keys,
        levelsList,
        layer_size=64,
        nr_layers=2,
        ):
    if not(isinstance(input_keys,int)):
        if len(FB_keys)!=sum([x.size for x in FB_keys]): raise Exception("ERROR: only FB_keys of size 1 allowed!")
        FB_keys=len(FB_keys)
    if not(isinstance(input_keys,int)):
        input_keys=sum([x.size for x in input_keys])
    if not(isinstance(output_keys,int)):
        if len(output_keys)!=sum([x.size for x in output_keys]): raise Exception("ERROR: only output_keys of size 1 allowed!")
        output_keys=len(output_keys)
    perNet=getTN([input_keys]+[layer_size]*nr_layers+[output_keys])
    netNumber=0
    for level in levelsList:
        netNumber+=level**FB_keys
    return netNumber*perNet
class FBarch:
    def __init__(
        self,
        keys_str:str,
        input_keys:List[Key],
        output_keys:List[Key],
        FB_keys: List[Key],
        startList: Union[float,None,List[Union[float,None]]],
        endList: Union[float,None,List[Union[float,None]]],
        overlapRatio: float,
        levelsList: Union[List[int],int],
        layer_size:int=64,
        nr_layers:int=2,
        tol: float= 1e-10,
    ) -> None:
        self.flow_net=[]
        self.window_net=[]
        self.multiply_net=[]
        self.outputsum_net=[]
        self.flow_net_str=[]
        dim=len(FB_keys)
        if isinstance(levelsList,int):
            levelsList=[levelsList]
        else:
            levelsList=np.sort(levelsList)
        if isinstance(startList,float) or startList is None:
            startList=[startList]*dim
        if isinstance(endList,float) or endList is None:
            endList=[endList]*dim
        FBout_keys=[]
        if dim!=sum([x.size for x in FB_keys]): raise Exception("ERROR: only FB_keys of size 1 allowed!")
        if len(output_keys)!=sum([x.size for x in output_keys]): raise Exception("ERROR: only output_keys of size 1 allowed!")
        for level in levelsList:
            
            if level==1:
                self.flow_net_str.append("_level"+str(level))
                FBout_keys.append([Key(keys_str+self.flow_net_str[-1]+"_"+x.name) for x in output_keys])
                self.flow_net.append(FullyConnectedArch(
                    input_keys=input_keys,
                    output_keys=FBout_keys[-1],
                    layer_size=layer_size,
                    nr_layers=nr_layers
                ))
                self.window_net.append(None)
                self.multiply_net.append(None)
            else:
                discrete=np.linspace(startList,endList,num=level+1)
                space_temp=(discrete[1]-discrete[0])
                overlapList=overlapRatio*space_temp
                discrete[0]=discrete[0]-space_temp
                discrete[-1]=discrete[-1]+space_temp
                windows_permute= np.unique(list(itertools.permutations(list(range(level))*dim,dim)),axis=0)
                for leveln in windows_permute:
                    self.flow_net_str.append("_level"+str(level)+"_"+"".join([str(x) for x in leveln]))
                    FBout_keys.append([Key(keys_str+self.flow_net_str[-1]+"_"+x.name) for x in output_keys])
                    intemediate_keys=[Key(keys_str+self.flow_net_str[-1]+"_pre", size=len(output_keys))]
                    self.flow_net.append(FullyConnectedArch(
                        input_keys=input_keys,
                        output_keys=intemediate_keys,
                        layer_size=layer_size,
                        nr_layers=nr_layers
                    ))
                    temp_start_end=np.array([[discrete[x,xn],discrete[x+1,xn]] for xn,x in enumerate(leveln)])
                    
                    self.window_net.append(CustomModuleArch(
                        FB_keys,
                        [Key(keys_str+self.flow_net_str[-1]+"_window")],
                        module=FBwindow(temp_start_end[:,0],temp_start_end[:,1],overlapList)
                        ))
                    self.multiply_net.append(MultiplicationArch(
                        input1_keys=intemediate_keys,
                        input2_keys=[Key(keys_str+self.flow_net_str[-1]+"_window")],
                        output_keys=FBout_keys[-1]
                        ))
        FBout_keys=np.array(FBout_keys)
        number_of_levels=len(levelsList)
        if number_of_levels==1:
            addstr=""
        else:
            addstr="_sum"+keys_str
        for output_keyN in range(len(output_keys)):
            self.outputsum_net.append(SumPoolArch(
                input_keys=list(FBout_keys[:,output_keyN]),
                output_keys=[Key(output_keys[output_keyN].name+addstr)],
                pooldim=1,
                keepdim=True,
                keepview=False,
                ))
        if number_of_levels>1:
            self.outputmean_net=CustomModuleArch(
                                    [Key(x.name+addstr) for x in output_keys],
                                    output_keys,
                                    module=FBleveldivide(float(number_of_levels))
                                    )
        else:
            self.outputmean_net=None
    def make_node(self,name='FB_net'):
        if self.outputmean_net is None:
            nodes=[]
        else:
            nodes=[self.outputmean_net.make_node(name=name+"_levelmean")]
        for n in range(len(self.flow_net_str)):
            nodes=nodes+[self.flow_net[n].make_node(name=name+self.flow_net_str[n])]
            if self.window_net[n] is not None:
                nodes=nodes+([self.window_net[n].make_node(name=name+self.flow_net_str[n]+"_window")]
                             +[self.multiply_net[n].make_node(name=name+self.flow_net_str[n]+"_multiply")])
        for n in range(len(self.outputsum_net)):
            nodes=nodes+[self.outputsum_net[n].make_node(name=name+self.flow_net_str[n]+"_sum")]
        return nodes
        
                
        