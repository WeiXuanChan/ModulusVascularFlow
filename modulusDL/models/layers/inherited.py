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

import logging
from typing import Callable
from typing import Optional
from typing import Union, List

import torch
import torch.nn as nn
from torch import Tensor

from modulus.models.layers.activation import Activation, get_activation_fn
logger = logging.getLogger(__name__)


class inheritedFCLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
    ) -> None:
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        #in_features = torch.tensor(in_features,dtype=torch.long)
        #self.register_buffer("in_features", in_features, persistent=False)
        #out_features = torch.tensor(out_features,dtype=torch.long)
        #self.register_buffer("out_features", out_features, persistent=False)
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(
            activation_fn, out_features=out_features
        )


    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def forward(self, x: Tensor, weights: Tensor,bias: Tensor) -> Tensor:
        y=bias+torch.matmul(x,weights)
        if self.activation_fn is not Activation.IDENTITY:
            y = self.exec_activation_fn(y)
        return y
class singleInputInheritedFCLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
    ) -> None:
        super().__init__()
        in_features = torch.tensor(in_features,dtype=torch.long)
        self.register_buffer("in_features", in_features, persistent=False)
        out_features = torch.tensor(out_features,dtype=torch.long)
        self.register_buffer("out_features", out_features, persistent=False)
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(
            activation_fn, out_features=out_features
        )


    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def forward(self, x: Tensor, o: List[Tensor]) -> Tensor:
        y=o[...,(self.in_features*self.out_features):((self.in_features+1)*self.out_features)]
        for n in range(self.in_features):
            y=y+x[...,n:(n+1)]*o[...,(n*self.out_features):((n+1)*self.out_features)]
        if self.activation_fn is not Activation.IDENTITY:
            y = self.exec_activation_fn(y)
        return y
class cosSinLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
    ) -> None:
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        #in_features = torch.tensor(in_features,dtype=torch.long)
        #self.register_buffer("in_features", in_features, persistent=False)
        #out_features = torch.tensor(out_features,dtype=torch.long)
        #self.register_buffer("out_features", out_features, persistent=False)
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(
            activation_fn, out_features=out_features
        )
        self.sinAmplitudes = nn.Parameter(torch.empty(in_features, out_features))
        self.cosAmplitudes = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.bias, 0)
        nn.init.xavier_uniform_(self.sinAmplitudes)
        nn.init.xavier_uniform_(self.cosAmplitudes)

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def forward(self, x: Tensor) -> Tensor:
        y=self.bias+torch.matmul(torch.sin(x),self.sinAmplitudes)+torch.matmul(torch.cos(x),self.cosAmplitudes)
        if self.activation_fn is not Activation.IDENTITY:
            y = self.exec_activation_fn(y)
        return y
        
class cosLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
    ) -> None:
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        #in_features = torch.tensor(in_features,dtype=torch.long)
        #self.register_buffer("in_features", in_features, persistent=False)
        #out_features = torch.tensor(out_features,dtype=torch.long)
        #self.register_buffer("out_features", out_features, persistent=False)
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(
            activation_fn, out_features=out_features
        )
        self.cosAmplitudes = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.bias, 0)
        nn.init.xavier_uniform_(self.cosAmplitudes)

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def forward(self, x: Tensor) -> Tensor:
        y=self.bias+torch.matmul(torch.cos(x),self.cosAmplitudes)
        if self.activation_fn is not Activation.IDENTITY:
            y = self.exec_activation_fn(y)
        return y
class sinLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_fn: Union[
            Activation, Callable[[Tensor], Tensor]
        ] = Activation.IDENTITY,
    ) -> None:
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        #in_features = torch.tensor(in_features,dtype=torch.long)
        #self.register_buffer("in_features", in_features, persistent=False)
        #out_features = torch.tensor(out_features,dtype=torch.long)
        #self.register_buffer("out_features", out_features, persistent=False)
        self.activation_fn = activation_fn
        self.callable_activation_fn = get_activation_fn(
            activation_fn, out_features=out_features
        )
        self.sinAmplitudes = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(1, out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.bias, 0)
        nn.init.xavier_uniform_(self.sinAmplitudes)

    def exec_activation_fn(self, x: Tensor) -> Tensor:
        return self.callable_activation_fn(x)

    def forward(self, x: Tensor) -> Tensor:
        y=self.bias+torch.matmul(torch.sin(x),self.sinAmplitudes)
        if self.activation_fn is not Activation.IDENTITY:
            y = self.exec_activation_fn(y)
        return y
