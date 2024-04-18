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
File: geometry.py
Description: Geometrical-Modes Model Architecture For Nvidia Modulus

History:
    Date    Programmer SAR# - Description
    ---------- ---------- ----------------------------
  Author: w.x.chan@gmail.com         10Mar2023           - Created
"""
_version='1.0.0'
import torch
import numpy as np

class HeartChamber(torch.nn.Module):
    '''
    inputs:
        Key("x"): 
        Key("y"): 
        Key("z"):  
    return:
        Key("dissq_wall"): 
        Key("cline"): 
        Key("cl_r_x"): 
        Key("cl_r_y"):  
        Key("il_r_x"): 
        Key("il_r_y"): 
        Key("ol_r_x"): 
        Key("split_z_sqratio")
        Key("dissqfromorigin")
        Key("hb")
    '''
    def __init__(self,longlength,halfpowerlaw=4):
        super().__init__()
        hemisphere_radius=torch.tensor(longlength/2.)
        self.register_buffer("hemisphere_radius", hemisphere_radius, persistent=False)
        halfpowerlaw=torch.tensor(int(halfpowerlaw))
        self.register_buffer("halfpowerlaw", halfpowerlaw, persistent=False)
        self.softplus=torch.nn.Softplus(beta=10, threshold=2)
    def forward(self,x):
        inoutlet_shift=self.hemisphere_radius/2.
        inoutlet_radius=self.hemisphere_radius/4.
        cline=-x[...,2:3]/self.hemisphere_radius#cline: base -2 to apex 1
        cl_r_x=x[...,0:1]/self.hemisphere_radius
        cl_r_y=x[...,1:2]/self.hemisphere_radius
        split_z_sqratio = 1.-torch.functional.relu(cline)**2.
        dissq_wall=split_z_sqratio-(cl_r_x)**2.-cl_r_y**2.
        inoutlet_shift_adjusted=inoutlet_shift*cline/2.
        dissqfromorigin_tobase=torch.functional.relu(-cline/2.)**2.
        spreading_radius=inoutlet_radius + (self.hemisphere_radius-inoutlet_radius)*(1.-dissqfromorigin_tobase)
        il_r_x=self.softplus(2.-self.softplus(1.-(x[...,0:1]-inoutlet_shift_adjusted)/spreading_radius))-1.
        ol_r_x=self.softplus(2.-self.softplus(1.-(x[...,0:1]-inoutlet_shift_adjusted)/spreading_radius))-1.
        il_r_y=self.softplus(2.-self.softplus(1.-x[...,1:2]/spreading_radius))-1.
        inlet=1.-self.softplus(1.-((x[...,0:1]-inoutlet_shift_adjusted)**2.+(x[...,1:2])**2./(spreading_radius)**2.)**self.halfpowerlaw)
        outlet=1.-self.softplus(1.-((x[...,0:1]+inoutlet_shift_adjusted)**2.+(x[...,1:2])**2./spreading_radius**2.)**self.halfpowerlaw)
        hb=(1.-(1.-dissq_wall)**self.halfpowerlaw)*(1.-(dissqfromorigin_tobase**self.halfpowerlaw*inlet*outlet))
        return torch.cat((dissq_wall,cline,cl_r_x,cl_r_y,il_r_x,il_r_y,ol_r_x,split_z_sqratio,dissqfromorigin_tobase,hb),dim=-1)
