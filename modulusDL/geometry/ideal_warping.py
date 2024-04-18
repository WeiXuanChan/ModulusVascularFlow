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

class Stenosis3D(torch.nn.Module):
    '''
    inputs:
        Key("x"): 
        Key("y"): 
        Key("z"): 
        Key('r0'): 
        Key('rr'): 
        Key("A"): 
        Key("alpha"): 
        Key("s1"): 
        Key("s2"): 
        Key("s3"): 
        Key("s4"): 
    return:
        Key("x_case"): 
        Key("y_case"): 
        Key("z_case"): 
        Key("delta_yx_of_centerline"): 
        Key("delta_zx_of_centerline")]: 
    '''
    def __init__(self,r,l):
        super().__init__()
        r=torch.tensor(r)
        self.register_buffer("r", r, persistent=False)
        l=torch.tensor(l)
        self.register_buffer("l", l, persistent=False)#length of line
    def forward(self,x):
        x_ref=x[...,0:1]/self.l
        y_ref=x[...,1:2]
        z_ref=x[...,2:3]
        r0=1+x[...,3:4]
        rr=x[...,4:5]*x_ref*2.
        A=x[...,5:6]
        alpha=-1.*(x_ref/x[...,6:7])**2./2.
        s1=x[...,7:8]
        s2=x[...,8:9]
        s3=x[...,9:10]
        s4=x[...,10:11]
        cline_y=self.r*(s1*torch.sin(np.pi*x_ref)+s2*torch.sin(2.*np.pi*x_ref))
        cline_z=self.r*(s3*torch.sin(np.pi*x_ref)+s4*torch.sin(2.*np.pi*x_ref))
        radius_y=y_ref*(r0+rr)*(1.-A*torch.exp(alpha))
        radius_z=z_ref*(r0+rr)*(1.-A*torch.exp(alpha))
        cline_dydx=self.r*np.pi/self.l*(s1*torch.cos(np.pi*x_ref)+2.*s2*torch.cos(2.*np.pi*x_ref))
        cline_dzdx=self.r*np.pi/self.l*(s3*torch.cos(np.pi*x_ref)+2.*s4*torch.cos(2.*np.pi*x_ref))
        norm_y=1./torch.sqrt(cline_dydx**2.+1.)
        norm_xy=-norm_y*cline_dydx
        norm_z=1./torch.sqrt(cline_dzdx**2.+1.)
        norm_xz=-norm_z*cline_dzdx
        return torch.cat((x[...,0:1]+norm_xy*radius_y+norm_xz*radius_z,cline_y+norm_y*radius_y,cline_z+norm_z*radius_z,cline_dydx,cline_dzdx),-1)
    
class HeatMyocardium_LCR_to_xyz(torch.nn.Module):
    '''
    fixed:
        Key('longitudinal_length'): 
        Key('radius'): 
        Key("center_to_base"): 
        Key("thickness"): 
    inputs:
        Key("l"):
        Key("c"): 
        Key("r"): 
    return:
        Key("x"): 
        Key("y"): 
        Key("z"): 
        Key("logivec_x"): 
        Key("logivec_y"): 
        Key("logivec_z"): 
        Key("cirvec_x"): 
        Key("cirvec_y"): 
        Key("cirvec_z"): 
        Key("radialvec_x"): 
        Key("radialvec_y"): 
        Key("radialvec_z"): 
    '''
    def __init__(self,longilength,center_to_base,endoradius,thickness):
        super().__init__()
        minor_semi_axis=torch.tensor(endoradius)
        self.register_buffer("minor_semi_axis", minor_semi_axis, persistent=False)
        major_semi_axis=torch.tensor(longilength-center_to_base)
        self.register_buffer("major_semi_axis", major_semi_axis, persistent=False)
        center_to_base=torch.tensor(center_to_base)
        self.register_buffer("center_to_base", center_to_base, persistent=False)
        #circumferential_halflength=torch.tensor(np.pi*endoradius)
    def forward(self,x):# l {0,logitudinal length}, c{-circumferential_halflength,circumferential_halflength} , r{0,thickness}
        l=x[...,0:1]
        c=x[...,1:2]
        r=x[...,2:3]
        longi=(l-self.center_to_base)/self.major_semi_axis
        oneminuslongisq=1.-longi**2.
        sqrt1minuslongisq=torch.sqrt(oneminuslongisq)
        slice_radius=self.minor_semi_axis* sqrt1minuslongisq
        theta=c/self.minor_semi_axis
        sintheta=torch.sin(theta)
        costheta=torch.cos(theta)
        func_x=slice_radius * costheta
        func_y=slice_radius * sintheta
        sqrt1minuslongisq_divide_minor=sqrt1minuslongisq/self.minor_semi_axis
        outward_normal_x=sqrt1minuslongisq_divide_minor* costheta
        outward_normal_y=sqrt1minuslongisq_divide_minor* sintheta
        outward_normal_z=-longi/self.major_semi_axis
        #outward_normal_z=Min(0.,outward_normal_z)
        outward_normal_correct=torch.sqrt(oneminuslongisq/self.minor_semi_axis**2.+outward_normal_z**2.)
        outward_normal_x=outward_normal_x/outward_normal_correct
        outward_normal_y=outward_normal_y/outward_normal_correct
        outward_normal_z=outward_normal_z/outward_normal_correct
        x_coord=func_x+outward_normal_x*r
        y_coord=func_y+outward_normal_y*r
        z_coord=-l+outward_normal_z*r
        cirvec_x=sintheta#circumferential vector is right hand rule with thumb towards apex
        cirvec_y=-costheta
        cirvec_z=torch.zeros_like(costheta)
        logivec_x=outward_normal_y*cirvec_z-outward_normal_z*cirvec_y
        logivec_y=outward_normal_z*cirvec_x-outward_normal_x*cirvec_z
        logivec_z=outward_normal_x*cirvec_y-outward_normal_y*cirvec_x
        return torch.cat((x_coord,y_coord,z_coord,logivec_x,logivec_y,logivec_z,cirvec_x,cirvec_y,cirvec_z,outward_normal_x,outward_normal_y,outward_normal_z),-1)
    
    
class HeatMyocardium_LCR_to_xyz_approx(torch.nn.Module):
    '''
    fixed:
        Key('longitudinal_length'): 
        Key('radius'): 
        Key("center_to_base"): 
        Key("thickness"): 
    inputs:
        Key("l"):
        Key("c"): 
        Key("r"): 
    return:
        Key("x"): 
        Key("y"): 
        Key("z"): 
        Key("logivec_x"): 
        Key("logivec_y"): 
        Key("logivec_z"): 
        Key("cirvec_x"): 
        Key("cirvec_y"): 
        Key("cirvec_z"): 
        Key("radialvec_x"): 
        Key("radialvec_y"): 
        Key("radialvec_z"): 
    '''
    def __init__(self,longilength,center_to_base,endoradius,thickness):
        super().__init__()
        minor_semi_axis=torch.tensor(endoradius)
        self.register_buffer("minor_semi_axis", minor_semi_axis, persistent=False)
        major_semi_axis=torch.tensor(longilength-center_to_base)
        self.register_buffer("major_semi_axis", major_semi_axis, persistent=False)
        center_to_base=torch.tensor(center_to_base)
        self.register_buffer("center_to_base", center_to_base, persistent=False)
        #circumferential_halflength=torch.tensor(np.pi*endoradius)
    def forward(self,x):# l {0,logitudinal length+thickness}, c{-circumferential_halflength-thickness,circumferential_halflength+thickness} , r{0,thickness}
        l=x[...,0:1]
        c=x[...,1:2]
        r=x[...,2:3]
        z_coord=l
        longi=(l-self.center_to_base)/(self.major_semi_axis+r)
        oneminuslongisq=1.-longi**2.
        sqrt1minuslongisq=torch.sqrt(oneminuslongisq)
        slice_radius=(self.minor_semi_axis+r)* sqrt1minuslongisq
        
        
        
        theta=c/(self.minor_semi_axis+r)
        sintheta=torch.sin(theta)
        costheta=torch.cos(theta)
        x_coord=slice_radius * costheta
        y_coord=slice_radius * sintheta
        return torch.cat((x_coord,y_coord,z_coord),-1)
    
    