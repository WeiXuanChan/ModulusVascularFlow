
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.insert(0,"/examples")
from modulusDL.models.arch import (
    CustomModuleArch,
    FullyConnectedFlexiLayerSizeArch, 
    CaseIDtoFeatureArch,
    CaseIDArch,
    FixedFeatureArch,
    MaxPoolArch,
    MatrixMultiplicationArch,
    SumMultiplicationArch,
    MatrixMultiplicationArchCore,
    FullyConnectedFlexiLayerSizeArchCore,
    MaxPoolArchCore,
    FixedFeatureArchCore,
    CaseIDArchCore,
    ParametricInsertArchCore,
    NominalSampledFixedFeatureArchCore
)
from modulusDL.eq.pde import NavierStokes_CoordTransformed
from modulusDL.solver.solver import Solver_ReduceLROnPlateauLoss
import shutil
from collections import OrderedDict

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

import modulus
from modulus.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.utils.io import csv_to_dict
from modulus.solver import Solver
from modulus.domain import Domain
from modulus.geometry import Bounds
from modulus.geometry.geometry import Geometry, csg_curve_naming
from modulus.geometry.curve import SympyCurve
from modulus.geometry.helper import _sympy_sdf_to_sdf
from modulus.geometry.parameterization import Parameterization, Parameter, Bounds
from modulus.models.fully_connected import FullyConnectedArch
from modulus.geometry.primitives_2d import Line, Circle, Channel2D
from modulus.eq.pdes.navier_stokes import NavierStokes
from modulus.eq.pdes.basic import NormalDotVec
from modulus.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)

from modulus.domain.inferencer import PointwiseInferencer
from modulus.utils.io.plotter import ValidatorPlotter, InferencerPlotter
from modulus.domain.validator import PointwiseValidator
from modulus.domain.monitor import PointwiseMonitor
from modulus.key import Key
from modulus import quantity
from modulus.eq.non_dim import NonDimensionalizer, Scaler
from modulus.eq.pde import PDE

import math
class HLine(Geometry):
    """
    2D Line parallel to y-axis

    Parameters
    ----------
    point_1 : tuple with 2 ints or floats
        lower bound point of line segment
    point_2 : tuple with 2 ints or floats
        upper bound point of line segment
    normal : int or float
        normal direction of line (+1 or -1)
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, normal=1, parameterization=Parameterization()):
        assert point_1[1] == point_2[1], "Points must have same y-coordinate"

        # make sympy symbols to use
        l = Symbol(csg_curve_naming(0))
        y = Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({l: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_y = point_2[0] - point_1[0]
        line_1 = SympyCurve(
            functions={
                "x": point_1[0] + l * dist_y,
                "y": point_1[1],
                "normal_x": 0,  # TODO rm 1e-10
                "normal_y": 1e-10 + normal,
            },
            parameterization=curve_parameterization,
            area=dist_y,
        )
        curves = [line_1]

        # calculate SDF
        sdf = normal * (point_1[1] - y)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize Line
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )
class tube(Geometry):
    """
    3D Cylinder
    Axis parallel to z-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of cylinder
    radius : int or float
        radius of cylinder
    height : int or float
        height of cylinder
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, length, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        l, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))

        # surface of the cylinder
        curve_parameterization = Parameterization(
            {l: (-1, 1), r: (0, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * l * length,
                "y": center[1] + radius * sin(theta),
                "z": center[2] + radius * cos(theta),
                "normal_x": 0,
                "normal_y": 1 * sin(theta),
                "normal_z": 1 * cos(theta),
            },
            parameterization=curve_parameterization,
            area=length * 2 * pi * radius,
        )
        curve_2 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * length,
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2] + sqrt(r) * radius * cos(theta),
                "normal_x": 1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=pi * radius**2,
        )
        curve_3 = SympyCurve(
            functions={
                "x": center[0] - 0.5 * length,
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2] + sqrt(r) * radius * cos(theta),
                "normal_x": -1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=pi * radius**2,
        )
        curves = [curve_1, curve_2, curve_3]

        # calculate SDF
        r_dist = sqrt((z - center[2]) ** 2 + (y - center[1]) ** 2)
        x_dist = Abs(x - center[0])
        outside_distance = sqrt(
            Min(0, radius - r_dist) ** 2 + Min(0, 0.5 * length - x_dist) ** 2
        )
        inside_distance = -1 * Min(
            Abs(Min(0, r_dist - radius)), Abs(Min(0, x_dist - 0.5 * length))
        )
        sdf = -(outside_distance + inside_distance)*(0.2+0.8*exp(-12.5*((x-center[0])/length)**2.))

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - length / 2, center[0] + length / 2),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - radius, center[2] + radius),
            },
            parameterization=parameterization,
        )

        # initialize Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )
class tubeInterior(Geometry):
    """
    3D Cylinder
    Axis parallel to z-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of cylinder
    radius : int or float
        radius of cylinder
    height : int or float
        height of cylinder
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, length, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        l, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))

        # surface of the cylinder
        curve_parameterization = Parameterization(
            {l: (-1, 1), r: (0, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * l * length,
                "y": center[1] + radius * r * sin(theta),
                "z": center[2] + radius * r * cos(theta),
                "normal_x": 0,
                "normal_y": 1 * sin(theta),
                "normal_z": 1 * cos(theta),
            },
            parameterization=curve_parameterization,
            area=length * pi * radius**2.,
        )
        
        curves = [curve_1]

        # calculate SDF
        sdf =  Min(0,radius - sqrt((y-center[1])**2 + (z-center[2])**2))

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - length / 2, center[0] + length / 2),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - radius, center[2] + radius),
            },
            parameterization=parameterization,
        )

        # initialize Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )
class tubeWall(Geometry):
    """
    3D Cylinder
    Axis parallel to z-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of cylinder
    radius : int or float
        radius of cylinder
    height : int or float
        height of cylinder
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, length, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        l= Symbol(csg_curve_naming(0))
        theta = Symbol(csg_curve_naming(1))

        # surface of the cylinder
        curve_parameterization = Parameterization(
            {l: (-1, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_1 = SympyCurve(
            functions={
                "x": center[0] + 0.5 * l * length,
                "y": center[1] + radius * sin(theta),
                "z": center[2] + radius * cos(theta),
                "normal_x": 0,
                "normal_y": 1 * sin(theta),
                "normal_z": 1 * cos(theta),
            },
            parameterization=curve_parameterization,
            area=length * 2 * pi * radius,
        )
        
        curves = [curve_1]

        # calculate SDF
        sdf = radius - sqrt((y-center[1])**2 + (z-center[2])**2)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - length / 2, center[0] + length / 2),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - radius, center[2] + radius),
            },
            parameterization=parameterization,
        )

        # initialize Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )
        def _boundary_criteria(criteria):
            def boundary_criteria(invar, params):
                return self.boundary_criteria(invar, criteria=criteria, params=params)

            return boundary_criteria

        closed_boundary_criteria = _boundary_criteria(None)

        for curve in self.curves:
            s, p = curve._sample(
            nr_points=10000,
            parameterization=parameterization,
            quasirandom=False,
            )
            computed_criteria = closed_boundary_criteria(s, p)
            total_area = np.sum(s["area"][computed_criteria[:, 0], :])
class tubeInlet(Geometry):
    """
    3D Cylinder
    Axis parallel to z-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of cylinder
    radius : int or float
        radius of cylinder
    height : int or float
        height of cylinder
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, length, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        r = Symbol(csg_curve_naming(0))
        theta = Symbol(csg_curve_naming(1))

        # surface of the cylinder
        curve_parameterization = Parameterization(
            {r: (0, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        curve_3 = SympyCurve(
            functions={
                "x": center[0] - 0.5 * length,
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2] + sqrt(r) * radius * cos(theta),
                "normal_x": -1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=pi * radius**2,
        )
        curves = [curve_3]

        # calculate SDF
        sdf = (center[0] - 0.5 * length - x)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - 0.5*length, center[0] - 0.5*length),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - radius, center[2] + radius),
            },
            parameterization=parameterization,
        )

        # initialize Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )
class tubeOutlet(Geometry):
    """
    3D Cylinder
    Axis parallel to z-axis

    Parameters
    ----------
    center : tuple with 3 ints or floats
        center of cylinder
    radius : int or float
        radius of cylinder
    height : int or float
        height of cylinder
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, center, radius, length=None, parameterization=Parameterization()):
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        r = Symbol(csg_curve_naming(0))
        theta = Symbol(csg_curve_naming(1))

        # surface of the cylinder
        curve_parameterization = Parameterization(
            {r: (0, 1), theta: (0, 2 * pi)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        if length is None:
            cir_center=center[0]
        else:
            cir_center=center[0] + 0.5 * length
        curve_2 = SympyCurve(
            functions={
                "x": cir_center,
                "y": center[1] + sqrt(r) * radius * sin(theta),
                "z": center[2] + sqrt(r) * radius * cos(theta),
                "normal_x": 1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=pi * radius**2,
        )
        
        curves = [curve_2]

        # calculate SDF
        sdf = (cir_center - x)

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (cir_center , cir_center),
                Parameter("y"): (center[1] - radius, center[1] + radius),
                Parameter("z"): (center[2] - radius, center[2] + radius),
            },
            parameterization=parameterization,
        )

        # initialize Cylinder
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )

class ParabolicInlet(PDE):
    def __init__(self, r_ref,l_ref,umax):
        # coordinates
        ref_vec_x=Symbol("ref_vec_x")
        ref_vec_y=Symbol("ref_vec_y")
        ref_vec_z=Symbol("ref_vec_z")
        ref_dissq=Symbol("ref_dissq")

        # make u function
        u=Symbol("u")
        v=Symbol("v")
        w=Symbol("w")
        
        # source term
        VelRatio=Symbol("Vin")
        umax=Number(umax)
        max_velocity=umax*(1+VelRatio)
        
        # set equations
        self.equations = {}
        self.equations["parabolic_inlet_x"] = (
            u- max_velocity*ref_dissq*ref_vec_x
        )  # "custom_pde" key name will be used in constraints
        self.equations["parabolic_inlet_y"] = (
            v- max_velocity*ref_dissq*ref_vec_y
        ) 
        self.equations["parabolic_inlet_z"] = (
            w- max_velocity*ref_dissq*ref_vec_z
        ) 
class GE(PDE):
    def __init__(self, r_ref,l_ref):
        # coordinates
        ref_x=Symbol("x")
        ref_y=Symbol("y")
        ref_z=Symbol("z")
        ref_vec_x=Symbol("ref_vec_x")
        ref_vec_y=Symbol("ref_vec_y")
        ref_vec_z=Symbol("ref_vec_z")
        ref_dissq=Symbol("ref_dissq")
        ref_arearatio=Symbol("ref_arearatio")
        
        cline=Symbol("cline")
        radius_y=Symbol("radius_y")
        radius_z=Symbol("radius_z")
        vec_x=Symbol("vec_x")
        vec_y=Symbol("vec_y")
        vec_z=Symbol("vec_z")
        dissq=Symbol("dissq")
        arearatio=Symbol("arearatio")
        
        
        # set equations
        self.equations = {}
        self.equations["cline_eq"] = (
            cline-2.*ref_x/l_ref
        )  # "custom_pde" key name will be used in constraints
        self.equations["radius_y_eq"] = (
            radius_y-ref_y/r_ref
        )
        self.equations["radius_z_eq"] = (
            radius_z-ref_z/r_ref
        )
        self.equations["vec_x_eq"] = (
            vec_x-ref_vec_x
        )
        self.equations["vec_y_eq"] = (
            vec_y-ref_vec_y
        )
        self.equations["vec_z_eq"] = (
            vec_z-ref_vec_z
        )
        self.equations["dissq_eq"] = (
            dissq-ref_dissq
        )
        self.equations["arearatio_eq"] = (
            arearatio-ref_arearatio
        )
class vel_sumtorefQ(PDE):
    def __init__(self,l,num,target=0.):
        # coordinates
        x = Symbol("x")
        Area=Symbol("area")
        u = Symbol("u")
        v = Symbol("v")
        w = Symbol("w")
        ref_vec_x=Symbol("ref_vec_x")
        ref_vec_y=Symbol("ref_vec_y")
        ref_vec_z=Symbol("ref_vec_z")
        VelRatio=Symbol("Vin")
        A=Symbol("A")
        alpha=Symbol("alpha")
        rr=Symbol("rr")
        r0=Symbol("r0")
        l = Number(l)
        target = Number(target)
        # set equations 
        self.equations = {}
        exp_term=(1.-A*exp(-1.*(x/alpha/l)**2./2.))
        exp_term0=(1.-A*exp(-0.25*(1./alpha)**2./2.))
        self.equations["vel_sumtorefQ"] = (
            (u*ref_vec_x+v*ref_vec_y+w*ref_vec_z)/(1+VelRatio)*((1+r0+rr*x*2./l)/(1+r0-rr)*exp_term/exp_term0)**2-target
        )  
        for n in range(num):
            self.equations["vel_sumtorefQ_"+str(n)] = (
                (u*ref_vec_x+v*ref_vec_y+w*ref_vec_z)/(1+VelRatio)*((1+r0+rr*x*2./l)/(1+r0-rr)*exp_term/exp_term0)**2-target
            )
import torch
from torch import Tensor

class Stenosis(torch.nn.Module):
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
        area_ratio=((r0+rr)/(r0-x[...,4:5])*(1.-A*torch.exp(alpha))/(1.-A*torch.exp(-0.25*(1./x[...,6:7])**2./2.)))**2.
        cline_dydx=self.r*np.pi/self.l*(s1*torch.cos(np.pi*x_ref)+2.*s2*torch.cos(2.*np.pi*x_ref))
        cline_dzdx=self.r*np.pi/self.l*(s3*torch.cos(np.pi*x_ref)+2.*s4*torch.cos(2.*np.pi*x_ref))
        norm_y=1./torch.sqrt(cline_dydx**2.+1.)
        norm_xy=-norm_y*cline_dydx
        norm_z=1./torch.sqrt(cline_dzdx**2.+1.)
        norm_xz=-norm_z*cline_dzdx
        return torch.cat((x[...,0:1]+norm_xy*radius_y+norm_xz*radius_z,cline_y+norm_y*radius_y,cline_z+norm_z*radius_z,cline_dydx,cline_dzdx,area_ratio),-1)
class vec_ref(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):#1:delta_xy,2:delta_xz,3:delta_y,4:delta_z
        cline_dydx=x[...,0:1]
        cline_dzdx=x[...,1:2]
        vec_x=1/torch.sqrt(cline_dydx**2.+cline_dzdx**2.+1)
        vec_y=cline_dydx*vec_x
        vec_z=cline_dzdx*vec_x
        return torch.cat((vec_x,vec_y,vec_z),-1)
class Dissq(torch.nn.Module):
    def __init__(self,r):
        super().__init__()
        r=torch.tensor(r)
        self.register_buffer("r", r, persistent=False)
    def forward(self,x):
        return 1-(x[...,0:1]**2.+x[...,1:2]**2.)/self.r**2.
class Cline(torch.nn.Module):
    def __init__(self,l):
        super().__init__()
        l=torch.tensor(l)
        self.register_buffer("l", l, persistent=False)
    def forward(self,x):
        return 2.*x/self.l
class warped(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x[...,3:6]-x[...,0:3]
class Newcoord(torch.nn.Module):
    def __init__(self,s):
        super().__init__()
        s=torch.tensor(s)
        self.register_buffer("s", s, persistent=False)
    def forward(self,x):
        return self.s*x[...,0:3]
class sumsq(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        mean_x=torch.mean(x,dim=0,keepdim=True)
        shifted=x-mean_x
        return shifted**2.
class varianceFromUniform(torch.nn.Module):
    def __init__(self,nlatent):
        super().__init__()
        uniVar=torch.tensor(np.ones((1,nlatent))*0.333,dtype=torch.float)
        self.register_buffer("uniVar", uniVar, persistent=False)
    def forward(self,x,range_x):
        input_max, max_indices=torch.max(torch.cat((range_x**2./12.,self.uniVar),dim=0),dim=0,keepdim=True)
        return torch.mean(x,dim=0,keepdim=True).expand(x.size())-input_max
class correlation(torch.nn.Module):
    def __init__(self,feature_num):
        super().__init__()
        combi1=np.zeros((feature_num,0))
        combi2=np.zeros((feature_num,0))
        for n in range(feature_num):
            for m in range(n+1,feature_num):
                temp=np.zeros((feature_num,1))
                temp[n,0]=1
                combi1=np.concatenate((combi1,temp),axis=1)
                temp=np.zeros((feature_num,1))
                temp[m,0]=1
                combi2=np.concatenate((combi2,temp),axis=1)
        combi1=torch.tensor(combi1,dtype=torch.float)
        self.register_buffer("combi1", combi1, persistent=False)
        combi2=torch.tensor(combi2,dtype=torch.float)
        self.register_buffer("combi2", combi2, persistent=False)
    def forward(self,x):
        mean_x=torch.mean(x,dim=0,keepdim=True)
        numerator=torch.sum(torch.matmul(x,self.combi1+self.combi2),dim=0)
        denominator=torch.matmul(mean_x,self.combi1)*torch.matmul(mean_x,self.combi2)
        out=(numerator/denominator).expand(x.size(0),self.combi1.size(1))
        return out
class latenbound(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        input_max, max_indices=torch.max(x**2.,dim=1,keepdim=True)
        return input_max
class latenmaxrelu(torch.nn.Module):
    def __init__(self,xmax):
        super().__init__()
        xmax=torch.tensor(xmax)
        self.register_buffer("xmax", xmax, persistent=False)
    def forward(self,x):
        return torch.relu(x-self.xmax)
class latenminrelu(torch.nn.Module):
    def __init__(self,xmin):
        super().__init__()
        xmin=torch.tensor(xmin)
        self.register_buffer("xmin", xmin, persistent=False)
    def forward(self,x):
        return torch.relu(self.xmin-x)
class tanhbound(torch.nn.Module):
    def __init__(self,amp):
        super().__init__()
        amp=torch.tensor(amp)
        self.register_buffer("amp", amp, persistent=False)
    def forward(self,x):
        return self.amp*torch.tanh(x)
class PointNet(torch.nn.Module):
    def __init__(self,bpts_sample,caseBoundarypt,pn_layer_sizeList1,pn_layer_sizeList2,total_modes,ge_features):
        super().__init__()
        caseNum=caseBoundarypt.shape[0]
        self.layers = torch.nn.ModuleList()
        self.layers.append(CaseIDArchCore(caseNum))#0
        self.layers.append(FixedFeatureArchCore(caseBoundarypt))#1
        self.layers.append(FullyConnectedFlexiLayerSizeArchCore(in_features = 3,
                                                                layer_sizeList = pn_layer_sizeList1))#2
        self.layers.append(MaxPoolArchCore(pooldim=1,keepdim=False))#3
        self.layers.append(FullyConnectedFlexiLayerSizeArchCore(in_features = pn_layer_sizeList1[-1],
                                                                layer_sizeList = pn_layer_sizeList2+[total_modes*ge_features],))#4
        
    def forward(self,x):#caseID
        caseBoundarypt=self.layers[1]()[...,:3]#(caseNum,bdpts,case_xyz)
        ptlatent=self.layers[2](caseBoundarypt)#(caseNum,bdpts,case_features)
        latent0, max_indices=torch.max(ptlatent,1,keepdim=False)
        fixed_ind=torch.cat([torch.index_select(caseBoundarypt[row:row+1],1,yind) for row, yind in enumerate(max_indices)]).detach()
        caseID_unit=self.layers[0](x)#(training points,caseNum)
        ptlatent=self.layers[2](fixed_ind)#(caseNum,bdpts,case_features)
        #latent0=self.layers[3](ptlatent)#(caseNum,case_features)
        latent0, max_indices=torch.max(ptlatent,1,keepdim=False)
        latent=self.layers[4](latent0)#(caseNum,bdpts,case_features)
        TPlatent=torch.matmul(caseID_unit,latent)
        return TPlatent
        
import itertools
param_ranges_key = OrderedDict()
param_ranges_key["s1"]=np.array([2.5,3.5,4.5,5.5])
param_ranges_key["s3"]=np.array([2.5,3.5,4.5,5.5])
param_ranges_key["A"]=np.array([0.25,0.35,0.45,0.55])


caseID_param_values=np.array(list(itertools.product(*param_ranges_key.values())))
caseID_param_values_total=caseID_param_values.shape[0]
caseID_param_testvalues=np.array([[5.,4.,0.5]])
param_ranges = {
    Symbol("Vin"): 0.,
    Symbol("alpha"): 0.15,
    Symbol("r0"): 0.,
    Symbol("rr"): 0.,
    Symbol("s2"): -1.,
    Symbol("s4"): 1.,
    Symbol("caseID"): np.arange(caseID_param_values.shape[0]).reshape((-1,1)),
}


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    inputfiles = [f for f in os.listdir("/examples/stenosis3dmulti/inputs/") if os.path.isfile(os.path.join("/examples/stenosis3d/inputs/", f)) and f[-4:]=='.csv']
    
    # physical quantities
    nu = quantity(0.0038, "kg/(m*s)")
    rho = quantity(1060, "kg/m^3")
    inlet_u = quantity(0.12, "m/s")
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    noslip_w = quantity(0.0, "m/s")
    outlet_p = quantity(0.0, "pa")
    velocity_scale = quantity(0.14, "m/s")
    density_scale = rho
    length_scale = quantity(0.012, "m")
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale ** 3),
    )

    # geometry
    channel_length = (quantity(-0.15, "m"), quantity(0.15, "m"))
    channel_width = (quantity(-0.006, "m"), quantity(0.006, "m"))
    channel_length_nd = tuple(map(lambda x: nd.ndim(x), channel_length))
    channel_width_nd = tuple(map(lambda x: nd.ndim(x), channel_width))
    channel_radius = quantity(0.006, "m")
    channel_center = (quantity(0., "m"),quantity(0., "m"),quantity(0., "m"))
    channel_center_nd = tuple(map(lambda x: nd.ndim(x), channel_center))
    pr = Parameterization(param_ranges)
    channel_checkpoint_nd=tuple(map(lambda x: nd.ndim(quantity(x, "m")), np.arange(-0.15,0.15,0.02)))
    channel = tubeWall(
        channel_center_nd,
        nd.ndim(channel_radius),
        channel_length_nd[1]-channel_length_nd[0],
        parameterization=pr,
    )
    tube_Interior=tubeInterior(
        channel_center_nd,
        nd.ndim(channel_radius),
        channel_length_nd[1]-channel_length_nd[0],
        parameterization=pr,
    )
    tubeinlet = tubeInlet(
        channel_center_nd,
        nd.ndim(channel_radius),
        channel_length_nd[1]-channel_length_nd[0],
        parameterization=pr,
    )
    tubeoutlet = tubeOutlet(
        channel_center_nd,
        nd.ndim(channel_radius),
        channel_length_nd[1]-channel_length_nd[0],
        parameterization=pr,
    )
    tubewall = tubeWall(
        channel_center_nd,
        nd.ndim(channel_radius),
        channel_length_nd[1]-channel_length_nd[0],
        parameterization=pr,
    )
    volume_geo = channel
    tubechkpt=[]
    for chkpt_x in channel_checkpoint_nd:
        
        tubechkpt.append(tubeOutlet(
            (chkpt_x,channel_center_nd[1],channel_center_nd[2]),
            nd.ndim(channel_radius),
            None,
            parameterization=pr,
        ))
    # make list of nodes to unroll graph on
    Stenosis_module=Stenosis(nd.ndim(channel_radius),channel_length_nd[1]-channel_length_nd[0])
    Stenosis_coordTransform=CustomModuleArch(
        [Key("x"), Key("y"), Key("z"),Key('r0'),Key('rr'),Key("A"),Key("alpha"),Key("s1"),Key("s2"),Key("s3"),Key("s4")],
        [Key("x_case"), Key("y_case"),Key("z_case"),Key("delta_yx"),Key("delta_zx"),Key("ref_arearatio")],
        module=Stenosis_module
        )
    vec_ref_mod=CustomModuleArch(
        [Key("delta_yx"),Key("delta_zx")],
        [Key("ref_vec_x"), Key("ref_vec_y"), Key("ref_vec_z")],
        module=vec_ref()
        )
    warp=CustomModuleArch(
        [Key('x'),Key('y'),Key('z'),Key("x_case"), Key("y_case"),Key("z_case")],
        [Key("warpx"), Key("warpy"), Key("warpz")],
        module=warped()
        )
    ref_dissq_module=CustomModuleArch(
        [Key("y"),Key("z")],
        [Key("ref_dissq")],
        module=Dissq(nd.ndim(channel_radius))
        )
    bdp_multiplier=8
    boundary_pts=volume_geo.sample_boundary(4000*bdp_multiplier)
    inlet_pts=tubeinlet.sample_boundary(500*bdp_multiplier)
    outlet_pts=tubeoutlet.sample_boundary(500*bdp_multiplier)
    total_pts=5000*bdp_multiplier
    addstr=''
    if bdp_multiplier!=1:
        addstr='x'+str(bdp_multiplier)
    if not(os.path.isfile("/examples/stenosis3dmulti/sboundary_pts"+addstr+".npy")):
        caseBoundary_pts={"x":np.concatenate((boundary_pts["x"],inlet_pts["x"],outlet_pts["x"]),axis=0),
                              "y":np.concatenate((boundary_pts["y"],inlet_pts["y"],outlet_pts["y"]),axis=0),
                              "z":np.concatenate((boundary_pts["z"],inlet_pts["z"],outlet_pts["z"]),axis=0),
                              "Vin": np.repeat([0.],total_pts).reshape((-1,1)),
                              "r0": np.repeat([0.],total_pts).reshape((-1,1)),
                              "rr": np.repeat([0.],total_pts).reshape((-1,1)),
                              "A": np.repeat([0.5],total_pts).reshape((-1,1)),
                              "alpha": np.repeat([0.05],total_pts).reshape((-1,1)),
                              "s1": np.repeat([5.],total_pts).reshape((-1,1)),
                              "s2": np.repeat([-1.],total_pts).reshape((-1,1)),
                              "s3": np.repeat([4.],total_pts).reshape((-1,1)),
                              "s4": np.repeat([1.],total_pts).reshape((-1,1)),
                              }
        caseBoundarypt=[]
        Stenosis_module.to(torch.device("cuda:0"))
        for n in range(caseID_param_values.shape[0]):
            caseBoundary_pts["s1"][:]=caseID_param_values[n,0]
            caseBoundary_pts["s3"][:]=caseID_param_values[n,1]
            caseBoundary_pts["A"][:]=caseID_param_values[n,2]
            temp_pt=np.concatenate((caseBoundary_pts["x"],
                                    caseBoundary_pts["y"],
                                    caseBoundary_pts["z"],
                                    caseBoundary_pts["r0"],
                                    caseBoundary_pts["rr"],
                                    caseBoundary_pts["A"],
                                    caseBoundary_pts["alpha"],
                                    caseBoundary_pts["s1"],
                                    caseBoundary_pts["s2"],
                                    caseBoundary_pts["s3"],
                                    caseBoundary_pts["s4"]),axis=-1)
            temp_pt=torch.tensor(temp_pt).to(torch.device("cuda:0"))
            caseBoundarypt.append(Stenosis_module(temp_pt).cpu().detach().numpy()[:,:3])
            del temp_pt
        for n in range(caseID_param_testvalues.shape[0]):
            caseBoundary_pts["s1"][:]=caseID_param_testvalues[n,0]
            caseBoundary_pts["s3"][:]=caseID_param_testvalues[n,1]
            caseBoundary_pts["A"][:]=caseID_param_testvalues[n,2]
            temp_pt=np.concatenate((caseBoundary_pts["x"],
                                    caseBoundary_pts["y"],
                                    caseBoundary_pts["z"],
                                    caseBoundary_pts["r0"],
                                    caseBoundary_pts["rr"],
                                    caseBoundary_pts["A"],
                                    caseBoundary_pts["alpha"],
                                    caseBoundary_pts["s1"],
                                    caseBoundary_pts["s2"],
                                    caseBoundary_pts["s3"],
                                    caseBoundary_pts["s4"]),axis=-1)
            temp_pt=torch.tensor(temp_pt).to(torch.device("cuda:0"))
            caseBoundarypt.append(Stenosis_module(temp_pt).cpu().detach().numpy()[:,:3])
            del temp_pt
        caseBoundarypt=np.array(caseBoundarypt)
        np.save("/examples/stenosis3dmulti/sboundary_pts"+addstr+".npy",caseBoundarypt)
        return 0
    else:
        caseBoundarypt=np.load("/examples/stenosis3dmulti/sboundary_pts"+addstr+".npy")
    total_modes=1024
    GE_features=["cline","radius_y","radius_z","vec_x","vec_y","vec_z","dissq","arearatio"]
    GE_features_nodissq=["cline","radius_y","radius_z","vec_x","vec_y","vec_z","arearatio"]
    caseNNSize=128
    caseID_net = CaseIDtoFeatureArch(
        input_key=Key("caseID"),
        output_keys=[Key(x) for x in list(param_ranges_key.keys())],
        feature_array=np.concatenate((caseID_param_values,caseID_param_testvalues),axis=0),
    )
    bpts_sample=8000
    PN_net = CustomModuleArch(
        [Key("caseID")],
        [Key("latent",size=16)],
        module=PointNet(bpts_sample,
                        caseBoundarypt,
                        [caseNNSize,caseNNSize,caseNNSize],
                        [caseNNSize,caseNNSize],
                        16,
                        1)
        )
    
    dissq_module=CustomModuleArch(
        [Key("radius_y"),Key("radius_z")],
        [Key("dissq")],
        module=Dissq(1.)
        )
    
    flow_net = FullyConnectedArch(
        input_keys=[Key("x_case"), Key("y_case"), Key("z_case"),Key("latent",size=16)],
        output_keys=[Key(x) for x in GE_features_nodissq],
        layer_size=2048,
        nr_layers=4,
    )
    GE_eq=GE(nd.ndim(channel_radius),channel_length_nd[1]-channel_length_nd[0])
    print("get nodes")
    nodes = (
        [Stenosis_coordTransform.make_node(name='coordtransform')]
        +[caseID_net.make_node(name='caseID_net')]
        +[vec_ref_mod.make_node(name='vec_ref_mod')]
        +[dissq_module.make_node(name='dissq_net')]
        +[ref_dissq_module.make_node(name='ref_dissq_net')]
        #+ ns.make_nodes()
        #+ normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + [PN_net.make_node(name="PN_net")]
        + [warp.make_node(name="newcoord")]
        +GE_eq.make_nodes()
    )
    
    # make domain
    domain = Domain()
    x, y,z = Symbol("x"), Symbol("y"), Symbol("z")
    batchsizefactor=20
    batchreductionfactor=4
    fixed_dataset=False
    quasirandom=True
   
    
        # inlet
        
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tubeinlet,
        outvar={"cline_eq":0., "radius_y_eq":0., "radius_z_eq":0., "vec_x_eq":0., "vec_y_eq":0., "vec_z_eq":0., "dissq_eq":0., "arearatio_eq":0.},
        batch_size=int(cfg.batch_size.inlet/batchreductionfactor),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor*batchreductionfactor,
        fixed_dataset = fixed_dataset,
        quasirandom =quasirandom
    )
    domain.add_constraint(inlet, "inlet")
    
    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tubeoutlet,
        outvar={"cline_eq":0., "radius_y_eq":0., "radius_z_eq":0., "vec_x_eq":0., "vec_y_eq":0., "vec_z_eq":0., "dissq_eq":0., "arearatio_eq":0.},
        batch_size=int(cfg.batch_size.inlet/batchreductionfactor),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor*batchreductionfactor,
        fixed_dataset = fixed_dataset,
        quasirandom =quasirandom
    )
    domain.add_constraint(outlet, "outlet")
    # no slip
    
    no_slip_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tubewall,
        outvar={"cline_eq":0., "radius_y_eq":0., "radius_z_eq":0., "vec_x_eq":0., "vec_y_eq":0., "vec_z_eq":0., "dissq_eq":0., "arearatio_eq":0.},
        batch_size=int(cfg.batch_size.walls/batchreductionfactor),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor*batchreductionfactor,
        fixed_dataset = fixed_dataset,
        quasirandom =quasirandom
    )
    domain.add_constraint(no_slip_wall, "no_slip_wall")
    # interior contraints
    #interior = PointwiseInteriorConstraint(
    #    nodes=nodes,
    #    geometry=volume_geo,
    #    outvar={"cline_eq":0., "radius_y_eq":0., "radius_z_eq":0., "vec_x_eq":0., "vec_y_eq":0., "vec_z_eq":0., "dissq_eq":0., "arearatio_eq":0.},
    #    batch_size=int(cfg.batch_size.interior/batchreductionfactor),
    #    bounds=None,#bounds=Bounds({x: channel_length_nd, y: channel_width_nd, z: channel_width_nd}),
    #    parameterization=param_ranges_withxyz,
    #    batch_per_epoch= 1000*batchsizefactor*batchreductionfactor,
    #    fixed_dataset = fixed_dataset,
    #    quasirandom =quasirandom
    #)
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tube_Interior,
        outvar={"cline_eq":0., "radius_y_eq":0., "radius_z_eq":0., "vec_x_eq":0., "vec_y_eq":0., "vec_z_eq":0., "dissq_eq":0., "arearatio_eq":0.},
        batch_size=int(cfg.batch_size.interior/batchreductionfactor),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor*batchreductionfactor,
        fixed_dataset = fixed_dataset,
        quasirandom =quasirandom
    )
    domain.add_constraint(interior, "interior")
    
    interior_pts=volume_geo.sample_interior(16000)
    boundary_pts=volume_geo.sample_boundary(4000)
    inlet_pts=tubeinlet.sample_boundary(500)
    outlet_pts=tubeoutlet.sample_boundary(500)
    total_pts=21000
    openfoam_invar_numpy_in = tubeinlet.sample_boundary(
                cfg.batch_size.outlet,
                criteria=None,
                parameterization=Parameterization(
                        {
                            Symbol("Vin"): 0.,
                            Symbol("r0"): 0.,
                            Symbol("rr"): 0.,
                            Symbol("alpha"): 0.15,
                            Symbol("s2"): -1.,
                            Symbol("s4"): 1.,
                            Symbol("caseID"): caseID_param_values.shape[0],
                        }),
                quasirandom=False,
            )
    openfoam_invar_numpy_out= tubeoutlet.sample_boundary(
                cfg.batch_size.outlet,
                criteria=None,
                parameterization=Parameterization(
                        {
                            Symbol("Vin"): 0.,
                            Symbol("r0"): 0.,
                            Symbol("rr"): 0.,
                            Symbol("alpha"): 0.15,
                            Symbol("s2"): -1.,
                            Symbol("s4"): 1.,
                            Symbol("caseID"): caseID_param_values.shape[0],
                        }),
                quasirandom=False,
            )
    
    openfoam_invar_numpy={"x":np.concatenate((interior_pts["x"],boundary_pts["x"],inlet_pts["x"],outlet_pts["x"]),axis=0),
                          "y":np.concatenate((interior_pts["y"],boundary_pts["y"],inlet_pts["y"],outlet_pts["y"]),axis=0),
                          "z":np.concatenate((interior_pts["z"],boundary_pts["z"],inlet_pts["z"],outlet_pts["z"]),axis=0),
                          "Vin": np.repeat([0.],total_pts).reshape((-1,1)),
                          "r0": np.repeat([0.],total_pts).reshape((-1,1)),
                          "rr": np.repeat([0.],total_pts).reshape((-1,1)),
                          "alpha": np.repeat([0.15],total_pts).reshape((-1,1)),
                          "s2": np.repeat([-1.],total_pts).reshape((-1,1)),
                          "s4": np.repeat([1.],total_pts).reshape((-1,1)),
                          "caseID": np.repeat([caseID_param_values.shape[0]],total_pts).reshape((-1,1)),
                          }

    openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=['warpx','warpy','warpz',"ref_vec_x","ref_vec_y","ref_vec_z","ref_arearatio","cline", "radius_y", "radius_z", "vec_x", "vec_y", "vec_z", "dissq","arearatio","ref_dissq","ref_vec_x","ref_vec_y","ref_vec_z","latent"]
    )
    domain.add_inferencer(openfoam_inferencer, "inf_data")
    openfoam_inferencer2=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy_in, output_names=['warpx','warpy','warpz',"ref_vec_x","ref_vec_y","ref_vec_z","ref_arearatio","cline", "radius_y", "radius_z", "vec_x", "vec_y", "vec_z", "dissq","arearatio"]#,"vel_sumtorefQ","area"]
    )
    domain.add_inferencer(openfoam_inferencer2, "inlet_data")
    openfoam_inferencer3=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy_out, output_names=['warpx','warpy','warpz',"ref_vec_x","ref_vec_y","ref_vec_z","ref_arearatio","cline", "radius_y", "radius_z", "vec_x", "vec_y", "vec_z", "dissq","arearatio"]#,"vel_sumtorefQ","area"]
    )
    domain.add_inferencer(openfoam_inferencer3, "outlet_data")
    
    
    interior_pts=volume_geo.sample_interior(65)
    total_pts=65
    openfoam_invar_numpy={"x":interior_pts["x"],
                          "y":interior_pts["y"],
                          "z":interior_pts["z"],
                          "Vin": np.repeat([0.],total_pts).reshape((-1,1)),
                          "r0": np.repeat([0.],total_pts).reshape((-1,1)),
                          "rr": np.repeat([0.],total_pts).reshape((-1,1)),
                          "alpha": np.repeat([0.15],total_pts).reshape((-1,1)),
                          "s2": np.repeat([-1.],total_pts).reshape((-1,1)),
                          "s4": np.repeat([1.],total_pts).reshape((-1,1)),
                          "caseID": np.arange(65).reshape((-1,1)),#np.repeat([0.],total_pts).reshape((-1,1)),
                          }
    openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["caseID","latent"]
    )
    #domain.add_inferencer(openfoam_inferencer, "eval_latent"+str(runevalID))
    domain.add_inferencer(openfoam_inferencer, "eval_latent_all")
    
    # make solver
    slv = Solver_ReduceLROnPlateauLoss(cfg, domain,batch_per_epoch=1000*batchsizefactor*batchreductionfactor,ReduceLROnPlateau_DictConFig={"min_lr":cfg.optimizer.lr*0.001})
    print("start solve")
    # start solver
    slv.eval()

if __name__ == "__main__":
    run()
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    latent_all=[]
    for n in range(1):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName("/examples/stenosis3dmulti/outputs/"+sys.argv[0][:-3]+"/inferencers/eval_latent_all.vtp")#"+str(n)+"
        reader.Update()
        polydata = reader.GetOutput()
        pointData = polydata.GetPointData()
        name1 = pointData.GetArray('latent')
        latent_array = vtk_to_numpy(name1)
        #latent_all.append(np.array(latent_array[1]))
    #latent_all=np.array(latent_all)
    latent_all=np.array(latent_array)
    print('latent_all.shape',latent_all.shape)
    np.save("/examples/stenosis3dmulti/outputs/"+sys.argv[0][:-3]+"/inferencers/eval_latent_all",latent_all)
                
            