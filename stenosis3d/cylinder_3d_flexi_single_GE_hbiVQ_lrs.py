
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
    CosSinArch,
    MinPoolArch,
    MeanPoolArch,
    SubtractionArch,
    CustomDualInputModuleArch,
    AdditionArch,
    MultiplicationArch
)
from modulusDL.solver.solver import Solver_ReduceLROnPlateauLoss
from modulusDL.eq.pde import NavierStokes_CoordTransformed,Diffusion_CoordTransformed
import shutil
from collections import OrderedDict
import torch
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
        
        cline=Symbol("cline")
        radius_y=Symbol("radius_y")
        radius_z=Symbol("radius_z")
        vec_x=Symbol("vec_x")
        vec_y=Symbol("vec_y")
        vec_z=Symbol("vec_z")
        dissq=Symbol("dissq")
        
        
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
        cline_dydx=self.r*np.pi/self.l*(s1*torch.cos(np.pi*x_ref)+2.*s2*torch.cos(2.*np.pi*x_ref))
        cline_dzdx=self.r*np.pi/self.l*(s3*torch.cos(np.pi*x_ref)+2.*s4*torch.cos(2.*np.pi*x_ref))
        norm_y=1./torch.sqrt(cline_dydx**2.+1.)
        norm_xy=-norm_y*cline_dydx
        norm_z=1./torch.sqrt(cline_dzdx**2.+1.)
        norm_xz=-norm_z*cline_dzdx
        return torch.cat((x[...,0:1]+norm_xy*radius_y+norm_xz*radius_z,cline_y+norm_y*radius_y,cline_z+norm_z*radius_z,cline_dydx,cline_dzdx),-1)
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
class omegamultiplier(torch.nn.Module):
    def __init__(self,feature_num,freq_num):
        super().__init__()
        omega=np.repeat(range(1,freq_num+1),feature_num)[:freq_num].reshape((1,-1))*np.pi
        omega=torch.tensor(omega,dtype=torch.float)
        self.register_buffer("omega", omega, persistent=False)
    def forward(self,x):
        return self.omega*x
class velInit(torch.nn.Module):
    def __init__(self,u_max,init_p):
        super().__init__()
        u_max=torch.tensor(u_max)
        self.register_buffer("u_max", u_max, persistent=False)
        init_p=torch.tensor(init_p)
        self.register_buffer("init_p", init_p, persistent=False)
    def forward(self,x):#dissq,arearatio,vecx,vecy,vecz,Vin,cline
        vel=self.u_max*(1+x[...,5:6])*x[...,0:1]/x[...,1:2]
        p=self.init_p*(1.-x[...,6:7])*0.5 + 0.25*self.u_max**2.*(1.-1./x[...,1:2]**2.)
        return torch.cat((vel*x[...,2:5],p),dim=-1)
class powerhbound(torch.nn.Module):
    def __init__(self,power):
        super().__init__()
        power=torch.tensor(power)
        self.register_buffer("power", power, persistent=False)
    def forward(self,x):
        return torch.nn.functional.relu(1.-(1.-x)**self.power)
import itertools

param_ranges_key = OrderedDict()
param_ranges_key["s1"]=np.array([-6.,-2.,2.,6.])
param_ranges_key["s3"]=np.array([-6.,-2.,2.,6.])
param_ranges_key["A"]=np.array([0.,0.2,0.4,0.6])

caseID_param_values=np.array(list(itertools.product(*param_ranges_key.values())))
param_ranges = {
    Symbol("Vin"): 0.,
    Symbol("alpha"): 0.15,
    Symbol("r0"): 0.,
    Symbol("rr"): 0.,
    Symbol("s2"): -1.,
    Symbol("s4"): 1.,
    Symbol("s1"): 5.,
    Symbol("s3"): 4.,
    Symbol("A"): 0.5,
}


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    inputfiles = [f for f in os.listdir("/examples/stenosis3d/inputs/") if os.path.isfile(os.path.join("/examples/stenosis3d/inputs/", f)) and f[-4:]=='.csv']
    os.makedirs("/examples/stenosis3d/outputs/"+sys.argv[0][:-3], exist_ok = True) 
    if not(os.path.isfile("/examples/stenosis3d/outputs/"+sys.argv[0][:-3]+"/flow_network.0.pth")):
        print("Copy file ","/examples/stenosis3d/outputs/"+sys.argv[0][:-3]+"/flow_network.0.pth")
        shutil.copy("/examples/stenosis3d/outputs/"+sys.argv[0][:-13]+"/flow_network.0.pth","/examples/stenosis3d/outputs/"+sys.argv[0][:-3]+"/flow_network.0.pth")
    # physical quantities
    nu = quantity(0.0038, "kg/(m*s)")
    rho = quantity(1060, "kg/m^3")
    inlet_u = quantity(0.12, "m/s")
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    noslip_w = quantity(0.0, "m/s")
    outlet_p = quantity(0.0, "pa")
    init_p  = quantity(80.0, "pa")
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
        [Key("x_case"), Key("y_case"),Key("z_case"),Key("delta_yx"),Key("delta_zx")],
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
    interior_pts=volume_geo.sample_interior(16000)
    boundary_pts=volume_geo.sample_boundary(4000)
    inlet_pts=tubeinlet.sample_boundary(500)
    outlet_pts=tubeoutlet.sample_boundary(500)
    
    
    dissq_module=CustomModuleArch(
        [Key("y"),Key("z")],
        [Key("ref_dissq")],
        module=Dissq(nd.ndim(channel_radius))
        )
    ns = NavierStokes_CoordTransformed(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=3, time=False)
    #normal_dot_vel = NormalDotVec(["u", "v", "w"])
    
    flow_net = FullyConnectedArch(
        input_keys=[Key("x_case"), Key("y_case"), Key("z_case")],
        output_keys=[Key("cline"),Key("radius_y"),Key("radius_z"),Key("vec_x"),Key("vec_y"),Key("vec_z"),Key("dissq"),Key("arearatio")],
        layer_size=1024,
        nr_layers=4
    )
    result_net = FullyConnectedFlexiLayerSizeArch(
            input_keys=[Key("x_case"), Key("y_case"), Key("z_case"),Key("cline"),Key("radius_y"),Key("radius_z"),Key("vec_x"),Key("vec_y"),Key("vec_z"),Key("dissq"),Key("arearatio")],
            output_keys=[Key("predu"),Key("predv"),Key("predw"),Key("dp")],
            layer_sizeList=[1024,1024,1024,1024,1024,1024],
        )
    initVel_module=CustomModuleArch(
        [Key("dissq"),Key("arearatio"),Key("vec_x"),Key("vec_y"),Key("vec_z"),Key("Vin"),Key("cline")],
        [Key("u0"),Key("v0"),Key("w0"),Key("p0")],
        module=velInit(nd.ndim(inlet_u),nd.ndim(init_p))
        )
    vel_module=AdditionArch(
        input1_keys=[Key("u0"),Key("v0"),Key("w0"),Key("p0")],
        input2_keys=[Key("du"),Key("dv"),Key("dw"),Key("dp")],
        output_keys=[ Key("u"),Key("v"),Key("w"),Key("p")],
        )
    hbvelmode_net = CustomModuleArch(
        [Key("dissq")],
        [Key("hb")],
        module=powerhbound(4.)
        )
    sethb_net = MultiplicationArch(
        input1_keys=[Key("predu"),Key("predv"),Key("predw")],
        input2_keys=[Key("hb")],
        output_keys=[Key("du"),Key("dv"),Key("dw")],
    )
    Qsum_eq=vel_sumtorefQ(channel_length_nd[1]-channel_length_nd[0],len(tubechkpt))
    inlet_eq=ParabolicInlet( nd.ndim(channel_radius),channel_length_nd[1]-channel_length_nd[0],nd.ndim(inlet_u))
    #GE_eq=GE(nd.ndim(channel_radius),channel_length_nd[1]-channel_length_nd[0])
    nodes = (
        [Stenosis_coordTransform.make_node(name='coordtransform')]
        +[vec_ref_mod.make_node(name='vec_ref_mod')]
        +[dissq_module.make_node(name='dissq_net')]
        + ns.make_nodes()
        #+ normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + [result_net.make_node(name="result_net")]
        + [initVel_module.make_node(name="initVel_net")]
        + [vel_module.make_node(name="vel_net")]
        + [hbvelmode_net.make_node(name="hbvelmode_net")]
        + [sethb_net.make_node(name="sethb_net")]
        + [warp.make_node(name="newcoord")]
        + Scaler(
            ["u", "v","w","p"],
            ["u_scaled", "v_scaled", "w_scaled", "p_scaled"],
            ["m/s", "m/s","m/s", "m^2/s^2"],
            nd,
        ).make_node()
        #+GE_eq.make_nodes()
        +inlet_eq.make_nodes()
        +Qsum_eq.make_nodes()
    )
    
    flow_net.load_state_dict(
        torch.load(
            "/examples/stenosis3d/outputs/"+sys.argv[0][:-3]+"/flow_network.0.pth"
        ))
    for param in flow_net.parameters():
        param.requires_grad = False
    
    # make domain
    domain = Domain()
    x, y,z = Symbol("x"), Symbol("y"), Symbol("z")
    batchsizefactor=10
    batchsize_correction=2
   
    # inlet
    
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tubeinlet,
        outvar={"parabolic_inlet_x": 0., "parabolic_inlet_y": 0., "parabolic_inlet_z": 0.},
        batch_size=int(cfg.batch_size.inlet/batchsize_correction),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor*batchsize_correction,
    )
    domain.add_constraint(inlet, "inlet")
    
    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tubeoutlet,
        outvar={"p": nd.ndim(outlet_p)},
        batch_size=int(cfg.batch_size.inlet/batchsize_correction),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor*batchsize_correction,
    )
    domain.add_constraint(outlet, "outlet")
    # no slip
    '''
    no_slip_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tubewall,
        outvar={"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v), "w": nd.ndim(noslip_w)},
        batch_size=int(cfg.batch_size.walls,
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor*batchsize_correction,
    )
    domain.add_constraint(no_slip_wall, "no_slip_wall")
    '''
    # interior contraints
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=int(cfg.batch_size.interior/batchsize_correction),
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd, z: channel_width_nd}),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor*batchsize_correction,
    )
    domain.add_constraint(interior, "interior")
    #print("set integral to",0.5*nd.ndim(inlet_u)*cfg.batch_size.inlet)
    
    default_sumweight=0.01
    sameQ_outlet=IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=tubeoutlet,
        outvar={"vel_sumtorefQ": 0.5*nd.ndim(inlet_u)*np.pi*nd.ndim(channel_radius)**2.}, #  0.},#
        batch_size=1,
        integral_batch_size=int(cfg.batch_size.outlet/batchsize_correction),
        batch_per_epoch= 100*batchsizefactor*batchsize_correction,
        lambda_weighting ={"vel_sumtorefQ":default_sumweight},
    )
    domain.add_constraint(sameQ_outlet, "sameQ_outlet")
    '''
    sameQ_chkpt=[]
    for n in range(len(tubechkpt)):
        if n==0:
            set_lambda_weighting ={"vel_sumtorefQ_"+str(n):0.00001*default_sumweight}
        else:
            set_lambda_weighting ={"vel_sumtorefQ_"+str(n):default_sumweight}
        sameQ_chkpt.append(IntegralBoundaryConstraint(
            nodes=nodes,
            geometry=tubechkpt[n],
            outvar={"vel_sumtorefQ_"+str(n): 0.5*nd.ndim(inlet_u)*np.pi*nd.ndim(channel_radius)**2.}, # 0.},#
            batch_size=1,
            integral_batch_size=int(cfg.batch_size.outlet,
            lambda_weighting =set_lambda_weighting,
            batch_per_epoch= 100*batchsizefactor*batchsize_correction,
        ))
        domain.add_constraint(sameQ_chkpt[n], "sameQ_chkpt"+str(n)) 
    '''
    
    total_pts=21000
    openfoam_invar_numpy_in = tubeinlet.sample_boundary(
                cfg.batch_size.outlet,
                criteria=None,
                parameterization=Parameterization(
                        {
                            Symbol("Vin"): 0.,
                            Symbol("r0"): 0.,
                            Symbol("rr"): 0.,
                            Symbol("A"): 0.5,
                            Symbol("alpha"): 0.15,
                            Symbol("s1"): 5.,
                            Symbol("s2"): -1.,
                            Symbol("s3"): 4.,
                            Symbol("s4"): 1.,
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
                            Symbol("A"): 0.5,
                            Symbol("alpha"): 0.15,
                            Symbol("s1"): 5.,
                            Symbol("s2"): -1.,
                            Symbol("s3"): 4.,
                            Symbol("s4"): 1.,
                        }),
                quasirandom=False,
            )
    
    openfoam_invar_numpy={"x":np.concatenate((interior_pts["x"],boundary_pts["x"],inlet_pts["x"],outlet_pts["x"]),axis=0),
                          "y":np.concatenate((interior_pts["y"],boundary_pts["y"],inlet_pts["y"],outlet_pts["y"]),axis=0),
                          "z":np.concatenate((interior_pts["z"],boundary_pts["z"],inlet_pts["z"],outlet_pts["z"]),axis=0),
                          "Vin": np.repeat([0.],total_pts).reshape((-1,1)),
                          "r0": np.repeat([0.],total_pts).reshape((-1,1)),
                          "rr": np.repeat([0.],total_pts).reshape((-1,1)),
                          "A": np.repeat([0.5],total_pts).reshape((-1,1)),
                          "alpha": np.repeat([0.15],total_pts).reshape((-1,1)),
                          "s1": np.repeat([5.],total_pts).reshape((-1,1)),
                          "s2": np.repeat([-1.],total_pts).reshape((-1,1)),
                          "s3": np.repeat([4.],total_pts).reshape((-1,1)),
                          "s4": np.repeat([1.],total_pts).reshape((-1,1)),
                          }

    openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u_scaled", "v_scaled", "w_scaled", "p_scaled",'warpx','warpy','warpz',"ref_vec_x","ref_vec_y","ref_vec_z","cline", "radius_y", "radius_z", "vec_x", "vec_y", "vec_z", "dissq"]
    )
    domain.add_inferencer(openfoam_inferencer, "inf_data")
    openfoam_inferencer2=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy_in, output_names=["u_scaled", "v_scaled", "w_scaled", "p_scaled",'warpx','warpy','warpz',"ref_vec_x","ref_vec_y","ref_vec_z","cline", "radius_y", "radius_z", "vec_x", "vec_y", "vec_z", "dissq"]#,"vel_sumtorefQ","area"]
    )
    domain.add_inferencer(openfoam_inferencer2, "inlet_data")
    openfoam_inferencer3=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy_out, output_names=["u_scaled", "v_scaled", "w_scaled", "p_scaled",'warpx','warpy','warpz',"ref_vec_x","ref_vec_y","ref_vec_z","cline", "radius_y", "radius_z", "vec_x", "vec_y", "vec_z", "dissq"]#,"vel_sumtorefQ","area"]
    )
    domain.add_inferencer(openfoam_inferencer3, "outlet_data")
    
    global_monitor = PointwiseMonitor(
        volume_geo.sample_interior(1024),
        output_names=["continuity", "momentum_x", "momentum_y"],
        metrics={
            "mass_imbalance": lambda var: torch.sum(
                var["area"] * torch.abs(var["continuity"])
            ),
            "momentum_imbalance": lambda var: torch.sum(
                var["area"]
                * (torch.abs(var["momentum_x"]) + torch.abs(var["momentum_y"]))
            ),
        },
        nodes=nodes,
        requires_grad=True,
    )
    domain.add_monitor(global_monitor)


    # make solver
    slv = Solver_ReduceLROnPlateauLoss(cfg, domain,batch_per_epoch=1000*batchsizefactor*batchsize_correction,use_moving_average=False)

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
