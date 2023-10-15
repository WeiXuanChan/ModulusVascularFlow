
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
sys.path.insert(0,"/examples")
from modulusDL.models.arch import CustomModuleArch,FullyConnectedFlexiLayerSizeArch, CaseIDtoFeatureArch,CaseIDArch,FixedFeatureArch,MaxPoolArch,MatrixMultiplicationArch,SumMultiplicationArch
from modulusDL.eq.pde import NavierStokes_CoordTransformed
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
from modulusDL.solver.solver import Solver_ReduceLROnPlateauLoss

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
import itertools
straighten=0.
shorten=0.2
param_ranges_key = OrderedDict()
param_ranges_key["s1"]=np.array([-6.,-2.,2.,6.])
param_ranges_key["s3"]=np.array([-6.,-2.,2.,6.])
param_ranges_key["A"]=np.array([0.,0.2,0.4,0.6])

caseID_param_values=np.array(list(itertools.product(*param_ranges_key.values())))
param_ranges = {
    Symbol("Vin"): 0.,
    Symbol("alpha"): 0.05,
    Symbol("r0"): 0.,
    Symbol("rr"): 0.,
    Symbol("s2"): -1.*straighten,
    Symbol("s4"): 1.*straighten,
    Symbol("s1"): 5.*straighten,
    Symbol("s3"): 4.*straighten,
    Symbol("A"): 0.5,
}


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    inputfiles = [f for f in os.listdir("/examples/stenosis3d/inputs/") if os.path.isfile(os.path.join("/examples/stenosis3d/inputs/", f)) and f[-4:]=='.csv']
    
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
    channel_length = (quantity(-0.15*shorten, "m"), quantity(0.15*shorten, "m"))
    channel_width = (quantity(-0.006, "m"), quantity(0.006, "m"))
    channel_length_nd = tuple(map(lambda x: nd.ndim(x), channel_length))
    channel_width_nd = tuple(map(lambda x: nd.ndim(x), channel_width))
    channel_radius = quantity(0.006, "m")
    channel_center = (quantity(0., "m"),quantity(0., "m"),quantity(0., "m"))
    channel_center_nd = tuple(map(lambda x: nd.ndim(x), channel_center))
    pr = Parameterization(param_ranges)
    channel_checkpoint_nd=tuple(map(lambda x: nd.ndim(quantity(x, "m")), np.arange(-0.15*shorten,0.15*shorten,0.02)))
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
    normal_dot_vel = NormalDotVec(["u", "v", "w"])
    
    flow_net = FullyConnectedArch(
        input_keys=[Key("x_case"), Key("y_case"), Key("z_case")],
        output_keys=[Key("u"),Key("v"),Key("w"),Key("p")],
        layer_size=1024,
    )
    Qsum_eq=vel_sumtorefQ(channel_length_nd[1]-channel_length_nd[0],len(tubechkpt))
    inlet_eq=ParabolicInlet( nd.ndim(channel_radius),channel_length_nd[1]-channel_length_nd[0],nd.ndim(inlet_u))
    nodes = (
        [Stenosis_coordTransform.make_node(name='coordtransform')]
        +[vec_ref_mod.make_node(name='vec_ref_mod')]
        +[dissq_module.make_node(name='dissq_net')]
        + ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + [warp.make_node(name="newcoord")]
        + Scaler(
            ["u", "v","w","p"],
            ["u_scaled", "v_scaled", "w_scaled", "p_scaled"],
            ["m/s", "m/s","m/s", "m^2/s^2"],
            nd,
        ).make_node()
        +inlet_eq.make_nodes()
        +Qsum_eq.make_nodes()
    )
        
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
        batch_per_epoch= 1000*batchsizefactor,
    )
    domain.add_constraint(inlet, "inlet")
    
    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tubeoutlet,
        outvar={"p": nd.ndim(outlet_p)},
        batch_size=int(cfg.batch_size.inlet/batchsize_correction),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor,
    )
    domain.add_constraint(outlet, "outlet")
    # no slip
    
    no_slip_wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=tubewall,
        outvar={"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v), "w": nd.ndim(noslip_w)},
        batch_size=int(cfg.batch_size.walls/batchsize_correction),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor,
    )
    domain.add_constraint(no_slip_wall, "no_slip_wall")
    # interior contraints
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        batch_size=int(cfg.batch_size.interior/batchsize_correction),
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd, z: channel_width_nd}),
        parameterization=param_ranges,
        batch_per_epoch= 1000*batchsizefactor,
    )
    domain.add_constraint(interior, "interior")
    print("set integral to",0.5*nd.ndim(inlet_u)*cfg.batch_size.inlet)
    
    default_sumweight=1./len(tubechkpt)
    sameQ_outlet=IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=tubeoutlet,
        outvar={"vel_sumtorefQ": 0.5*nd.ndim(inlet_u)*np.pi*nd.ndim(channel_radius)**2.}, #  0.},#
        batch_size=1,
        integral_batch_size=int(cfg.batch_size.outlet/batchsize_correction),
        batch_per_epoch= 100*batchsizefactor,
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
            integral_batch_size=cfg.batch_size.outlet,
            lambda_weighting =set_lambda_weighting,
            batch_per_epoch= 100*batchsizefactor,
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
                            Symbol("alpha"): 0.05,
                            Symbol("s1"): 5.*straighten,
                            Symbol("s2"): -1.*straighten,
                            Symbol("s3"): 4.*straighten,
                            Symbol("s4"): 1.*straighten,
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
                            Symbol("alpha"): 0.05,
                            Symbol("s1"): 5.*straighten,
                            Symbol("s2"): -1.*straighten,
                            Symbol("s3"): 4.*straighten,
                            Symbol("s4"): 1.*straighten,
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
                          "alpha": np.repeat([0.05],total_pts).reshape((-1,1)),
                          "s1": np.repeat([5.*straighten],total_pts).reshape((-1,1)),
                          "s2": np.repeat([-1.*straighten],total_pts).reshape((-1,1)),
                          "s3": np.repeat([4.*straighten],total_pts).reshape((-1,1)),
                          "s4": np.repeat([1.*straighten],total_pts).reshape((-1,1)),
                          }

    openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u_scaled", "v_scaled", "w_scaled", "p_scaled",'warpx','warpy','warpz',"ref_vec_x","ref_vec_y","ref_vec_z"]
    )
    domain.add_inferencer(openfoam_inferencer, "inf_data")
    openfoam_inferencer2=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy_in, output_names=["u_scaled", "v_scaled", "w_scaled", "p_scaled",'warpx','warpy','warpz',"ref_vec_x","ref_vec_y","ref_vec_z"]#,"vel_sumtorefQ","area"]
    )
    domain.add_inferencer(openfoam_inferencer2, "inlet_data")
    openfoam_inferencer3=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy_out, output_names=["u_scaled", "v_scaled", "w_scaled", "p_scaled",'warpx','warpy','warpz',"ref_vec_x","ref_vec_y","ref_vec_z"]#,"vel_sumtorefQ","area"]
    )
    domain.add_inferencer(openfoam_inferencer3, "outlet_data")
    
    # make solver
    slv = Solver_ReduceLROnPlateauLoss(cfg, domain,batch_per_epoch=1000*batchsizefactor*batchsize_correction,ReduceLROnPlateau_DictConFig={"min_lr":cfg.optimizer.lr*0.001})

    # start solver
    slv.solve()

if __name__ == "__main__":
    run()
