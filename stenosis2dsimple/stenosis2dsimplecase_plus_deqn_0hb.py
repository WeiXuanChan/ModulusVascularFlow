import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.insert(0,"/examples")
from modulusDL.models.arch import CustomModuleArch, InheritedFullyConnectedFlexiLayerSizeArch, SingleInputInheritedFullyConnectedFlexiLayerSizeArch, CaseIDtoFeatureArch
from modulusDL.eq.pde import NavierStokes_CoordTransformed
import shutil
            
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
class Channel2D_centerfocused(Geometry):
    """
    2D Channel (no bounding curves in x-direction)

    Parameters
    ----------
    point_1 : tuple with 2 ints or floats
        lower bound point of channel
    point_2 : tuple with 2 ints or floats
        upper bound point of channel
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, point_1, point_2, parameterization=Parameterization()):
        # make sympy symbols to use
        l = Symbol(csg_curve_naming(0))
        x = Symbol("x")
        y = Symbol("y")

        # curves for each side
        curve_parameterization = Parameterization({l: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        dist_x = point_2[0] - point_1[0]
        dist_y = point_2[1] - point_1[1]
        line_1 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_1[1],
                "normal_x": 0,
                "normal_y": -1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        line_2 = SympyCurve(
            functions={
                "x": l * dist_x + point_1[0],
                "y": point_2[1],
                "normal_x": 0,
                "normal_y": 1,
            },
            parameterization=curve_parameterization,
            area=dist_x,
        )
        curves = [line_1, line_2]

        # calculate SDF
        center_y = point_1[1] + (dist_y) / 2
        center_x = point_1[0] + (dist_x) / 2
        y_diff = Abs(y - center_y) - (point_2[1] - center_y)
        outside_distance = sqrt(Max(y_diff, 0) ** 2)
        inside_distance = Min(y_diff, 0)
        sdf = -(outside_distance + inside_distance)*(0.2+0.8*exp(-12.5*((x-center_x)/dist_x)**2.))#sigma=0.02

        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (point_1[0], point_2[0]),
                Parameter("y"): (point_1[1], point_2[1]),
            },
            parameterization=parameterization,
        )

        # initialize Channel2D
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=2,
            bounds=bounds,
            parameterization=parameterization,
        )


class ParabolicInlet(PDE):
    def __init__(self, r_ref,l_ref,umax):
        # coordinates
        dissq = Symbol("dissq")
        
        # make input variables

        # make u function
        u = Symbol("u")
        

        # source term
        umax=Number(umax)
        
        # set equations
        self.equations = {}
        self.equations["parabolic_inlet"] = (
            u- umax*dissq
        )  # "custom_pde" key name will be used in constraints
        
class vel_sumtorefQ(PDE):
    def __init__(self,l,num,target=0.):
        # coordinates
        x = Symbol("x")
        u = Symbol("u")
        A=Symbol("A")
        sigma=Symbol("sigma")
        l = Number(l)
        target = Number(target)
        # set equations 
        self.equations = {}
        exp_term=(1.-A/0.05*exp(-1.*(x/sigma/l)**2./2.))
        exp_term0=(1.-A/0.05*exp(-0.25*(1./sigma)**2./2.))
        for n in range(num):
            self.equations["vel_sumtorefQ_"+str(n)] = (
                u*(exp_term/exp_term0)-target
            )   
class dEQN(PDE):
    def __init__(self,ANorm,sigmaNorm):
        # coordinates
        A=Symbol("A")
        sigma=Symbol("sigma")
        input_variables = {"A": A, "sigma": sigma}
        continuity = Function("continuity")(*input_variables)
        momentum_x = Function("momentum_x")(*input_variables)
        momentum_y = Function("momentum_y")(*input_variables)
        parabolic_inlet = Function("parabolic_inlet")(*input_variables)
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        p = Function("p")(*input_variables)
        ANorm = Number(ANorm)
        sigmaNorm = Number(sigmaNorm)
        
        # set equations 
        self.equations = {}
        self.equations["deqn_dA_continuity"] =continuity.diff(A)*ANorm
        self.equations["deqn_dsigma_continuity"] =continuity.diff(sigma)*sigmaNorm
        self.equations["deqn_dA_momentum_x"] =momentum_x.diff(A)*ANorm
        self.equations["deqn_dsigma_momentum_x"] =momentum_x.diff(sigma)*sigmaNorm
        self.equations["deqn_dA_momentum_y"] =momentum_y.diff(A)*ANorm
        self.equations["deqn_dsigma_momentum_y"] =momentum_y.diff(sigma)*sigmaNorm
        self.equations["deqn_dA_u"] =u.diff(A)*ANorm
        self.equations["deqn_dsigma_u"] =u.diff(sigma)*sigmaNorm
        self.equations["deqn_dA_v"] =v.diff(A)*ANorm
        self.equations["deqn_dsigma_v"] =v.diff(sigma)*sigmaNorm
        self.equations["deqn_dA_p"] =p.diff(A)*ANorm
        self.equations["deqn_dsigma_p"] =p.diff(sigma)*sigmaNorm
        self.equations["deqn_dA_parabolic_inlet"] =parabolic_inlet.diff(A)*ANorm
        self.equations["deqn_dsigma_parabolic_inlet"] =parabolic_inlet.diff(sigma)*sigmaNorm
         
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
        x_ref=x[...,0:1]
        y_ref=x[...,1:2]
        A=x[...,2:3]
        sigma=x[...,3:4]
        y_case=y_ref*(1.-A/0.05*torch.exp(-1.*(x_ref/self.l)**2./(2.*sigma**2.)))
        return torch.cat((x_ref,y_case),-1)
class GE(torch.nn.Module):
    def __init__(self,r,l):
        super().__init__()
        r=torch.tensor(r)
        self.register_buffer("r", r, persistent=False)
        l=torch.tensor(l)
        self.register_buffer("l", l, persistent=False)#length of line
    def forward(self,x):
        x_case=x[...,0:1]
        y_case=x[...,1:2]
        A=x[...,2:3]
        sigma=x[...,3:4]
        radius=y_case/(1.-A/0.05*torch.exp(-1.*(x_case/self.l)**2./(2.*sigma**2.)))/self.r
        cline=2.*x_case/self.l
        return torch.cat((cline,radius),-1)
class Dissq(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return 1.-x[...,0:1]**2.
class warped(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x[...,2:4]-x[...,0:2]
class Newcoord(torch.nn.Module):
    def __init__(self,s):
        super().__init__()
        s=torch.tensor(s)
        self.register_buffer("s", s, persistent=False)
    def forward(self,x):
        return self.s*x[...,0:3]
class hardbound(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):#0:u_0,1:u_1,2:u_2,3:v_0,4:v_1,5:v_2,6:cline,7:radius_y,8:vec_x,9:vec_y
        parabola=1.-x[...,7:8]**2.
        u=(x[...,0:1]+x[...,1:2]*x[...,8:9]+x[...,2:3]*x[...,9:10]*x[...,7:8])*parabola
        v=(x[...,3:4]+x[...,4:5]*x[...,9:10]+x[...,5:6]*x[...,8:9]*x[...,7:8])*parabola
        return torch.cat((u,v),-1)
class incrorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):#cline,radius_y
        return torch.cat((x[...,0:1]**2.,x[...,1:2]**2.,x[...,0:1]*x[...,1:2],x[...,0:1]*x[...,2:3],x[...,1:2]*x[...,2:3]),-1)
caseID_param_keys=[Key("A"),Key("sigma")]
caseID_param_values=np.array([[0.015,0.1],
                              [0.022,0.1],
                              [0.028,0.1],
                              [0.035,0.1],
                              [0.015,0.122],
                              [0.022,0.122],
                              [0.028,0.122],
                              [0.035,0.122],
                              [0.015,0.148],
                              [0.022,0.148],
                              [0.028,0.148],
                              [0.035,0.148],
                              [0.015,0.18],
                              [0.022,0.18],
                              [0.028,0.18],
                              [0.035,0.18]])
caseID_param_testvalues=np.array([[0.025,0.134]])
param_ranges = {
    Symbol("caseID"): np.arange(caseID_param_values.shape[0]).reshape((-1,1)),
}

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    #inputfiles = [f for f in os.listdir("/examples/stenosis3d/inputs/") if os.path.isfile(os.path.join("/examples/stenosis3d/inputs/", f)) and f[-4:]=='.csv']
    #cfg.optimizer.lr=1e-06
    deqn_lambda=5e-7
    pthfiles=["flow_network.0.pth"]
    os.makedirs("/examples/stenosis2dsimple/outputs/"+str(os.path.basename(__file__))[:-3],exist_ok =True)
    folder=''
    folder="optimizer.lr="+str(cfg.optimizer.lr)+"/"
    for pthfile in pthfiles:
        if not(os.path.isfile(pthfile)):
            shutil.copy("/examples/stenosis2dsimple/outputs/"+str(os.path.basename(__file__))[:-11]+str(os.path.basename(__file__))[-6:-3]+"/"+pthfile, "/examples/stenosis2dsimple/outputs/"+folder+str(os.path.basename(__file__))[:-3]+"/"+pthfile)
            print("copy ""/examples/stenosis2dsimple/outputs/"+str(os.path.basename(__file__))[:-11]+str(os.path.basename(__file__))[-6:-3]+"/"+pthfile+" to "+ "/examples/stenosis2dsimple/outputs/"+folder+str(os.path.basename(__file__))[:-3]+"/"+pthfile)
    # physical quantities
    nu = quantity(0.00185, "kg/(m*s)")
    rho = quantity(1000, "kg/m^3")
    inlet_u = quantity(0.00925, "m/s")
    noslip_u = quantity(0.0, "m/s")
    noslip_v = quantity(0.0, "m/s")
    outlet_p = quantity(0.0, "pa")
    velocity_scale = inlet_u
    density_scale = rho
    length_scale = quantity(0.1, "m")
    nd = NonDimensionalizer(
        length_scale=length_scale,
        time_scale=length_scale / velocity_scale,
        mass_scale=density_scale * (length_scale ** 3),
    )

    # geometry
    channel_length = (quantity(-0.5, "m"), quantity(0.5, "m"))
    channel_width = (quantity(-0.05, "m"), quantity(0.05, "m"))
    channel_length_nd = tuple(map(lambda x: nd.ndim(x), channel_length))
    channel_width_nd = tuple(map(lambda x: nd.ndim(x), channel_width))
    channel_radius = quantity(0.05, "m")
    channel_center = (quantity(0., "m"),quantity(0., "m"))
    channel_center_nd = tuple(map(lambda x: nd.ndim(x), channel_center))
    pr = Parameterization(param_ranges)

    channel = Channel2D_centerfocused(
        (channel_length_nd[0], channel_width_nd[0]),  #x1,y1
        (channel_length_nd[1], channel_width_nd[1]),  #x2,y2
        parameterization=pr,
    )
    inlet_2d = Line(
        (channel_length_nd[0], channel_width_nd[0]),  #x1,y1
        (channel_length_nd[0], channel_width_nd[1]),  #x1,y2
        normal=-1,
        parameterization=pr,
    )
    outlet_2d = Line(
        (channel_length_nd[1], channel_width_nd[0]),  #x2,y1
        (channel_length_nd[1], channel_width_nd[1]),  #x2,y2
        normal=1,
        parameterization=pr,
    )
    wall_btm = HLine(
        (channel_length_nd[0], channel_width_nd[0]),  #x1, y1
        (channel_length_nd[1], channel_width_nd[0]),  #x2, y1
        normal=-1,
        parameterization=pr,
    )
    wall_top = HLine(
        (channel_length_nd[0], channel_width_nd[1]),  #x1,y2
        (channel_length_nd[1], channel_width_nd[1]),  #x2,y2
        normal=1,
        parameterization=pr,
    )
    volume_geo = channel

    # make list of nodes to unroll graph on
    Stenosis_coordTransform=CustomModuleArch(
        [Key("x"), Key("y"),Key("A"),Key("sigma")],
        [Key("x_case"), Key("y_case")],
        module=Stenosis(nd.ndim(channel_radius),channel_length_nd[1]-channel_length_nd[0])
        )
    ge_net=CustomModuleArch(
        [Key("x_case"), Key("y_case"), Key("A"),Key("sigma")],
        [Key("cline"), Key("radius")],
        module=GE(nd.ndim(channel_radius),channel_length_nd[1]-channel_length_nd[0])
        )
    warp=CustomModuleArch(
        [Key('x'),Key('y'),Key("x_case"), Key("y_case")],
        [Key("warpx"), Key("warpy")],
        module=warped()
        )
    
    ns = NavierStokes_CoordTransformed(nu=nd.ndim(nu), rho=nd.ndim(rho), dim=2, time=False)
    normal_dot_vel = NormalDotVec(["u", "v"])
    caseID_net = CaseIDtoFeatureArch(
        input_key=Key("caseID"),
        output_keys=caseID_param_keys,
        feature_array=np.concatenate((caseID_param_values,caseID_param_testvalues),axis=0),
    )
    Dissq_net=CustomModuleArch(
        [Key("radius")],
        [Key("dissq")],
        module=Dissq()
        )
    flow_net = SingleInputInheritedFullyConnectedFlexiLayerSizeArch(
        input_keys=[Key("x_case"), Key("y_case"),Key("cline"), Key("radius"),Key("dissq"),Key("cline_sq"),Key("radius_sq"),Key("cline_radius"),Key("cline_dissq"),Key("radius_dissq")],
        output_keys=[Key("u"), Key("v"), Key("p")],
        case_input_keys=[Key("A"), Key("sigma"),Key("caseID")],
        layer_sizeList=[256]*4,#200963 parameters in domainNN
        caseNN_layer_sizeList=[32,32,32,32,32,10],#2215243 trainable weights
        with_caseID=caseID_param_values.shape[0]+caseID_param_testvalues.shape[0],
    )
    #all_hb = CustomModuleArch(
    #    input_keys=[Key("u_0"),Key("u_1"),Key("u_2"), Key("v_0"), Key("v_1"),Key("v_2"), Key("cline"), Key("radius_y"),Key("vec_x"),Key("vec_y")],
    #    output_keys=[Key("u"),Key("v")],
    #    module=hardbound(),
    #)
    incrorder_NN = CustomModuleArch(
        input_keys=[Key("cline"),Key("radius"),Key("dissq")],
        output_keys=[Key("cline_sq"),Key("radius_sq"),Key("cline_radius"),Key("cline_dissq"),Key("radius_dissq")],
        module=incrorder(),
    )

    inlet_eq=ParabolicInlet(nd.ndim(channel_radius),channel_length_nd[1]-channel_length_nd[0],nd.ndim(inlet_u))
    dEQN_eq=dEQN(0.02,0.08)
    nodes = (
        [Stenosis_coordTransform.make_node(name='coordtransform')]
        +[caseID_net.make_node(name='caseID_net')]
        +[Dissq_net.make_node(name='Dissq_net')]
        +[ge_net.make_node(name='ge_net')]
        #+[all_hb.make_node(name='all_hb')]
        +[incrorder_NN.make_node(name='incrorder_NN')]
        + ns.make_nodes()
        + normal_dot_vel.make_nodes()
        + [flow_net.make_node(name="flow_network")]
        + [warp.make_node(name="newcoord")]
        + Scaler(
            ["u", "v","p"],
            ["u_scaled", "v_scaled", "p_scaled"],
            ["m/s", "m/s", "m^2/s^2"],
            nd,
        ).make_node()
        +inlet_eq.make_nodes()
        +dEQN_eq.make_nodes()
    )
        
   
    # if len(inputfiles)>0:
        # newcoord = CustomModuleArch(
            # input_keys=[Key("x_case"), Key("y_case"), Key("z_case")],
            # output_keys=[Key("newx"),Key("newy"),Key("newz")],
            # module=Newcoord(length_scale.magnitude),
        # )
        # nodes=nodes+[newcoord.make_node(name="newcoord")]
    # make domain
    domain = Domain()
    x, y = Symbol("x"), Symbol("y")
    batchsizefactor=1
    
        
    # inlet
    inlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_2d,
        outvar={"parabolic_inlet": 0.},
        batch_size=cfg.batch_size.inlet*batchsizefactor,
        parameterization=param_ranges,
    )
    domain.add_constraint(inlet, "inlet")

    # outlet
    outlet = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_2d,
        outvar={"p": nd.ndim(outlet_p)},
        batch_size=cfg.batch_size.outlet*batchsizefactor,
        parameterization=param_ranges,
    )
    domain.add_constraint(outlet, "outlet")

    # no slip
    no_slip_wall_btm = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_btm,
        outvar={"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v)},
        batch_size=cfg.batch_size.walls*batchsizefactor,
        parameterization=param_ranges,
    )
    domain.add_constraint(no_slip_wall_btm, "no_slip_wall")
    no_slip_wall_top = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_top,
        outvar={"u": nd.ndim(noslip_u), "v": nd.ndim(noslip_v)},
        batch_size=cfg.batch_size.walls*batchsizefactor,
        parameterization=param_ranges,
    )
    domain.add_constraint(no_slip_wall_top, "no_slip_wall")

    # interior contraints
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0},
        batch_size=cfg.batch_size.interior*batchsizefactor,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        parameterization=param_ranges,
    )
    domain.add_constraint(interior, "interior")
    
    #dEQN
    inlet_deqn = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=inlet_2d,
        outvar={"deqn_dA_parabolic_inlet": 0.,"deqn_dsigma_parabolic_inlet": 0.},
        batch_size=cfg.batch_size.inlet*batchsizefactor,
        parameterization=param_ranges,
        lambda_weighting={"deqn_dA_parabolic_inlet": deqn_lambda,"deqn_dsigma_parabolic_inlet": deqn_lambda}
    )
    domain.add_constraint(inlet_deqn, "inlet_deqn")
    outlet_deqn = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=outlet_2d,
        outvar={"deqn_dA_p": nd.ndim(outlet_p),"deqn_dsigma_p": nd.ndim(outlet_p)},
        batch_size=cfg.batch_size.outlet*batchsizefactor,
        parameterization=param_ranges,
        lambda_weighting={"deqn_dA_p": deqn_lambda,"deqn_dsigma_p": deqn_lambda}
    )
    domain.add_constraint(outlet_deqn, "outlet_deqn")
    no_slip_wall_btm_deqn = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_btm,
        outvar={"deqn_dA_u": nd.ndim(noslip_u),"deqn_dsigma_u": nd.ndim(noslip_u), "deqn_dA_v": nd.ndim(noslip_v), "deqn_dsigma_v": nd.ndim(noslip_v)},
        batch_size=cfg.batch_size.walls*batchsizefactor,
        parameterization=param_ranges,
        lambda_weighting={"deqn_dA_u": deqn_lambda,"deqn_dsigma_u": deqn_lambda,"deqn_dA_v": deqn_lambda,"deqn_dsigma_v": deqn_lambda}
    )
    domain.add_constraint(no_slip_wall_btm_deqn, "no_slip_wall_deqn")
    no_slip_wall_top_deqn = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=wall_top,
        outvar={"deqn_dA_u": nd.ndim(noslip_u),"deqn_dsigma_u": nd.ndim(noslip_u), "deqn_dA_v": nd.ndim(noslip_v), "deqn_dsigma_v": nd.ndim(noslip_v)},
        batch_size=cfg.batch_size.walls*batchsizefactor,
        parameterization=param_ranges,
        lambda_weighting={"deqn_dA_u": deqn_lambda,"deqn_dsigma_u": deqn_lambda,"deqn_dA_v": deqn_lambda,"deqn_dsigma_v": deqn_lambda}
    )
    domain.add_constraint(no_slip_wall_top_deqn, "no_slip_wall_deqn")
    interior_deqn = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=volume_geo,
        outvar={"deqn_dA_continuity": 0,"deqn_dsigma_continuity": 0, "deqn_dA_momentum_x": 0, "deqn_dsigma_momentum_x": 0, "deqn_dA_momentum_y": 0, "deqn_dsigma_momentum_y": 0},
        batch_size=cfg.batch_size.interior*batchsizefactor,
        bounds=Bounds({x: channel_length_nd, y: channel_width_nd}),
        parameterization=param_ranges,
        lambda_weighting={"deqn_dA_continuity": deqn_lambda,"deqn_dsigma_continuity": deqn_lambda,"deqn_dA_momentum_x": deqn_lambda,"deqn_dsigma_momentum_x": deqn_lambda,"deqn_dA_momentum_y": deqn_lambda,"deqn_dsigma_momentum_y": deqn_lambda}
    )
    domain.add_constraint(interior_deqn, "interior_deqn")
    
    interior_pts=volume_geo.sample_interior(1600)
    boundary_pts=volume_geo.sample_boundary(400)
    total_pts=2000
    openfoam_invar_numpy={"x":np.concatenate((interior_pts["x"],boundary_pts["x"]),axis=0),
                          "y":np.concatenate((interior_pts["y"],boundary_pts["y"]),axis=0),
                          "caseID": np.repeat([caseID_param_values.shape[0]],total_pts).reshape((-1,1)),
                          }
    openfoam_invar_numpy_in = inlet_2d.sample_boundary(
                cfg.batch_size.outlet*batchsizefactor,
                criteria=None,
                parameterization=Parameterization(
                        {
                            Symbol("caseID"): caseID_param_values.shape[0],
                        }),
                quasirandom=False,
            )
    openfoam_invar_numpy_out= outlet_2d.sample_boundary(
                cfg.batch_size.outlet*batchsizefactor,
                criteria=None,
                parameterization=Parameterization(
                        {
                            Symbol("caseID"): caseID_param_values.shape[0],
                        }),
                quasirandom=False,
            )
    openfoam_inferencer=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy, output_names=["u_scaled", "v_scaled", "p_scaled",'warpx','warpy',"cline","radius"]
    )
    #domain.add_inferencer(openfoam_inferencer, "inf_data")
    openfoam_inferencer2=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy_in, output_names=["u_scaled", "v_scaled", "p_scaled",'warpx','warpy',"cline","radius"]
    )
    #domain.add_inferencer(openfoam_inferencer2, "inlet_data")
    openfoam_inferencer2=PointwiseInferencer(
    nodes=nodes, invar=openfoam_invar_numpy_out, output_names=["u_scaled", "v_scaled", "p_scaled",'warpx','warpy',"cline","radius"]
    )
    #domain.add_inferencer(openfoam_inferencer2, "outlet_data")
    # for ifile in inputfiles:
        # mapping = {
            # "x": "x",
            # "y": "y",
            # "z": "z",
        # }
        # openfoam_var = csv_to_dict(
            # ifile, mapping
        # )
        # openfoam_invar_numpy_alt={}
        # if run_geometry_encoding_only:
            # alt_output_names=['newx','newy',"newz"]
        # else:
            # alt_output_names=["u_scaled", "v_scaled", "w_scaled", "p_scaled",'warpx','warpy','warpz',"cline","radius_y","radius_z","vec_x","vec_y","vec_z"]
        # for key, value in openfoam_var.items():
            # if key in ["x", "y","z"]:
                # openfoam_invar_numpy_alt[key]=value / length_scale.magnitude
        # openfoam_inferencer_alt=PointwiseInferencer(
        # nodes=nodes, invar=openfoam_invar_numpy_alt, output_names=alt_output_names
        # )
        # domain.add_inferencer(openfoam_inferencer_alt, ifile[:-4])
    
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.eval()

if __name__ == "__main__":
    run()
