"""Equations related to Navier Stokes Equations with coordinate transformation
"""

from sympy import Symbol, Function, Number
from sympy import exp , Matrix
import numpy as np

from modulus.eq.pde import PDE
from modulus.node import Node

from typing import Optional, Callable, List, Dict, Union, Tuple
from modulus.key import Key
import torch
class NavierStokes_CoordTransformed(PDE):
    """
    This class is adapted from NVIDIAModulus v22.09 pde.NavierStokes
    the coordinates are adapted to allow for user defined str
    Compressible Navier Stokes equations

    Parameters
    ==========
    nu : float, Sympy Symbol/Expr, str
        The kinematic viscosity. If `nu` is a str then it is
        converted to Sympy Function of form `nu(x,y,z,t)`.
        If `nu` is a Sympy Symbol or Expression then this
        is substituted into the equation. This allows for
        variable viscosity.
    rho : float, Sympy Symbol/Expr, str
        The density of the fluid. If `rho` is a str then it is
        converted to Sympy Function of form 'rho(x,y,z,t)'.
        If 'rho' is a Sympy Symbol or Expression then this
        is substituted into the equation to allow for
        compressible Navier Stokes. Default is 1.
    dim : int
        Dimension of the Navier Stokes (2 or 3). Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the Navier-Stokes equations.

    Examples
    ========
    >>> ns = NavierStokes(nu=0.01, rho=1, dim=2)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: u*u__x + v*u__y + p__x + u__t - 0.01*u__x__x - 0.01*u__y__y
      momentum_y: u*v__x + v*v__y + p__y + v__t - 0.01*v__x__x - 0.01*v__y__y
    >>> ns = NavierStokes(nu='nu', rho=1, dim=2, time=False)
    >>> ns.pprint()
      continuity: u__x + v__y
      momentum_x: -nu*u__x__x - nu*u__y__y + u*u__x + v*u__y - nu__x*u__x - nu__y*u__y + p__x
      momentum_y: -nu*v__x__x - nu*v__y__y + u*v__x + v*v__y - nu__x*v__x - nu__y*v__y + p__y
    """

    name = "NavierStokes_CoordTransformed"

    def __init__(self, nu,case_coord_strList=None,case_param_strList=None, rho=1, dim=3, time=True, mixed_form=False,Reynolds=False):
        # set params
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form
        if case_param_strList is None:
            case_param_strList={}
        if case_coord_strList is None:
            case_coord_strList=["x_case","y_case","z_case"]
        if (case_coord_strList)==1:
            case_coord_strList=case_coord_strList+["y_case","z_case"]
        elif (case_coord_strList)==2:
            case_coord_strList=case_coord_strList+["z_case"]
        # coordinates
        t= Symbol("t")

        x = Symbol(case_coord_strList[0])
        y = Symbol(case_coord_strList[1])
        z = Symbol(case_coord_strList[2])
        input_variables = {case_coord_strList[0]: x, case_coord_strList[1]: y, case_coord_strList[2]: z, "t": t}
        for key in case_param_strList:
            input_variables[key]=case_param_strList[key]
        if self.dim == 2:
            input_variables.pop("z_case")
        if not self.time:
            input_variables.pop("t")

        # velocity componets
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        if self.dim == 3:
            w = Function("w")(*input_variables)
        else:
            w = Number(0)

        # pressure
        p = Function("p")(*input_variables)

        # kinematic viscosity
        if isinstance(nu, str):
            nu = Function(nu)(*input_variables)
        elif isinstance(nu, (float, int)):
            nu = Number(nu)

        # density
        if isinstance(rho, str):
            rho = Function(rho)(*input_variables)
        elif isinstance(rho, (float, int)):
            rho = Number(rho)

        # dynamic viscosity
        mu = rho * nu

        # set equations
        self.equations = {}
        self.equations["continuity"] = (
            rho.diff(t) + (rho * u).diff(x) + (rho * v).diff(y) + (rho * w).diff(z)
        )

        if not self.mixed_form:
            curl = Number(0) if rho.diff(x) == 0 else u.diff(x) + v.diff(y) + w.diff(z)
            self.equations["momentum_x"] = (
                (rho * u).diff(t)
                + (
                    u * ((rho * u).diff(x))
                    + v * ((rho * u).diff(y))
                    + w * ((rho * u).diff(z))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u.diff(x)).diff(x)
                - (mu * u.diff(y)).diff(y)
                - (mu * u.diff(z)).diff(z)
                - (mu * (curl).diff(x))
            )
            self.equations["momentum_y"] = (
                (rho * v).diff(t)
                + (
                    u * ((rho * v).diff(x))
                    + v * ((rho * v).diff(y))
                    + w * ((rho * v).diff(z))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v.diff(x)).diff(x)
                - (mu * v.diff(y)).diff(y)
                - (mu * v.diff(z)).diff(z)
                - (mu * (curl).diff(y))
            )
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w).diff(x))
                    + v * ((rho * w).diff(y))
                    + w * ((rho * w).diff(z))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w.diff(x)).diff(x)
                - (mu * w.diff(y)).diff(y)
                - (mu * w.diff(z)).diff(z)
                - (mu * (curl).diff(z))
            )
            if Reynolds is not False:
                ref_vec_x=Symbol(Reynolds[0])
                ref_vec_y=Symbol(Reynolds[1])
                ref_vec_z=Symbol(Reynolds[2])
                VELOCITY=(u*ref_vec_x+v*ref_vec_y+w*ref_vec_z)
                self.equations["Re_Inertial"] = (
                    (rho * VELOCITY).diff(t)
                    + (
                        VELOCITY * ((rho * VELOCITY).diff(x)*ref_vec_x
                        + (rho * VELOCITY).diff(y)*ref_vec_y
                        + (rho * VELOCITY).diff(z)*ref_vec_z
                        + rho * (curl))
                    )
                )
                vel_diff=VELOCITY.diff(x)*ref_vec_x+VELOCITY.diff(y)*ref_vec_y+VELOCITY.diff(z)*ref_vec_z
                
            
                self.equations["Re_Viscous"] = (
                    - (-2 / 3 * mu * (curl)).diff(x)*ref_vec_x
                    - (-2 / 3 * mu * (curl)).diff(y)*ref_vec_y
                    - (-2 / 3 * mu * (curl)).diff(z)*ref_vec_z
                    - (mu * vel_diff).diff(x)*ref_vec_x
                    - (mu * vel_diff).diff(y)*ref_vec_y
                    - (mu * vel_diff).diff(z)*ref_vec_z
                    - (mu * (curl).diff(x)*ref_vec_x)
                    - (mu * (curl).diff(y)*ref_vec_y)
                    - (mu * (curl).diff(z)*ref_vec_z)
                )
            if self.dim == 2:
                self.equations.pop("momentum_z")

        elif self.mixed_form:
            u_x = Function("u_x")(*input_variables)
            u_y = Function("u_y")(*input_variables)
            u_z = Function("u_z")(*input_variables)
            v_x = Function("v_x")(*input_variables)
            v_y = Function("v_y")(*input_variables)
            v_z = Function("v_z")(*input_variables)

            if self.dim == 3:
                w_x = Function("w_x")(*input_variables)
                w_y = Function("w_y")(*input_variables)
                w_z = Function("w_z")(*input_variables)
            else:
                w_x = Number(0)
                w_y = Number(0)
                w_z = Number(0)
                u_z = Number(0)
                v_z = Number(0)

            curl = Number(0) if rho.diff(x) == 0 else u_x + v_y + w_z
            self.equations["momentum_x"] = (
                (rho * u).diff(t)
                + (
                    u * ((rho * u.diff(x)))
                    + v * ((rho * u.diff(y)))
                    + w * ((rho * u.diff(z)))
                    + rho * u * (curl)
                )
                + p.diff(x)
                - (-2 / 3 * mu * (curl)).diff(x)
                - (mu * u_x).diff(x)
                - (mu * u_y).diff(y)
                - (mu * u_z).diff(z)
                - (mu * (curl).diff(x))
            )
            self.equations["momentum_y"] = (
                (rho * v).diff(t)
                + (
                    u * ((rho * v.diff(x)))
                    + v * ((rho * v.diff(y)))
                    + w * ((rho * v.diff(z)))
                    + rho * v * (curl)
                )
                + p.diff(y)
                - (-2 / 3 * mu * (curl)).diff(y)
                - (mu * v_x).diff(x)
                - (mu * v_y).diff(y)
                - (mu * v_z).diff(z)
                - (mu * (curl).diff(y))
            )
            self.equations["momentum_z"] = (
                (rho * w).diff(t)
                + (
                    u * ((rho * w.diff(x)))
                    + v * ((rho * w.diff(y)))
                    + w * ((rho * w.diff(z)))
                    + rho * w * (curl)
                )
                + p.diff(z)
                - (-2 / 3 * mu * (curl)).diff(z)
                - (mu * w_x).diff(x)
                - (mu * w_y).diff(y)
                - (mu * w_z).diff(z)
                - (mu * (curl).diff(z))
            )
            self.equations["compatibility_u_x"] = u.diff(x) - u_x
            self.equations["compatibility_u_y"] = u.diff(y) - u_y
            self.equations["compatibility_u_z"] = u.diff(z) - u_z
            self.equations["compatibility_v_x"] = v.diff(x) - v_x
            self.equations["compatibility_v_y"] = v.diff(y) - v_y
            self.equations["compatibility_v_z"] = v.diff(z) - v_z
            self.equations["compatibility_w_x"] = w.diff(x) - w_x
            self.equations["compatibility_w_y"] = w.diff(y) - w_y
            self.equations["compatibility_w_z"] = w.diff(z) - w_z
            self.equations["compatibility_u_xy"] = u_x.diff(y) - u_y.diff(x)
            self.equations["compatibility_u_xz"] = u_x.diff(z) - u_z.diff(x)
            self.equations["compatibility_u_yz"] = u_y.diff(z) - u_z.diff(y)
            self.equations["compatibility_v_xy"] = v_x.diff(y) - v_y.diff(x)
            self.equations["compatibility_v_xz"] = v_x.diff(z) - v_z.diff(x)
            self.equations["compatibility_v_yz"] = v_y.diff(z) - v_z.diff(y)
            self.equations["compatibility_w_xy"] = w_x.diff(y) - w_y.diff(x)
            self.equations["compatibility_w_xz"] = w_x.diff(z) - w_z.diff(x)
            self.equations["compatibility_w_yz"] = w_y.diff(z) - w_z.diff(y)

            if self.dim == 2:
                self.equations.pop("momentum_z")
                self.equations.pop("compatibility_u_z")
                self.equations.pop("compatibility_v_z")
                self.equations.pop("compatibility_w_x")
                self.equations.pop("compatibility_w_y")
                self.equations.pop("compatibility_w_z")
                self.equations.pop("compatibility_u_xz")
                self.equations.pop("compatibility_u_yz")
                self.equations.pop("compatibility_v_xz")
                self.equations.pop("compatibility_v_yz")
                self.equations.pop("compatibility_w_xy")
                self.equations.pop("compatibility_w_xz")
                self.equations.pop("compatibility_w_yz")
class Diffusion_CoordTransformed(PDE):
    """
    This class is adapted from NVIDIAModulus v22.09 pde.NavierStokes
    the coordinates are adapted to allow for user defined str
    Diffusion equation

    Parameters
    ==========
    T : str
        The dependent variable.
    D : float, Sympy Symbol/Expr, str
        Diffusivity. If `D` is a str then it is
        converted to Sympy Function of form 'D(x,y,z,t)'.
        If 'D' is a Sympy Symbol or Expression then this
        is substituted into the equation.
    Q : float, Sympy Symbol/Expr, str
        The source term. If `Q` is a str then it is
        converted to Sympy Function of form 'Q(x,y,z,t)'.
        If 'Q' is a Sympy Symbol or Expression then this
        is substituted into the equation. Default is 0.
    dim : int
        Dimension of the diffusion equation (1, 2, or 3).
        Default is 3.
    time : bool
        If time-dependent equations or not. Default is True.
    mixed_form: bool
        If True, use the mixed formulation of the diffusion equations.

    Examples
    ========
    >>> diff = Diffusion(D=0.1, Q=1, dim=2)
    >>> diff.pprint()
      diffusion_T: T__t - 0.1*T__x__x - 0.1*T__y__y - 1
    >>> diff = Diffusion(T='u', D='D', Q='Q', dim=3, time=False)
    >>> diff.pprint()
      diffusion_u: -D*u__x__x - D*u__y__y - D*u__z__z - Q - D__x*u__x - D__y*u__y - D__z*u__z
    """

    name = "Diffusion_CoordTransformed"

    def __init__(self, case_coord_strList=None,T="T", D="D", Q=0, dim=3, time=True, mixed_form=False):
        # set params
        self.T = T
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form

        # coordinates
        if case_coord_strList is None:
            case_coord_strList=["x_case","y_case","z_case"]
        if (case_coord_strList)==1:
            case_coord_strList=case_coord_strList+["y_case","z_case"]
        elif (case_coord_strList)==2:
            case_coord_strList=case_coord_strList+["z_case"]

        # time
        t = Symbol("t")
        x = Symbol(case_coord_strList[0])
        y = Symbol(case_coord_strList[1])
        z = Symbol(case_coord_strList[2])
        # make input variables
        input_variables = {case_coord_strList[0]: x, case_coord_strList[1]: y, case_coord_strList[2]: z, "t": t}
        if self.dim == 1:
            input_variables.pop("y_case")
            input_variables.pop("z_case")
        elif self.dim == 2:
            input_variables.pop("z_case")
        if not self.time:
            input_variables.pop("t")
        

        # Temperature
        assert type(T) == str, "T needs to be string"
        T = Function(T)(*input_variables)

        # Diffusivity
        if type(D) is str:
            D = Function(D)(*input_variables)
        elif type(D) in [float, int]:
            D = Number(D)

        # Source
        if type(Q) is str:
            Q = Function(Q)(*input_variables)
        elif type(Q) in [float, int]:
            Q = Number(Q)

        # set equations
        self.equations = {}

        if not self.mixed_form:
            self.equations["diffusion_" + self.T] = (
                T.diff(t)
                - (D * T.diff(x)).diff(x)
                - (D * T.diff(y)).diff(y)
                - (D * T.diff(z)).diff(z)
                - Q
            )
        elif self.mixed_form:
            T_x = Function("T_x")(*input_variables)
            T_y = Function("T_y")(*input_variables)
            if self.dim == 3:
                T_z = Function("T_z")(*input_variables)
            else:
                T_z = Number(0)

            self.equations["diffusion_" + self.T] = (
                T.diff(t)
                - (D * T_x).diff(x)
                - (D * T_y).diff(y)
                - (D * T_z).diff(z)
                - Q
            )
            self.equations["compatibility_T_x"] = T.diff(x) - T_x
            self.equations["compatibility_T_y"] = T.diff(y) - T_y
            self.equations["compatibility_T_z"] = T.diff(z) - T_z
            self.equations["compatibility_T_xy"] = T_x.diff(y) - T_y.diff(x)
            self.equations["compatibility_T_xz"] = T_x.diff(z) - T_z.diff(x)
            self.equations["compatibility_T_yz"] = T_y.diff(z) - T_z.diff(y)
            if self.dim == 2:
                self.equations.pop("compatibility_T_z")
                self.equations.pop("compatibility_T_xz")
                self.equations.pop("compatibility_T_yz")


class Vector_Potential(PDE):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    

    def __init__(self, instr_List,outstr_List,case_coord_strList=None):
        # set params
        
        if case_coord_strList is None:
            case_coord_strList=["x_case","y_case","z_case"]
        x = Symbol(case_coord_strList[0])
        y = Symbol(case_coord_strList[1])
        z = Symbol(case_coord_strList[2])
        input_variables = {case_coord_strList[0]: x, case_coord_strList[1]: y, case_coord_strList[2]: z}
        
        vec_potential=[]
        for n_str in instr_List:
            vec_potential.append(Function(n_str)(*input_variables))
        self.equations = {}
        self.equations[outstr_List[0]] = vec_potential[2].diff(y)-vec_potential[1].diff(z)
        self.equations[outstr_List[1]] = vec_potential[0].diff(z)-vec_potential[2].diff(x)
        self.equations[outstr_List[2]] = vec_potential[1].diff(x)-vec_potential[0].diff(y)
        

              
class FungModel_ZeroDerivative(PDE):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    

    def __init__(self, F,p_str,ID_string='', Q=None,case_coord_strList=None):
        # set params
        
        self.name = ID_string
        if case_coord_strList is None:
            case_coord_strList=["x_case","y_case","z_case"]
        x = Symbol(case_coord_strList[0])
        y = Symbol(case_coord_strList[1])
        z = Symbol(case_coord_strList[2])
        input_variables = {case_coord_strList[0]: x, case_coord_strList[1]: y, case_coord_strList[2]: z}
        p = Function(p_str)(*input_variables)
        if Q is None:
            Finv=Matrix(F).inv(method="LU")
        else:
            Finv=Matrix(F).inv(method="LU")*Matrix(Q).T
        pdu=Finv*Matrix([[p.diff(x)],[p.diff(y)],[p.diff(z)]])
        self.equations = {}
        self.equations[ID_string+"pdu"] = pdu[0,0]
        self.equations[ID_string+"pdv"] = pdu[1,0]
        self.equations[ID_string+"pdw"] = pdu[2,0]
class pDerivative(PDE):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    

    def __init__(self,p_str,ID_string='',case_coord_strList=None):
        # set params
        
        self.name = ID_string
        if case_coord_strList is None:
            case_coord_strList=["x_case","y_case","z_case"]
        x = Symbol(case_coord_strList[0])
        y = Symbol(case_coord_strList[1])
        z = Symbol(case_coord_strList[2])
        input_variables = {case_coord_strList[0]: x, case_coord_strList[1]: y, case_coord_strList[2]: z}
        p = Function(p_str)(*input_variables)
        self.equations = {}
        self.equations[ID_string+p_str+"dx"] = p.diff(x)
        self.equations[ID_string+p_str+"dy"] = p.diff(y)
        self.equations[ID_string+p_str+"dz"] = p.diff(z)
class FungModel_ZeroDerivative_withpDerivative(torch.nn.Module):
    '''
    inputs:
        x:
            Key("pdx"):
            Key("pdy"):
            Key("pdz"):
        o:
            Key("Rux"):
            Key("Ruy"):
            Key("Ruz"):
            Key("Rvx"):
            Key("Rvy"):
            Key("Rvz"):
            Key("Rwx"):
            Key("Rwy"):
            Key("Rwz"):
    return:
        Key("pdu"):
        Key("pdv"):
        Key("pdw"):
    '''
    def __init__(self,minuseye=False):
        super().__init__()
        if minuseye:
            self.eye=torch.eye(3,dtype=torch.float)
        else:
            self.eye=torch.zeros((3,3),dtype=torch.float)
    def forward(self,x,o):#o is the unloaded deformation gradient tensor
        S=torch.stack([x[...,0:1],x[...,1:2],x[...,2:3]],dim=x.dim()-1)
        F=torch.stack([o[...,0:3],o[...,3:6],o[...,6:9]],dim=x.dim())-self.eye
        P=torch.linalg.solve(F, S, left=True)
        #F=torch.matmul(F,torch.linalg.inv(G))
        return torch.flatten(P, start_dim=x.dim()-1)
class Fmat(PDE):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    
    
    def __init__(self,case_coord_strList=None,matrix_string='F',returnQR=False):
        # set params
        self.name = matrix_string+"mat"
        velstr=['u','v','w']
        coordstr=['x','y','z']
        if case_coord_strList is None:
            case_coord_strList=["x","y","z"]
        x = Symbol(case_coord_strList[0])
        y = Symbol(case_coord_strList[1])
        z = Symbol(case_coord_strList[2])
        input_variables = {case_coord_strList[0]: x, case_coord_strList[1]: y, case_coord_strList[2]: z}
        u = Function("u")(*input_variables)
        v = Function("v")(*input_variables)
        w = Function("w")(*input_variables)
        
        # velocity componets  0.5*(F.T*F-I)
        F=np.array([[u.diff(x)+Number(1.),u.diff(y),u.diff(z)],
                [v.diff(x),v.diff(y)+Number(1.),v.diff(z)],
                [w.diff(x),w.diff(y),w.diff(z)+Number(1.)]])
        self.equations = {}
        if returnQR:
            Q, R=Matrix(F).QRdecomposition()
            self.equations[matrix_string+"Q_"+velstr[0]+coordstr[0]] = Q[0,0]
            self.equations[matrix_string+"Q_"+velstr[0]+coordstr[1]] = Q[0,1]
            self.equations[matrix_string+"Q_"+velstr[0]+coordstr[2]] = Q[0,2]
            self.equations[matrix_string+"Q_"+velstr[1]+coordstr[0]] = Q[1,0]
            self.equations[matrix_string+"Q_"+velstr[1]+coordstr[1]] = Q[1,1]
            self.equations[matrix_string+"Q_"+velstr[1]+coordstr[2]] = Q[1,2]
            self.equations[matrix_string+"Q_"+velstr[2]+coordstr[0]] = Q[2,0]
            self.equations[matrix_string+"Q_"+velstr[2]+coordstr[1]] = Q[2,1]
            self.equations[matrix_string+"Q_"+velstr[2]+coordstr[2]] = Q[2,2]
            self.equations[matrix_string+"R_"+velstr[0]+coordstr[0]] = R[0,0]
            self.equations[matrix_string+"R_"+velstr[0]+coordstr[1]] = R[0,1]
            self.equations[matrix_string+"R_"+velstr[0]+coordstr[2]] = R[0,2]
            self.equations[matrix_string+"R_"+velstr[1]+coordstr[0]] = R[1,0]
            self.equations[matrix_string+"R_"+velstr[1]+coordstr[1]] = R[1,1]
            self.equations[matrix_string+"R_"+velstr[1]+coordstr[2]] = R[1,2]
            self.equations[matrix_string+"R_"+velstr[2]+coordstr[0]] = R[2,0]
            self.equations[matrix_string+"R_"+velstr[2]+coordstr[1]] = R[2,1]
            self.equations[matrix_string+"R_"+velstr[2]+coordstr[2]] = R[2,2]
            self.equations[matrix_string+"_det"] = R[0,0]*R[1,1]*R[2,2]
        else:
            self.equations[matrix_string+"_"+velstr[0]+coordstr[0]] = u.diff(x)+Number(1.)
            self.equations[matrix_string+"_"+velstr[0]+coordstr[1]] = u.diff(y)
            self.equations[matrix_string+"_"+velstr[0]+coordstr[2]] = u.diff(z)
            self.equations[matrix_string+"_"+velstr[1]+coordstr[0]] = v.diff(x)
            self.equations[matrix_string+"_"+velstr[1]+coordstr[1]] = v.diff(y)+Number(1.)
            self.equations[matrix_string+"_"+velstr[1]+coordstr[2]] = v.diff(z)
            self.equations[matrix_string+"_"+velstr[2]+coordstr[0]] = w.diff(x)
            self.equations[matrix_string+"_"+velstr[2]+coordstr[1]] = w.diff(y)
            self.equations[matrix_string+"_"+velstr[2]+coordstr[2]] = w.diff(z)+Number(1.)
            #self.equations[matrix_string+"_det"] = F[0,0]*F[1,1]*F[2,2]
class FungModel(torch.nn.Module):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    inputs:
        Key("Cux"):
        Key("Cuy"):
        Key("Cuz"):
        Key("Cvx"):
        Key("Cvy"):
        Key("Cvz"):
        Key("Cwx"):
        Key("Cwy"):
        Key("Cwz"):
    return:
        Key("Sux"):
        Key("Suy"):
        Key("Suz"):
        Key("Svx"):
        Key("Svy"):
        Key("Svz"):
        Key("Swx"):
        Key("Swy"):
        Key("Swz"):
    """

    

    def __init__(self, Cstrain,Cff,Css,Cnn,Cfs,Cfn,Csn):
        super().__init__()
        Cstrain=torch.tensor(Cstrain/2.)
        self.register_buffer("Cstrain", Cstrain, persistent=False)
        Cfsn=torch.tensor([Cff,Cfs,Cfn,Cfs,Css,Csn,Cfn,Csn,Cnn],dtype=torch.float)
        self.register_buffer("Cfsn", Cfsn, persistent=False)
        #self.p=torch.nn.Parameter(torch.empty((1,1)))
        #self.reset_parameters()
    def forward(self,x,o):#3x3 strain,determinant
        QQ =torch.sum(self.Cfsn*x[...,0:9]**2.0,dim=-1,keepdim=True)
        W=self.Cstrain*(torch.exp(QQ) -  1.0)
        return W-o[...,0:1]*(1.-o[...,1:2])
    def reset_parameters(self,set_to=0.1) -> None:
        torch.nn.init.constant_(self.p, set_to)
class FungModel_incompressible(torch.nn.Module):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    inputs:
        Key("Cux"):
        Key("Cuy"):
        Key("Cuz"):
        Key("Cvx"):
        Key("Cvy"):
        Key("Cvz"):
        Key("Cwx"):
        Key("Cwy"):
        Key("Cwz"):
    return:
        Key("Sux"):
        Key("Suy"):
        Key("Suz"):
        Key("Svx"):
        Key("Svy"):
        Key("Svz"):
        Key("Swx"):
        Key("Swy"):
        Key("Swz"):
    """

    

    def __init__(self, Cstrain,Cff,Css,Cnn,Cfs,Cfn,Csn):
        super().__init__()
        Cstrain=torch.tensor(Cstrain/2.)
        self.register_buffer("Cstrain", Cstrain, persistent=False)
        Cfsn=torch.tensor([Cff,Cfs,Cfn,Cfs,Css,Csn,Cfn,Csn,Cnn],dtype=torch.float)
        self.register_buffer("Cfsn", Cfsn, persistent=False)
        #self.p=torch.nn.Parameter(torch.empty((1,1)))
        #self.reset_parameters()
    def forward(self,x):#3x3 strain,determinant
        QQ =torch.sum(self.Cfsn*x[...,0:9]**2.0,dim=-1,keepdim=True)
        W=self.Cstrain*(torch.exp(QQ) -  1.0)
        return W
    def reset_parameters(self,set_to=0.1) -> None:
        torch.nn.init.constant_(self.p, set_to)
class FungModel_fsn_endo_surfaceStress_FTF(torch.nn.Module):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    inputs:
        Key("W"):
        Key("Cux"):
        Key("Cuy"):
        Key("Cuz"):
        Key("Cvx"):
        Key("Cvy"):
        Key("Cvz"):
        Key("Cwx"):
        Key("Cwy"):
        Key("Cwz"):
    o:
        Key("detJ"):
        Key("Fux"):
        Key("Fuy"):
        Key("Fuz"):
        Key("Fvx"):
        Key("Fvy"):
        Key("Fvz"):
        Key("Fwx"):
        Key("Fwy"):
        Key("Fwz"):
    return:
        Key("Surface stress"):
    """

    

    def __init__(self,Cff,Css,Cnn,Cfs,Cfn,Csn,transpose_F=False):
        super().__init__()
        Cfsn=torch.tensor([Cff,Cfs,Cfn,Cfs,Css,Csn,Cfn,Csn,Cnn],dtype=torch.float)
        self.register_buffer("Cfsn", Cfsn, persistent=False)
        self.transpose_F=transpose_F
    def forward(self,x,o):#[W,strain_fsn_nn,]
        dWdEf_flat= self.Cfsn*x[...,0:1]*x[...,1:10]*2.
        dWdEf=torch.stack([dWdEf_flat[...,0:3],dWdEf_flat[...,3:6],dWdEf_flat[...,6:9]],dim=x.dim()-1)
        F=torch.stack([o[...,1:4],o[...,4:7],o[...,7:10]],dim=x.dim()-1)
        FT=torch.transpose(F,-1,-2)
        if self.transpose_F:
            stress=torch.matmul(FT,torch.matmul(dWdEf,F))/torch.unsqueeze(o[...,0:1], dim=x.dim()-1)
        else:
            stress=torch.matmul(F,torch.matmul(dWdEf,FT))/torch.unsqueeze(o[...,0:1], dim=x.dim()-1)
        
        return torch.flatten(stress, start_dim=x.dim()-1)
class Inverse(torch.nn.Module):
    '''
    inputs:
        Key("Fux"):
        Key("Fuy"):
        Key("Fuz"):
        Key("Fvx"):
        Key("Fvy"):
        Key("Fvz"):
        Key("Fwx"):
        Key("Fwy"):
        Key("Fwz"):
    return:
        Key("Sux"):
        Key("Suy"):
        Key("Suz"):
        Key("Svx"):
        Key("Svy"):
        Key("Svz"):
        Key("Swx"):
        Key("Swy"):
        Key("Swz"):
    '''
    def __init__(self,inverse=False):
        super().__init__()
        self.inverse=inverse
        F=torch.eye(3,dtype=torch.float)
        self.register_buffer("F", F, persistent=False)
    def forward(self,x):
        G=torch.stack([x[...,0:3],x[...,3:6],x[...,6:9]],dim=x.dim()-1)
        F=torch.linalg.solve_triangular(G, self.F, upper=True, left=False)
        return torch.flatten(F, start_dim=x.dim()-1)
class MatrixRotate(torch.nn.Module):
    '''
    inputs:
        Key("Sux"):
        Key("Suy"):
        Key("Suz"):
        Key("Svx"):
        Key("Svy"):
        Key("Svz"):
        Key("Swx"):
        Key("Swy"):
        Key("Swz"):
    return:
        Key("Qux"):
        Key("Quy"):
        Key("Quz"):
        Key("Qvx"):
        Key("Qvy"):
        Key("Qvz"):
        Key("Qwx"):
        Key("Qwy"):
        Key("Qwz"):
    '''
    def __init__(self,do_transpose=False):
        super().__init__()
        self.do_transpose=do_transpose
    def forward(self,x,o):
        S=torch.stack([x[...,0:3],x[...,3:6],x[...,6:9]],dim=x.dim()-1)
        Q=torch.stack([o[...,0:3],o[...,3:6],o[...,6:9]],dim=x.dim()-1)
        QT=torch.transpose(Q,-1,-2)
        if self.do_transpose:
            QSQ=torch.matmul(Q,torch.matmul(S,QT))
        else:
            QSQ=torch.matmul(QT,torch.matmul(S,Q))
        return torch.flatten(QSQ, start_dim=x.dim()-1)
class Strain_E(torch.nn.Module):
    '''
    inputs:
        Key("Fux"):
        Key("Fuy"):
        Key("Fuz"):
        Key("Fvx"):
        Key("Fvy"):
        Key("Fvz"):
        Key("Fwx"):
        Key("Fwy"):
        Key("Fwz"):
    return:
        Key("Sux"):
        Key("Suy"):
        Key("Suz"):
        Key("Svx"):
        Key("Svy"):
        Key("Svz"):
        Key("Swx"):
        Key("Swy"):
        Key("Swz"):
    '''
    def __init__(self,inverse=False):
        super().__init__()
        self.inverse=inverse
        F=torch.eye(3,dtype=torch.float)
        self.register_buffer("F", F, persistent=False)
    def forward(self,x):
        if self.inverse:
            G=torch.stack([x[...,0:3],x[...,3:6],x[...,6:9]],dim=x.dim()-1)
            F=torch.linalg.solve_triangular(G, self.F, upper=True, left=False)
        else:
            F=torch.stack([x[...,0:3],x[...,3:6],x[...,6:9]],dim=x.dim()-1)
        FTF=torch.matmul(torch.transpose(F,-1,-2),F)
        return torch.flatten(FTF, start_dim=x.dim()-1)
class Strain_E_withunload(torch.nn.Module):
    '''
    inputs:
        x:
            Key("Fux"):
            Key("Fuy"):
            Key("Fuz"):
            Key("Fvx"):
            Key("Fvy"):
            Key("Fvz"):
            Key("Fwx"):
            Key("Fwy"):
            Key("Fwz"):
        o:
            Key("Gux"):
            Key("Guy"):
            Key("Guz"):
            Key("Gvx"):
            Key("Gvy"):
            Key("Gvz"):
            Key("Gwx"):
            Key("Gwy"):
            Key("Gwz"):
    return:
        Key("Sux"):
        Key("Suy"):
        Key("Suz"):
        Key("Svx"):
        Key("Svy"):
        Key("Svz"):
        Key("Swx"):
        Key("Swy"):
        Key("Swz"):
    '''
    def __init__(self):
        super().__init__()
    def forward(self,x,o):#o is the unloaded deformation gradient tensor
        F=torch.stack([x[...,0:3],x[...,3:6],x[...,6:9]],dim=x.dim()-1)
        G=torch.stack([o[...,0:3],o[...,3:6],o[...,6:9]],dim=x.dim()-1)
        F=torch.linalg.solve_triangular(G, F, upper=True, left=False)
        #F=torch.matmul(F,torch.linalg.inv(G))
        FTF=torch.matmul(torch.transpose(F,-1,-2),F)
        return torch.flatten(FTF, start_dim=x.dim()-1)
class to_fsn_Rotation_Matrix(torch.nn.Module):
    '''
    inputs:
        Key("radial_01"):
        Key("lvec_x"):
        Key("lvec_y"):
        Key("lvec_z"):
        Key("cvec_x"):
        Key("cvec_y"):
        Key("cvec_z"):
        Key("rvec_x"):
        Key("rvec_y"):
        Key("rvec_z"):
    return:
        flatten rotation matrix Q where Q[xyz]=[fsn]
    '''
    def __init__(self,pm_deg=60.):
        super().__init__()
        rad=torch.tensor(pm_deg/180*np.pi)
        self.register_buffer("rad", rad, persistent=False)
    def forward(self,x):
        theta=2.*(x[...,0:1]-0.5)*self.rad
        costheta=torch.cos(theta)
        sintheta=torch.sin(theta)
        zero=torch.zeros_like(x[...,0:1])
        one=torch.ones_like(x[...,0:1])
        rotate=torch.stack([torch.cat((sintheta,costheta,zero),dim=-1),torch.cat((-costheta,sintheta,zero),dim=-1),torch.cat((zero,zero,one),dim=-1)],dim=x.dim()-1)
        lcr=torch.stack((x[...,1:4],x[...,4:7],x[...,7:10]), dim=x.dim()-1)
        Q=torch.matmul(rotate,lcr)#Q where Q[xyz]=[fsn]
        return torch.flatten(Q, start_dim=x.dim()-1)

class strain_to_fsn_Rotation_Matrix(torch.nn.Module):
    '''
    inputs:
        x:
            Key("Sux"):
            Key("Suy"):
            Key("Suz"):
            Key("Svx"):
            Key("Svy"):
            Key("Svz"):
            Key("Swx"):
            Key("Swy"):
            Key("Swz"):
        o:
            Key("radial_01"):
            Key("lvec_x"):
            Key("lvec_y"):
            Key("lvec_z"):
            Key("cvec_x"):
            Key("cvec_y"):
            Key("cvec_z"):
            Key("rvec_x"):
            Key("rvec_y"):
            Key("rvec_z"):
    return:
        flatten rotated Strain matrix
    '''
    def __init__(self,pm_deg=60,limit=None):
        super().__init__()
        rad=torch.tensor(pm_deg/180*np.pi)
        self.register_buffer("rad", rad, persistent=False)
        if limit is None:
            self.limit=None
        else:
            limit=torch.tensor(limit)
            self.register_buffer("limit", limit, persistent=False)
        eye=torch.eye(3,dtype=torch.float)
        self.register_buffer("eye", eye, persistent=False)
    def forward(self,x,o):
        S=torch.stack((x[...,0:3],x[...,3:6],x[...,6:9]), dim=x.dim()-1)
        theta=2.*(o[...,0:1]-0.5)*self.rad
        costheta=torch.cos(theta)
        sintheta=torch.sin(theta)
        zero=torch.zeros_like(o[...,0:1])
        one=torch.ones_like(o[...,0:1])
        rotate=torch.stack([torch.cat((sintheta,costheta,zero),dim=-1),torch.cat((-costheta,sintheta,zero),dim=-1),torch.cat((zero,zero,one),dim=-1)],dim=x.dim()-1)
        lcr=torch.stack((o[...,1:4],o[...,4:7],o[...,7:10]), dim=x.dim()-1)
        Q=torch.matmul(rotate,lcr)#Q where Q[xyz]=[fsn]
        S=torch.matmul(torch.matmul(Q,S),torch.transpose(Q,-1,-2))
        if self.limit is None:
            S=0.5*(S-self.eye)
        else:
            S=self.limit*torch.tanh(0.5*(S-self.eye)/self.limit)
        return torch.flatten(S, start_dim=x.dim()-1)