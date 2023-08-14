"""Equations related to Navier Stokes Equations with coordinate transformation
"""

from sympy import Symbol, Function, Number
from sympy import exp , Matrix
import numpy as np

from modulus.eq.pde import PDE
from modulus.node import Node

from typing import Optional, Callable, List, Dict, Union, Tuple
from modulus.key import Key

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

    def __init__(self, nu,case_coord_strList=None,case_param_strList=None, rho=1, dim=3, time=True, mixed_form=False):
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


class Fiber_Strain(PDE):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    

    def __init__(self, f0_s0_n0,Fmat_toFinal=None,Gmat_to_ref=None,ID_string=''):
        # f0_s0_n0 is constant (non solvable)
        self.name = ID_string+"Fiber_Strain"
        fxyz=np.array([[f0_s0_n0[0][0]],[f0_s0_n0[0][1]],[f0_s0_n0[0][2]]])
        sxyz=np.array([[f0_s0_n0[1][0]],[f0_s0_n0[1][1]],[f0_s0_n0[1][2]]])
        nxyz=np.array([[f0_s0_n0[2][0]],[f0_s0_n0[2][1]],[f0_s0_n0[2][2]]])
        
        I=np.array([[Number(1),Number(0),Number(0)],
                    [Number(0),Number(1),Number(0)],
                    [Number(0),Number(0),Number(1)]])
        if Fmat_toFinal is None:
            F=I
        else:
            F=np.array(Fmat_toFinal)
        if Gmat_to_ref is None:
            G=I
        else:
            G=np.array(Gmat_to_ref)
        S=0.5*(F.T.dot(F)-G.T.dot(G))
        stretch_f=(np.sum(G.dot(fxyz)**2.))**0.5
        stretch_s=(np.sum(G.dot(sxyz)**2.))**0.5
        stretch_n=(np.sum(G.dot(nxyz)**2.))**0.5
        self.equations={}
        self.equations[ID_string+"strain_ff"] = np.sum(fxyz*fxyz.reshape((1,-1))*S)/stretch_f/stretch_f
        self.equations[ID_string+"strain_fs"] = np.sum(fxyz*sxyz.reshape((1,-1))*S)/stretch_f/stretch_s
        self.equations[ID_string+"strain_fn"] = np.sum(fxyz*nxyz.reshape((1,-1))*S)/stretch_f/stretch_n
        self.equations[ID_string+"strain_sf"] = np.sum(fxyz*sxyz.reshape((1,-1))*S)/stretch_f/stretch_s
        self.equations[ID_string+"strain_ss"] = np.sum(sxyz*sxyz.reshape((1,-1))*S)/stretch_s/stretch_s
        self.equations[ID_string+"strain_sn"] = np.sum(sxyz*nxyz.reshape((1,-1))*S)/stretch_s/stretch_n
        self.equations[ID_string+"strain_nf"] = np.sum(fxyz*nxyz.reshape((1,-1))*S)/stretch_f/stretch_n
        self.equations[ID_string+"strain_ns"] = np.sum(sxyz*nxyz.reshape((1,-1))*S)/stretch_s/stretch_n
        self.equations[ID_string+"strain_nn"] = np.sum(nxyz*nxyz.reshape((1,-1))*S)/stretch_n/stretch_n
        self.equations[ID_string+"Jacobian"] = Matrix(F).det()/Matrix(G).det()
        
class FungModel(PDE):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    

    def __init__(self, Cstrain,Cff,Css,Cnn,Cfs,Cfn,Csn,p,strain_fsnMatrix,Jacobian,ID_string=''):
        # set params
        self.name = ID_string+"FungModel"
        QQ = Cff*strain_fsnMatrix[0][0]**2.0 +\
            Css*strain_fsnMatrix[1][1]**2.0 +\
                Cnn*strain_fsnMatrix[2][2]**2.0 +\
                    Csn*strain_fsnMatrix[1][2]**2.0 +\
                    Csn*strain_fsnMatrix[2][1]**2.0 +\
                        Cfs*strain_fsnMatrix[0][1]**2.0 +\
                        Cfs*strain_fsnMatrix[1][0]**2.0 +\
                            Cfs*strain_fsnMatrix[0][2]**2.0 +\
                            Cfn*strain_fsnMatrix[2][0]**2.0
        self.equations = {}
        self.equations[ID_string+"strain_energy_density_constant"] = Cstrain/2.*(exp(QQ) -  1.0) - p
        self.equations[ID_string+"incompressibility"]=p*(Jacobian-Number(1))
        self.equations[ID_string+"strain_energy_density_eqn"] = Cstrain/2.*(exp(QQ) -  1.0) - p*(Jacobian-Number(1))
        self.equations[ID_string+"strain_energy_density"] = Cstrain/2.*(exp(QQ) -  1.0)
class SurfaceNormal_Stress(PDE):
    name = "SurfaceNormal_Stress"
    def __init__(self, f0_s0_n0,strain_energy_densityStr,Strain_fsn_matrix,normal_SymList=None,Fmat_toFinal=None,Gmat_to_ref=None):
        # f0_s0_n0 is constant (non solvable)
        f0_s0_n0=np.array(f0_s0_n0)
        if normal_SymList is None:
            normal_SymList = [Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")]
        normal_SymList = np.array(normal_SymList)
        I=np.array([[Number(1),Number(0),Number(0)],
                    [Number(0),Number(1),Number(0)],
                    [Number(0),Number(0),Number(1)]])
        if Fmat_toFinal is None:
            F=I
        else:
            F=np.array(Fmat_toFinal)
        if Gmat_to_ref is None:
            G=I
        else:
            G=np.array(Gmat_to_ref)
        fdotnorm=F.dot(normal_SymList.reshape((-1,1)))
        fsn=G.dot(f0_s0_n0.T).T
        fsn_normal=(fsn/(np.sum(fsn**2.,axis=-1,keepdims=True))**0.5).dot(fdotnorm)/(np.sum(fdotnorm**2.))**0.5
        inputVar={}
        for n in range(3):
            for m in range(3):
                inputVar[Strain_fsn_matrix[n][m].name]=Strain_fsn_matrix[n][m]
        strain_energy_density=Function(strain_energy_densityStr)(*inputVar)
        sigma=[]
        for n in range(3):
            sigma.append([])
            for m in range(3):
                sigma[-1].append(strain_energy_density.diff(Strain_fsn_matrix[n][m]))
        sigma_np=np.array(sigma)
        self.equations = {}
        self.equations["surface_normal_stress"] = np.sum(fsn_normal.reshape((1,-1))*fsn_normal*sigma_np)
class Fmat(PDE):
    """
    FungModel

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    
    
    def __init__(self,case_coord_strList=None,invert=False,matrix_string='F'):
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
        F_temp=np.array([[u.diff(x)+Number(1.),u.diff(y),u.diff(z)],
                [v.diff(x),v.diff(y)+Number(1.),v.diff(z)],
                [w.diff(x),w.diff(y),w.diff(z)+Number(1.)]])
        if invert is True:
            F=Matrix(F_temp).inv(method="LU")
        elif invert is False:
            F=F_temp
        else:
            F=F_temp.dot(np.array(invert))
        self.equations = {}
        self.equations[matrix_string+"_"+velstr[0]+coordstr[0]] = F[0,0]
        self.equations[matrix_string+"_"+velstr[0]+coordstr[1]] = F[0,1]
        self.equations[matrix_string+"_"+velstr[0]+coordstr[2]] = F[0,2]
        self.equations[matrix_string+"_"+velstr[1]+coordstr[0]] = F[1,0]
        self.equations[matrix_string+"_"+velstr[1]+coordstr[1]] = F[1,1]
        self.equations[matrix_string+"_"+velstr[1]+coordstr[2]] = F[1,2]
        self.equations[matrix_string+"_"+velstr[2]+coordstr[0]] = F[2,0]
        self.equations[matrix_string+"_"+velstr[2]+coordstr[1]] = F[2,1]
        self.equations[matrix_string+"_"+velstr[2]+coordstr[2]] = F[2,2]

        
