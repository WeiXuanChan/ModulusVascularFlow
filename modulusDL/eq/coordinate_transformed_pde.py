"""Equations related to Navier Stokes Equations with coordinate transformation
"""

from sympy import Symbol, Function, Number

from modulus.eq.pde import PDE
from modulus.node import Node

from typing import Optional, Callable, List, Dict, Union, Tuple
from modulus.key import Key

class NavierStokes_CoordTransformed(PDE):
    """
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

    def __init__(self, nu,case_param_symbolDict=None, rho=1, dim=3, time=True, mixed_form=False):
        # set params
        self.dim = dim
        self.time = time
        self.mixed_form = mixed_form
        if case_param_symbolDict is None:
            case_param_symbolDict={}
        # coordinates
        #x_ref, y_ref, z_ref = Symbol("x"), Symbol("y"), Symbol("z")

        # time
        t= Symbol("t")

        # make input variables
        #input_variables_ref = {"x": x_ref, "y": y_ref, "z": z_ref, "t": t}
        #if self.dim == 2:
        #    input_variables_ref.pop("z")
        #if not self.time:
        #    input_variables_ref.pop("t")
        #for case_param_symbol in case_param_symbolDict:
        #    input_variables_ref[case_param_symbol]=case_param_symbolDict[case_param_symbol]
        # case coordinates
        #x = Function("x_case")(*input_variables_ref)
        #y = Function("y_case")(*input_variables_ref)
        #z = Function("z_case")(*input_variables_ref)
        x = Symbol("x_case")
        y = Symbol("y_case")
        z = Symbol("z_case")
        input_variables = {"x_case": x, "y_case": y, "z_case": z, "t": t}
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
class NormalDotVec_CoordTransformed(PDE):
    """
    Normal dot velocity !!! in progress

    Parameters
    ==========
    dim : int
        Dimension of the equations (1, 2, or 3). Default is 3.
    """

    name = "NormalDotVec"

    def __init__(self, vec=["u", "v", "w"]):
        # normal
        normal = [Symbol("normal_x"), Symbol("normal_y"), Symbol("normal_z")]

        # make input variables
        self.equations = {}
        self.equations["normal_dot_vel"] = 0
        for v, n in zip(vec, normal):
            self.equations["normal_dot_vel"] += Symbol(v) * n
