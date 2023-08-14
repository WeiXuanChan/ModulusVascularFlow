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

import numpy as np
from sympy import (
    Symbol,
    Abs,
    Min,
    sqrt,
    pi,
    sin,
    cos,
    exp,
)

from modulus.geometry import Bounds
from modulus.geometry.geometry import Geometry, csg_curve_naming
from modulus.geometry.curve import SympyCurve
from modulus.geometry.helper import _sympy_sdf_to_sdf
from modulus.geometry.parameterization import Parameterization, Parameter

class HLine(Geometry):
    """
    This class is adapted from NVIDIAModulus v22.09 geometry.Vline
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
class FlexibleTube(Geometry):
    """
    3D Cylinder
    centerline parallel to z-axis

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

    def __init__(self, center, radius, length, curve_control=None,parameterization=Parameterization(),geom='tube'):#geom=["wall","inlet","outlet"]
        # make sympy symbols to use
        l, r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1))
        theta = Symbol(csg_curve_naming(2))
        r0,rr,A,alpha = Symbol(csg_curve_naming(3)), Symbol(csg_curve_naming(4)), Symbol(csg_curve_naming(5)), Symbol(csg_curve_naming(6))
        s1,s2,s3,s4 = Symbol(csg_curve_naming(7)), Symbol(csg_curve_naming(8)), Symbol(csg_curve_naming(9)), Symbol(csg_curve_naming(10))
        curve_control_symbol={"r0":r0, 
                              "rr":rr,
                              "A":A,
                              "alpha":alpha,
                              "s1":s1,
                              "s2":s2,
                              "s3":s3,
                              "s4":s4}
        default_curve_control={"r0":(-0.3,0.3), 
                              "rr":(-0.3,0.3),
                              "A":(-0.6,0.6),
                              "alpha":(0.05,0.15),
                              "s1":(-6.,6.),
                              "s2":(-3.,3.),
                              "s3":(-6.,6.),
                              "s4":(-3.,3.)}
        # surface of the cylinder
        if geom=="wall":
            curve_parameterization = Parameterization(
                {l: (-1, 1), theta: (0, 2 * pi)}
            )
        elif geom in ["inlet","outlet"]:
            curve_parameterization = Parameterization(
                { r: (0, 1), theta: (0, 2 * pi)}
            )
        else:
            curve_parameterization = Parameterization(
                {l: (-1, 1), r: (0, 1), theta: (0, 2 * pi)}
            )
        if curve_control is not None:
            for key in curve_control.keys():
                default_curve_control[key]=curve_control[key]
        for key in curve_control_symbol.keys():
            curve_parameterization[curve_control_symbol[key]]=default_curve_control[key]
        
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        x_ref= 0.5 * l * length
        cline_y=radius*(s1*sin(pi*x_ref/length)+s2*sin(2.*pi*x_ref/length))
        cline_z=radius*(s3*sin(pi*x_ref/length)+s4*sin(2.*pi*x_ref/length))
        new_radius=radius*(1.+r0+rr*(x_ref/length)*2.)*(1.-A*exp(-1.*(x_ref/length/alpha)**2./2.))
        cline_dydx=radius*pi/length*(s1*cos(pi*x_ref/length)+2.*s2*cos(2.*pi*x_ref/length))
        cline_dzdx=radius*pi/length*(s3*cos(pi*x_ref/length)+2.*s4*cos(2.*pi*x_ref/length))
        norm_y=1./sqrt(cline_dydx**2.+1.)
        norm_xy=-norm_y*cline_dydx
        norm_z=1./sqrt(cline_dzdx**2.+1.)
        norm_xz=-norm_z*cline_dzdx
        if not(geom in ["inlet","outlet"]):
            curve_1 = SympyCurve(
                functions={
                    "x": center[0] + x_ref + new_radius * (norm_xy * sin(theta) + norm_xz * cos(theta)),
                    "y": center[1] + cline_y + new_radius * norm_y * sin(theta),
                    "z": center[2] + cline_z + new_radius * norm_z * cos(theta),
                    "normal_x": norm_xy * sin(theta) + norm_xz * cos(theta),
                    "normal_y": norm_y * sin(theta),
                    "normal_z": norm_xz * cos(theta),
                },
                parameterization=curve_parameterization,
                area=length * 2 * pi * radius * (1.+r0),#!!!! did not consider the stenosis
            )
        outlet_cline_y=radius*s1
        outlet_cline_z=radius*s3
        end_reduction=(1.-A*exp(-1.*(0.5/alpha)**2./2.))
        outlet_radius=radius*(1.+r0+rr)*end_reduction
        outlet_cline_dydx=radius*pi/length*-2.*s2
        outlet_cline_dzdx=radius*pi/length*-2.*s4
        outlet_norm_y=1./sqrt(outlet_cline_dydx**2.+1.)
        outlet_norm_xy=-outlet_norm_y*outlet_cline_dydx
        outlet_norm_z=1./sqrt(outlet_cline_dzdx**2.+1.)
        outlet_norm_xz=-outlet_norm_z*outlet_cline_dzdx
        outlet_vec_x=1/sqrt(outlet_cline_dydx**2.+outlet_cline_dzdx**2.+1)
        outlet_vec_y=outlet_cline_dydx*outlet_vec_x
        outlet_vec_z=outlet_cline_dzdx*outlet_vec_x
        if not(geom in ["inlet","wall"]):
            curve_2 = SympyCurve(
                functions={
                    "x": center[0] + 0.5 * length + sqrt(r) * outlet_radius * (outlet_norm_xy * sin(theta) + outlet_norm_xz * cos(theta)),
                    "y": center[1] + outlet_cline_y + sqrt(r) * outlet_radius * outlet_norm_y * sin(theta),
                    "z": center[2] + outlet_cline_z + sqrt(r) * outlet_radius * outlet_norm_z * cos(theta),
                    "normal_x": -outlet_vec_x,
                    "normal_y": -outlet_vec_y,
                    "normal_z": -outlet_vec_z,
                    "norm dissq_from_wall": 1-r,
                },
                parameterization=curve_parameterization,
                area=pi * outlet_radius**2,
            )
        intlet_cline_y=-outlet_cline_y
        intlet_cline_z=-outlet_cline_z
        intlet_radius=radius*(1.+r0-rr)*end_reduction
        intlet_cline_dydx=outlet_cline_dydx
        intlet_cline_dzdx=outlet_cline_dzdx
        intlet_norm_y=1./sqrt(intlet_cline_dydx**2.+1.)
        intlet_norm_xy=-intlet_norm_y*intlet_cline_dydx
        intlet_norm_z=1./sqrt(intlet_cline_dzdx**2.+1.)
        intlet_norm_xz=-intlet_norm_z*intlet_cline_dzdx
        intlet_vec_x=1/sqrt(intlet_cline_dydx**2.+intlet_cline_dzdx**2.+1)
        intlet_vec_y=intlet_cline_dydx*intlet_vec_x
        intlet_vec_z=intlet_cline_dzdx*intlet_vec_x
        if not(geom in ["wall","outlet"]):
            curve_3 = SympyCurve(
                functions={
                    "x": center[0] - 0.5 * length + sqrt(r) * intlet_radius * (intlet_norm_xy * sin(theta) + intlet_norm_xz * cos(theta)),
                    "y": center[1] + intlet_cline_y + sqrt(r) * intlet_radius * intlet_norm_y * sin(theta),
                    "z": center[2] + intlet_cline_z + sqrt(r) * intlet_radius * intlet_norm_z * cos(theta),
                    "normal_x": intlet_vec_x,
                    "normal_y": intlet_vec_y,
                    "normal_z": intlet_vec_z,
                    "norm dissq_from_wall": 1-r,
                },
                parameterization=curve_parameterization,
                area=pi * intlet_radius**2,
            )
        if geom=="wall":
            curves = [curve_1]
        elif geom=="inlet":
            curves = [curve_3]
        elif geom=="outlet":
            curves = [curve_2]
        else:
            curves = [curve_1, curve_2, curve_3]

        # calculate SDF
        if geom in ["inlet","outlet"]:
            outside_distance = Min(0, 1-sqrt(r))
            inside_distance = -1 * Abs(Min(0, sqrt(r) - 1))
            sdf = -(outside_distance + inside_distance)
        else:
            outside_distance = sqrt(
                Min(0, 1-sqrt(r)) ** 2 + Min(0, 0.5  - Abs(l)) ** 2
            )
            inside_distance = -1 * Min(
                Abs(Min(0, sqrt(r) - 1)), Abs(Min(0, Abs(l) - 0.5))
            )
            sdf = -(outside_distance + inside_distance)*(0.2+0.8*exp(-12.5*((l)**2.)))
        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (center[0] - length / 2 - radius*(1.+max(default_curve_control["r0"])-min(default_curve_control["rr"])), center[0] + length / 2 + radius*(1.+max(default_curve_control["r0"])+max(default_curve_control["rr"]))),
                Parameter("y"): (center[1] - radius * ( 1+ np.max(default_curve_control["r0"]) + np.max(default_curve_control["rr"]) + np.max(np.abs(default_curve_control["s1"])) + np.max(np.abs(default_curve_control["s2"]))) , center[1] + radius * ( 1+ np.max(default_curve_control["r0"]) + np.max(default_curve_control["rr"]) + np.max(np.abs(default_curve_control["s1"])) + np.max(np.abs(default_curve_control["s2"])))),
                Parameter("z"): (center[2] - radius * ( 1+ np.max(default_curve_control["r0"]) + np.max(default_curve_control["rr"]) + np.max(np.abs(default_curve_control["s3"])) + np.max(np.abs(default_curve_control["s4"]))) , center[2] + radius * ( 1+ np.max(default_curve_control["r0"]) + np.max(default_curve_control["rr"]) + np.max(np.abs(default_curve_control["s3"])) + np.max(np.abs(default_curve_control["s4"])))),
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
