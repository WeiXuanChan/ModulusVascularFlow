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
    Max,
    sqrt,
    pi,
    sin,
    cos,
    atan2,
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
class StraightTube(Geometry):
    """
    3D Cylinder Straigtht
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

    def __init__(self, center, radius, length,cline_vec=None,ry_vec=None, parameterization=Parameterization(),geom='interior'):#geom=["interior","wall","inlet","outlet"]
        # make sympy symbols to use
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        if cline_vec is None:
            cline_vec=np.array([1.,0.,0.])
        cline_vec=cline_vec/np.linalg.norm(cline_vec)
        if ry_vec is None:
            if abs(cline_vec[2])<0.9:
                ry_vec=np.cross(np.array([0.,0.,1.]),cline_vec)
            else:
                ry_vec=np.cross(np.array([0.,1.,0.]),cline_vec)
            ry_vec=ry_vec/np.linalg.norm(ry_vec)
        rz_vec=np.cross(cline_vec,ry_vec)
        if geom=='interior':
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
                    "x": center[0] + 0.5 * l * length *cline_vec[0] + radius * r * sin(theta) *ry_vec[0] + radius * r * cos(theta) *rz_vec[0],
                    "y": center[1] + 0.5 * l * length *cline_vec[1] + radius * r * sin(theta) *ry_vec[1] + radius * r * cos(theta) *rz_vec[1],
                    "z": center[2] + 0.5 * l * length *cline_vec[2] + radius * r * sin(theta) *ry_vec[2] + radius * r * cos(theta) *rz_vec[2],
                    "normal_x": sin(theta)*ry_vec[0] + cos(theta)*rz_vec[0],
                    "normal_y": sin(theta)*ry_vec[1] + cos(theta)*rz_vec[1],
                    "normal_z": sin(theta)*ry_vec[2] + cos(theta)*rz_vec[2],
                },
                parameterization=curve_parameterization,
                area=length * pi * radius**2.,
            )
            
            curves = [curve_1]
    
            # calculate SDF
            sdf =  Min(0,radius - sqrt((y*cline_vec[2]-z*cline_vec[1])**2 + (z*cline_vec[0]-x*cline_vec[2])**2 + (x*cline_vec[1]-y*cline_vec[0])**2))
    
            # calculate bounds
            bounds = Bounds(
                {
                    Parameter("x"): (center[0] - abs(0.5 *  length *cline_vec[0]) - abs(radius * sin(theta) *ry_vec[0]) - abs(radius * cos(theta) *rz_vec[0]), center[0] + abs(0.5 *  length *cline_vec[0]) + abs(radius * sin(theta) *ry_vec[0]) + abs(radius * cos(theta) *rz_vec[0])),
                    Parameter("y"): (center[1] - abs(0.5 *  length *cline_vec[1]) - abs(radius * sin(theta) *ry_vec[1]) - abs(radius * cos(theta) *rz_vec[1]), center[1] + abs(0.5 *  length *cline_vec[1]) + abs(radius * sin(theta) *ry_vec[1]) + abs(radius * cos(theta) *rz_vec[1])),
                    Parameter("z"): (center[2] - abs(0.5 *  length *cline_vec[2]) - abs(radius * sin(theta) *ry_vec[2]) - abs(radius * cos(theta) *rz_vec[2]), center[2] + abs(0.5 *  length *cline_vec[2]) + abs(radius * sin(theta) *ry_vec[2]) + abs(radius * cos(theta) *rz_vec[2])),
                },
                parameterization=parameterization,
            )
        elif geom=='outlet':
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
                cir_center=center
            else:
                cir_center=[center[0] + 0.5 * length*cline_vec[0],center[1] + 0.5 * length *cline_vec[1],center[2] + 0.5 * length *cline_vec[2]]
            curve_2 = SympyCurve(
                functions={
                    "x": cir_center[0] + sqrt(r) * radius * sin(theta)*ry_vec[0] + sqrt(r) * radius * cos(theta)*rz_vec[0],
                    "y": cir_center[1] + sqrt(r) * radius * sin(theta)*ry_vec[1] + sqrt(r) * radius * cos(theta)*rz_vec[1],
                    "z": cir_center[2] + sqrt(r) * radius * sin(theta)*ry_vec[2] + sqrt(r) * radius * cos(theta)*rz_vec[2],
                    "normal_x": cline_vec[0],
                    "normal_y": cline_vec[1],
                    "normal_z": cline_vec[2],
                },
                parameterization=curve_parameterization,
                area=pi * radius**2,
            )
            
            curves = [curve_2]
    
            # calculate SDF
            sdf = (sqrt(cir_center[0]**2.+cir_center[1]**2.+cir_center[2]**2.) - (x*cline_vec[0]+y*cline_vec[1]+z*cline_vec[2]))
    
            # calculate bounds
            bounds = Bounds(
                {
                    Parameter("x"): (center[0] + (0.5 *  length *cline_vec[0]) - abs(radius * sin(theta) *ry_vec[0]) - abs(radius * cos(theta) *rz_vec[0]), center[0] + abs(0.5 *  length *cline_vec[0]) + abs(radius * sin(theta) *ry_vec[0]) + abs(radius * cos(theta) *rz_vec[0])),
                    Parameter("y"): (center[1] + (0.5 *  length *cline_vec[1]) - abs(radius * sin(theta) *ry_vec[1]) - abs(radius * cos(theta) *rz_vec[1]), center[1] + abs(0.5 *  length *cline_vec[1]) + abs(radius * sin(theta) *ry_vec[1]) + abs(radius * cos(theta) *rz_vec[1])),
                    Parameter("z"): (center[2] + (0.5 *  length *cline_vec[2]) - abs(radius * sin(theta) *ry_vec[2]) - abs(radius * cos(theta) *rz_vec[2]), center[2] + abs(0.5 *  length *cline_vec[2]) + abs(radius * sin(theta) *ry_vec[2]) + abs(radius * cos(theta) *rz_vec[2])),
                },
                parameterization=parameterization,
            )
        elif geom=='inlet':
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
                cir_center=center
            else:
                cir_center=[center[0] - 0.5 * length*cline_vec[0],center[1] - 0.5 * length *cline_vec[1],center[2] - 0.5 * length *cline_vec[2]]
            curve_3 = SympyCurve(
                functions={
                    "x": cir_center[0] + sqrt(r) * radius * sin(theta)*ry_vec[0] + sqrt(r) * radius * cos(theta)*rz_vec[0],
                    "y": cir_center[1] + sqrt(r) * radius * sin(theta)*ry_vec[1] + sqrt(r) * radius * cos(theta)*rz_vec[1],
                    "z": cir_center[2] + sqrt(r) * radius * sin(theta)*ry_vec[2] + sqrt(r) * radius * cos(theta)*rz_vec[2],
                    "normal_x": -cline_vec[0],
                    "normal_y": -cline_vec[1],
                    "normal_z": -cline_vec[2],
                },
                parameterization=curve_parameterization,
                area=pi * radius**2,
            )
            curves = [curve_3]
    
            # calculate SDF
            sdf = (sqrt(cir_center[0]**2.+cir_center[1]**2.+cir_center[2]**2.) - (x*cline_vec[0]+y*cline_vec[1]+z*cline_vec[2]))
    
            # calculate bounds
            bounds = Bounds(
                {
                    Parameter("x"): (center[0] - (0.5 *  length *cline_vec[0]) - abs(radius * sin(theta) *ry_vec[0]) - abs(radius * cos(theta) *rz_vec[0]), center[0] + abs(0.5 *  length *cline_vec[0]) + abs(radius * sin(theta) *ry_vec[0]) + abs(radius * cos(theta) *rz_vec[0])),
                    Parameter("y"): (center[1] - (0.5 *  length *cline_vec[1]) - abs(radius * sin(theta) *ry_vec[1]) - abs(radius * cos(theta) *rz_vec[1]), center[1] + abs(0.5 *  length *cline_vec[1]) + abs(radius * sin(theta) *ry_vec[1]) + abs(radius * cos(theta) *rz_vec[1])),
                    Parameter("z"): (center[2] - (0.5 *  length *cline_vec[2]) - abs(radius * sin(theta) *ry_vec[2]) - abs(radius * cos(theta) *rz_vec[2]), center[2] + abs(0.5 *  length *cline_vec[2]) + abs(radius * sin(theta) *ry_vec[2]) + abs(radius * cos(theta) *rz_vec[2])),
                },
                parameterization=parameterization,
            )
        elif geom=='wall':
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
                    "x": center[0] + 0.5 * l * length *cline_vec[0] + radius * sin(theta) *ry_vec[0] + radius * cos(theta) *rz_vec[0],
                    "y": center[1] + 0.5 * l * length *cline_vec[1] + radius * sin(theta) *ry_vec[1] + radius * cos(theta) *rz_vec[1],
                    "z": center[2] + 0.5 * l * length *cline_vec[2] + radius * sin(theta) *ry_vec[2] + radius * cos(theta) *rz_vec[2],
                    "normal_x": sin(theta)*ry_vec[0] + cos(theta)*rz_vec[0],
                    "normal_y": sin(theta)*ry_vec[1] + cos(theta)*rz_vec[1],
                    "normal_z": sin(theta)*ry_vec[2] + cos(theta)*rz_vec[2],
                },
                parameterization=curve_parameterization,
                area=length * 2 * pi * radius,
            )
            
            curves = [curve_1]
    
            # calculate SDF
            sdf = Min(0,radius - sqrt((y*cline_vec[2]-z*cline_vec[1])**2 + (z*cline_vec[0]-x*cline_vec[2])**2 + (x*cline_vec[1]-y*cline_vec[0])**2))
    
            # calculate bounds
            bounds = Bounds(
                {
                    Parameter("x"): (center[0] - abs(0.5 *  length *cline_vec[0]) - abs(radius * sin(theta) *ry_vec[0]) - abs(radius * cos(theta) *rz_vec[0]), center[0] + abs(0.5 *  length *cline_vec[0]) + abs(radius * sin(theta) *ry_vec[0]) + abs(radius * cos(theta) *rz_vec[0])),
                    Parameter("y"): (center[1] - abs(0.5 *  length *cline_vec[1]) - abs(radius * sin(theta) *ry_vec[1]) - abs(radius * cos(theta) *rz_vec[1]), center[1] + abs(0.5 *  length *cline_vec[1]) + abs(radius * sin(theta) *ry_vec[1]) + abs(radius * cos(theta) *rz_vec[1])),
                    Parameter("z"): (center[2] - abs(0.5 *  length *cline_vec[2]) - abs(radius * sin(theta) *ry_vec[2]) - abs(radius * cos(theta) *rz_vec[2]), center[2] + abs(0.5 *  length *cline_vec[2]) + abs(radius * sin(theta) *ry_vec[2]) + abs(radius * cos(theta) *rz_vec[2])),
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
        if geom=='wall':
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
class HeartMyocardium(Geometry):
    """
    3D heart Myocardium
    Longitudinal axis parallel to z-axis
    base at (0,0,0)
    apex at (0,0,-longlength)
    center of  prolate spheroid at (0,0,-center_to_base)
    radius of prolate spheroid =radius

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

    def __init__(self, longitudinal_length, radius, center_to_base, thickness, parameterization=Parameterization()):
        # make sympy symbols to use
        longi, theta, radial,longi_epi = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1)), Symbol(csg_curve_naming(2)), Symbol(csg_curve_naming(3))
        
        minor_semi_axis=radius
        major_semi_axis=longitudinal_length-center_to_base
        longi_min=-center_to_base/major_semi_axis
        longi_min_outer=-center_to_base/(major_semi_axis+thickness)
        # surface of the cylinder
        curve_parameterization = Parameterization(
            {longi: (longi_min, 1),longi_epi: (longi_min_outer, 1), theta: (0, 2 * pi), radial:(0,1)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        slice_radius=minor_semi_axis* sqrt(1.-longi**2.)
        func_x=slice_radius * cos(theta)
        func_y=slice_radius * sin(theta)
        func_z=-longi * major_semi_axis
        outward_normal_x=sqrt(1.-longi**2.)/minor_semi_axis* cos(theta)
        outward_normal_y=sqrt(1.-longi**2.)/minor_semi_axis* sin(theta)
        outward_normal_z=-longi/major_semi_axis
        #outward_normal_z=Min(0.,outward_normal_z)
        outward_normal_correct=sqrt((1.-longi**2.)/minor_semi_axis**2.+outward_normal_z**2.)
        outward_normal_x=outward_normal_x/outward_normal_correct
        outward_normal_y=outward_normal_y/outward_normal_correct
        outward_normal_z=outward_normal_z/outward_normal_correct
        curve_1 = SympyCurve(
            functions={
                "x": func_x,
                "y": func_y,
                "z": -center_to_base +func_z,
                "normal_x": -outward_normal_x,
                "normal_y": -outward_normal_y,
                "normal_z": -outward_normal_z,
            },
            parameterization=curve_parameterization,
            area=np.pi*minor_semi_axis*(np.pi/2.-np.arcsin(longi_min)+longi_min*np.sqrt(1-longi_min**2.)),
        )
        outer_slice_radius=(minor_semi_axis+thickness)* sqrt(1.-longi_epi**2.)
        outer_func_x=slice_radius * cos(theta)
        outer_func_y=slice_radius * sin(theta)
        outer_func_z=-longi_epi * (major_semi_axis+thickness)
        outer_outward_normal_x=sqrt(1.-longi_epi**2.)/(minor_semi_axis+thickness)* cos(theta)
        outer_outward_normal_y=sqrt(1.-longi_epi**2.)/(minor_semi_axis+thickness)* sin(theta)
        outer_outward_normal_z=-longi_epi/(major_semi_axis+thickness)
        #outward_normal_z=Min(0.,outward_normal_z)
        outer_outward_normal_correct=sqrt((1.-longi_epi**2.)/(major_semi_axis+thickness)**2.+outer_outward_normal_z**2.)
        outer_outward_normal_x=outer_outward_normal_x/outer_outward_normal_correct
        outer_outward_normal_y=outer_outward_normal_y/outer_outward_normal_correct
        outer_outward_normal_z=outer_outward_normal_z/outer_outward_normal_correct
        curve_2 = SympyCurve(
            functions={
                "x": outer_func_x,
                "y": outer_func_y,
                "z": -center_to_base +outer_func_z,
                "normal_x": outer_outward_normal_x,
                "normal_y": outer_outward_normal_y,
                "normal_z": outer_outward_normal_z,
            },
            parameterization=curve_parameterization,
            area=np.pi*(minor_semi_axis+thickness)*(np.pi/2.-np.arcsin(longi_min_outer)+longi_min_outer*np.sqrt(1-longi_min_outer**2.)),
        )
        curve_3 = SympyCurve(
            functions={
                "x": (minor_semi_axis* np.sqrt(1.-longi_min**2.)+thickness*radial)* cos(theta),
                "y": (minor_semi_axis* np.sqrt(1.-longi_min**2.)+thickness*radial)* sin(theta),
                "z": 0.,
                "normal_x": 0.,
                "normal_y": 0.,
                "normal_z": 1.,
            },
            parameterization=curve_parameterization,
            area=np.pi*((minor_semi_axis* np.sqrt(1.-longi_min**2.)+thickness)**2.-(minor_semi_axis* np.sqrt(1.-longi_min**2.))**2.),
        )
        curves = [curve_1, curve_2, curve_3]
        
        x,y,z = Symbol("x"),Symbol("y"),Symbol("z")
        # calculate SDF
        toEndo_distance = (x/minor_semi_axis)**2.+(y/minor_semi_axis)**2.+((z+center_to_base)/major_semi_axis)**2.-1.
        toEpi_distance = 1.-(x/(minor_semi_axis+thickness))**2.+(y/(minor_semi_axis+thickness))**2.+((z+center_to_base)/(major_semi_axis+thickness))**2.
        base_distance = -z
        sdf = toEndo_distance*toEpi_distance*Max(0.,base_distance)+Min(0.,base_distance)
        
        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (-minor_semi_axis-thickness, minor_semi_axis+thickness),
                Parameter("y"): (-minor_semi_axis-thickness, minor_semi_axis+thickness),
                Parameter("z"): (-longitudinal_length-thickness,0.),
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
        
class HeartMyocardium_LCR(Geometry):
    """
    ellipsoid HeartMyocardium with thickness added via the surface normal

    Parameters
    ----------
    point_1 : tuple with 3 ints or floats
        lower bound point of box
    point_2 : tuple with 3 ints or floats
        upper bound point of box
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self,longilength,center_to_base,endoradius,thickness,parameterization=Parameterization(),geom='myocardium'):#geom=["myocardium","endo","epi"]
        # make sympy symbols to use
        l,c,r = Symbol("z"), Symbol("y"), Symbol("x")
        s_l, s_c, s_r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1)), Symbol(csg_curve_naming(2))
        circumferential_halflength=np.pi*endoradius

        # surface of the box
        curve_parameterization = Parameterization({s_l: (0, 1), s_c: (-1, 1), s_r: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        major_semiaxis=longilength-center_to_base
        #base_endoradius=endoradius*np.sqrt(1.-(center_to_base/major_semiaxis)**2.)
        #centroid_to_base_thru_l=sqrt((endoradius+s_r*thickness)**2.*(1.-(center_to_base/(major_semiaxis+s_r*thickness))**2.)+center_to_base**2.)
        curve_l0 = SympyCurve(
            functions={
                "z": 0.,
                "y": s_c*circumferential_halflength,
                "x": 0.,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=1.,
        )
        curve_r0 = SympyCurve(
            functions={
                "z": s_l*longilength,
                "y": s_c*circumferential_halflength,
                "x": 0.,
                "normal_x": -1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=1.,
        )
        #epi_longi=center_to_base*thickness/np.sqrt((endoradius+thickness)**2.*(1.-(center_to_base/(major_semiaxis+thickness))**2.)+center_to_base**2.)
        curve_r1 = SympyCurve(
            functions={
                "z": s_l*longilength,
                "y": s_c*circumferential_halflength,
                "x": thickness,
                "normal_x": 1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=1.,
        )
        if geom=='base':
            curves = [curve_l0]
        elif geom=='endo':
            curves = [curve_r0]
        elif geom=='epi':
            curves = [curve_r1]
        elif geom=='cirjoint':
            curve_cirjoint0 = SympyCurve(
                functions={
                    "z": s_l*longilength,
                    "y": -1.*circumferential_halflength,
                    "x": s_r*thickness,
                    "normal_x": 0,
                    "normal_y": -1,
                    "normal_z": 0,
                },
                parameterization=curve_parameterization,
                area=1.,
            )
            curve_cirjoint1 = SympyCurve(
                functions={
                    "z": s_l*longilength,
                    "y": circumferential_halflength,
                    "x": s_r*thickness,
                    "normal_x": 0,
                    "normal_y": -1,
                    "normal_z": 0,
                },
                parameterization=curve_parameterization,
                area=1.,
            )
            curves = [curve_cirjoint0,curve_cirjoint1]
        elif geom=='myocardium':
            curve_myocardium = SympyCurve(
                functions={
                    "z": s_l*longilength,
                    "y": s_c*circumferential_halflength,
                    "x": s_r*thickness,
                    "normal_x": 0,
                    "normal_y": 0,
                    "normal_z": -1,
                },
                parameterization=curve_parameterization,
                area=1.,
            )
            curves = [curve_myocardium]
        elif geom=='myocardium+endo':
            curve_myocardium = SympyCurve(
                functions={
                    "z": s_l*longilength,
                    "y": s_c*circumferential_halflength,
                    "x": s_r*thickness,
                    "normal_x": 0,
                    "normal_y": 0,
                    "normal_z": -1,
                },
                parameterization=curve_parameterization,
                area=10.,
            )
            curves = [curve_myocardium,curve_r0]
        else:
            curves = [curve_l0,curve_r0,curve_r1]

        # calculate SDF
        l_dist = Abs(l-0.5*longilength) - 0.5 * longilength
        c_dist = Abs(c) - circumferential_halflength
        r_dist = Abs(r-0.5*thickness) - 0.5 * thickness
        outside_distance = sqrt(
            Max(l_dist, 0) ** 2 + Max(c_dist, 0) ** 2 + Max(r_dist, 0) ** 2
        )
        centroid_to_base_thru_l_nocurve=sqrt((endoradius+r)**2.*(1.-(center_to_base/(major_semiaxis+r))**2.)+center_to_base**2.)
        #inside_distance = Max(center_to_base*r/centroid_to_base_thru_l_nocurve-l, 0)
        #inside_distance = Min(Max(l_dist, c_dist, r_dist), 0)
        if geom=='cirjoint':
            sdf = Abs(c)-circumferential_halflength
        else:
            sdf = -outside_distance

        # calculate bounds
        if geom=='base':
            bounds = Bounds(
                {
                    Parameter("z"): (0., 0.),
                    Parameter("y"): (-circumferential_halflength, circumferential_halflength),
                    Parameter("x"): (0, 0.),
                },
                parameterization=parameterization,
            )
        elif geom=='endo':
            bounds = Bounds(
                {
                    Parameter("z"): (0., longilength),
                    Parameter("y"): (-circumferential_halflength, circumferential_halflength),
                    Parameter("x"): (0., 0.),
                },
                parameterization=parameterization,
            )
        elif geom=='epi':
            bounds = Bounds(
                {
                    Parameter("z"): (0., longilength),
                    Parameter("y"): (-circumferential_halflength, circumferential_halflength),
                    Parameter("x"): (thickness, thickness),
                },
                parameterization=parameterization,
            )
        else:
            bounds = Bounds(
                {
                    Parameter("z"): (0., longilength),
                    Parameter("y"): (-circumferential_halflength, circumferential_halflength),
                    Parameter("x"): (0, thickness),
                },
                parameterization=parameterization,
            )

        # initialize Box
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )

class HeartMyocardium_LCR_approx(Geometry):
    """
    ellipsoid Heartmyocardium with thickness added to semi-major and semi-minor axis(thickness is non-constant)

    Parameters
    ----------
    point_1 : tuple with 3 ints or floats
        lower bound point of box
    point_2 : tuple with 3 ints or floats
        upper bound point of box
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self,longilength,center_to_base,endoradius,thickness,parameterization=Parameterization(),geom='myocardium'):#geom=["myocardium","endo","epi"]
        # make sympy symbols to use
        l,c,r = Symbol("z"), Symbol("y"), Symbol("x")
        s_l, s_c, s_r = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1)), Symbol(csg_curve_naming(2))
        circumferential_halflength=np.pi*endoradius

        # surface of the box
        curve_parameterization = Parameterization({s_l: (0, 1), s_c: (-1, 1), s_r: (0, 1)})
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        major_semiaxis=longilength-center_to_base
        #base_endoradius=endoradius*np.sqrt(1.-(center_to_base/major_semiaxis)**2.)
        #centroid_to_base_thru_l=sqrt((endoradius+s_r*thickness)**2.*(1.-(center_to_base/(major_semiaxis+s_r*thickness))**2.)+center_to_base**2.)
        curve_l0 = SympyCurve(
            functions={
                "z": 0.,
                "y": s_c*(circumferential_halflength+np.pi*s_r*thickness),
                "x": s_r*thickness,
                "normal_x": 0,
                "normal_y": 0,
                "normal_z": -1,
            },
            parameterization=curve_parameterization,
            area=1.,
        )
        curve_r0 = SympyCurve(
            functions={
                "z": s_l*longilength,
                "y": s_c*circumferential_halflength,
                "x": 0.,
                "normal_x": -1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=1.,
        )
        #epi_longi=center_to_base*thickness/np.sqrt((endoradius+thickness)**2.*(1.-(center_to_base/(major_semiaxis+thickness))**2.)+center_to_base**2.)
        curve_r1 = SympyCurve(
            functions={
                "z": s_l*(longilength+thickness),
                "y": s_c*(circumferential_halflength+np.pi*thickness),
                "x": thickness,
                "normal_x": 1,
                "normal_y": 0,
                "normal_z": 0,
            },
            parameterization=curve_parameterization,
            area=1.,
        )
        if geom=='base':
            curves = [curve_l0]
        elif geom=='endo':
            curves = [curve_r0]
        elif geom=='epi':
            curves = [curve_r1]
        elif geom=='cirjoint':
            curve_cirjoint0 = SympyCurve(
                functions={
                    "z": s_l*(longilength+s_r*thickness),
                    "y": -1.*(circumferential_halflength+np.pi*s_r*thickness),
                    "x": s_r*thickness,
                    "normal_x": 0,
                    "normal_y": -1,
                    "normal_z": 0,
                },
                parameterization=curve_parameterization,
                area=1.,
            )
            curve_cirjoint1 = SympyCurve(
                functions={
                    "z": s_l*(longilength+s_r*thickness),
                    "y": (circumferential_halflength+np.pi*s_r*thickness),
                    "x": s_r*thickness,
                    "normal_x": 0,
                    "normal_y": -1,
                    "normal_z": 0,
                },
                parameterization=curve_parameterization,
                area=1.,
            )
            curves = [curve_cirjoint0,curve_cirjoint1]
        elif geom=='myocardium':
            curve_myocardium = SympyCurve(
                functions={
                    "z": s_l*(longilength+s_r*thickness),
                    "y": s_c*(circumferential_halflength+np.pi*s_r*thickness),
                    "x": s_r*thickness,
                    "normal_x": 0,
                    "normal_y": 0,
                    "normal_z": -1,
                },
                parameterization=curve_parameterization,
                area=1.,
            )
            curves = [curve_myocardium]
        elif geom=='myocardium+endo':
            curve_myocardium = SympyCurve(
                functions={
                    "z": s_l*(longilength+s_r*thickness),
                    "y": s_c*(circumferential_halflength+np.pi*s_r*thickness),
                    "x": s_r*thickness,
                    "normal_x": 0,
                    "normal_y": 0,
                    "normal_z": -1,
                },
                parameterization=curve_parameterization,
                area=10.,
            )
            curves = [curve_myocardium,curve_r0]
        else:
            curves = [curve_l0,curve_r0,curve_r1]

        # calculate SDF
        l_dist = Abs(l-0.5*(longilength+r)) - 0.5 * (longilength+r)
        c_dist = Abs(c) - (circumferential_halflength+np.pi*r)
        r_dist = Abs(r-0.5*thickness) - 0.5 * thickness
        outside_distance = sqrt(
            Max(l_dist, 0) ** 2 + Max(c_dist, 0) ** 2 + Max(r_dist, 0) ** 2
        )
        if geom=='cirjoint':
            sdf = Abs(c)-(circumferential_halflength+np.pi*r)
        else:
            sdf = -outside_distance

        # calculate bounds
        if geom=='base':
            bounds = Bounds(
                {
                    Parameter("z"): (0., 0.),
                    Parameter("y"): (-(circumferential_halflength+np.pi*thickness), (circumferential_halflength+np.pi*thickness)),
                    Parameter("x"): (0, 0.),
                },
                parameterization=parameterization,
            )
        elif geom=='endo':
            bounds = Bounds(
                {
                    Parameter("z"): (0., longilength+thickness),
                    Parameter("y"): (-(circumferential_halflength+np.pi*thickness), (circumferential_halflength+np.pi*thickness)),
                    Parameter("x"): (0., 0.),
                },
                parameterization=parameterization,
            )
        elif geom=='epi':
            bounds = Bounds(
                {
                    Parameter("z"): (0., longilength+thickness),
                    Parameter("y"): (-(circumferential_halflength+np.pi*thickness), (circumferential_halflength+np.pi*thickness)),
                    Parameter("x"): (thickness, thickness),
                },
                parameterization=parameterization,
            )
        else:
            bounds = Bounds(
                {
                    Parameter("z"): (0., longilength+thickness),
                    Parameter("y"): (-(circumferential_halflength+np.pi*thickness), (circumferential_halflength+np.pi*thickness)),
                    Parameter("x"): (0, thickness),
                },
                parameterization=parameterization,
            )

        # initialize Box
        super().__init__(
            curves,
            _sympy_sdf_to_sdf(sdf),
            dims=3,
            bounds=bounds,
            parameterization=parameterization,
        )


class HeartChamber(Geometry):
    """
    3D heart Chamber
    Longitudinal axis parallel to z-axis
    straight tube with semisphere joint at (0,0,0)
    with inlet and outlet circle at base
    base at (0,0,-longlength)
    apex at (0,0,-longlength/2)
    radius =longlength/2
    center of inlet (-longlength/4,0,-longlength) radius =longlength/8
    center of outlet (longlength/4,0,-longlength) radius =longlength/8

    Parameters
    ----------
    parameterization : Parameterization
        Parameterization of geometry.
    """

    def __init__(self, longitudinal_length, parameterization=Parameterization(),geom='chamber'):
        # make sympy symbols to use
        z_l, radial, angle = Symbol(csg_curve_naming(0)), Symbol(csg_curve_naming(1)), Symbol(csg_curve_naming(2)), Symbol(csg_curve_naming(3))
        
        # surface of the cylinder
        curve_parameterization = Parameterization(
            {z_l: (0, 1),radial: (0, 1), angle:(-1,1)}
        )
        curve_parameterization = Parameterization.combine(
            curve_parameterization, parameterization
        )
        base_radius=longitudinal_length/2.
        theta=np.pi*angle
        curve_base = SympyCurve(
            functions={
                "x": base_radius*radial* cos(theta),
                "y": base_radius*radial* sin(theta),
                "z": longitudinal_length,
                "normal_x": 0.,
                "normal_y": 0.,
                "normal_z": 1.,
            },
            parameterization=curve_parameterization,
            area=np.pi*base_radius**2.,
        )
        curve_inlet = SympyCurve(
            functions={
                "x": base_radius/4.*radial* cos(theta)-base_radius/2.,
                "y": base_radius/4.*radial* sin(theta),
                "z": longitudinal_length,
                "normal_x": 0.,
                "normal_y": 0.,
                "normal_z": 1.,
            },
            parameterization=curve_parameterization,
            area=np.pi*base_radius**2.,
        )
        curve_outlet = SympyCurve(
            functions={
                "x": base_radius/4.*radial* cos(theta)+base_radius/2.,
                "y": base_radius/4.*radial* sin(theta),
                "z": longitudinal_length,
                "normal_x": 0.,
                "normal_y": 0.,
                "normal_z": 1.,
            },
            parameterization=curve_parameterization,
            area=np.pi*base_radius**2.,
        )
        curve_topwall = SympyCurve(
            functions={
                "x": base_radius* cos(theta),
                "y": base_radius* sin(theta),
                "z": z_l*longitudinal_length,
                "normal_x": cos(theta),
                "normal_y": sin(theta),
                "normal_z": 0.,
            },
            parameterization=curve_parameterization,
            area=np.pi*2.*base_radius*longitudinal_length,
        )
        hemisphere_radius=base_radius
        z_plane_radius_ratio=sqrt(1.-z_l**2.)
        curve_apex = SympyCurve(
            functions={
                "x": z_plane_radius_ratio*hemisphere_radius* cos(theta),
                "y": z_plane_radius_ratio*hemisphere_radius* sin(theta),
                "z": -z_l*hemisphere_radius,
                "normal_x": z_plane_radius_ratio*cos(theta),
                "normal_y": z_plane_radius_ratio*sin(theta),
                "normal_z": -z_l,
            },
            parameterization=curve_parameterization,
            area=2.*np.pi*hemisphere_radius**2.,
        )
        split_z_plane_radius_ratio=sqrt(1.-Max(0.,z_l*3.-2.)**2.)
        curve_interior = SympyCurve(
            functions={
                "x": split_z_plane_radius_ratio*radial*hemisphere_radius* cos(theta),
                "y": split_z_plane_radius_ratio*radial*hemisphere_radius* sin(theta),
                "z": -(z_l*3.-2.)*hemisphere_radius*radial,
                "normal_x": split_z_plane_radius_ratio*cos(theta),
                "normal_y": split_z_plane_radius_ratio*sin(theta),
                "normal_z": -z_l,
            },
            parameterization=curve_parameterization,
            area=2.*np.pi*hemisphere_radius**2.,
        )
        if geom=='chamber':
            curves = [curve_interior]
        elif geom=='inlet':
            curves = [curve_inlet]
        elif geom=='outlet':
            curves = [curve_outlet]
        elif geom=='base':
            curves = [curve_base]
        elif geom=='wall':
            curves = [curve_topwall,curve_apex]
        
        x,y,z = Symbol("x"),Symbol("y"),Symbol("z")
        # calculate SDF
        split_z_plane_radius_ratio_distance = sqrt(1.-Max(0.,-z/hemisphere_radius)**2.)
        wall_distance=split_z_plane_radius_ratio_distance-(x/hemisphere_radius)**2.-(y/hemisphere_radius)**2.
        base_distance = longitudinal_length-z
        sdf = base_distance*wall_distance
        if geom=='chamber':
            sdf = Max(0.,sdf)
        
        # calculate bounds
        bounds = Bounds(
            {
                Parameter("x"): (-hemisphere_radius, hemisphere_radius),
                Parameter("y"): (-hemisphere_radius, hemisphere_radius),
                Parameter("z"): (-longitudinal_length/2,longitudinal_length),
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