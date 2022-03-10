#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:33:39 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
"""
import numpy as np
from numpy.linalg import multi_dot

from math import cos, sin
from cmath import exp


# These are the orthogonal/unitary matrices factorising the PMNS matrix, 
# U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12}
def R23(th):
    return np.matrix([
        [1, 0, 0],
        [0, cos(th), sin(th)],
        [0, -sin(th), cos(th)]
    ])

def R13(th):
    return np.matrix([
        [cos(th), 0, sin(th)],
        [0, 1, 0],
        [-sin(th), 0, cos(th)]
    ])

def R12(th):
    return np.matrix([
        [cos(th), sin(th), 0],
        [-sin(th), cos(th), 0],
        [0, 0, 1]
    ])

def Delta(d):
    return np.matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, exp(1j*d)]
    ])


def PMNS (th12, th13, th23, d):
    """PMNS(th12, th13, th23, d) returns the PMNS mixing, ignoring Majorana phases,
    U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12}:
    - thij are the mixing angles;
    - d is the CP-violating phase."""
    
    return multi_dot([R23(th23), Delta(d), R13(th13), Delta(-d), R12(th12)])
