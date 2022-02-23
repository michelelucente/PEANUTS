#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:33:39 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
"""
import numpy as np

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
