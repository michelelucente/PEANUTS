#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 23 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""
import numpy as np
import numba as nb
from numba.experimental import jitclass

from math import cos, sin
from cmath import exp

pmns = [('theta12', nb.float64),
        ('theta13', nb.float64),
        ('theta23', nb.float64),
        ('delta', nb.float64),
        ('pmns', nb.complex128[:,::1]),
        ('U', nb.complex128[:,::1])]

@jitclass(pmns)
class PMNS:

  def __init__(self,th12,th13,th23,d):

    self.theta12 = th12
    self.theta13 = th13
    self.theta23 = th23
    self.delta = d

    # Fill PMNS matrix
    r13 = self.R13(th13)
    r12 = self.R12(th12)
    r23 = self.R23(th23)
    delta = self.Delta(d)

    self.pmns = np.dot(np.dot(np.dot(r23, delta), np.dot(r13, delta.conjugate())), r12)

    self.U = np.dot(r13, r12)


  # These are the orthogonal/unitary matrices factorising the PMNS matrix,
  # U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12}
  def R23(self,th):
      return np.array([
          [1., 0., 0.],
          [0., cos(th), sin(th)],
          [0., -sin(th), cos(th)]
      ], dtype=nb.complex128)

  def R13(self,th):
      return np.array([
          [cos(th), 0., sin(th)],
          [0., 1., 0.],
          [-sin(th), 0., cos(th)]
      ], dtype=nb.complex128)

  def R12(self,th):
      return np.array([
          [cos(th), sin(th), 0.],
          [-sin(th), cos(th), 0.],
          [0., 0., 1.]
      ], dtype=nb.complex128)

  def Delta(self,d):
      delta = np.array([[1.,0.,0.],[0.,1.,0.],[0., 0., 1.]], dtype=nb.complex128)
      delta[2][2] = exp(1j*d)
      return delta

  def __getitem__(self, i):
      return self.pmns[i]

  def transpose(self):
      return self.pmns.transpose()

  def conjugate(self):
      return self.pmns.conjugate()
