#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 23 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""
import numpy as np
from math import cos, sin
from cmath import exp


class PMNS:
  def __init__(self, th12, th13, th23, d):

    self.theta12 = th12
    self.theta13 = th13
    self.theta23 = th23
    self.delta = d

    # Fill PMNS matrix
    r13 = self.R13(th13)
    r12 = self.R12(th12)
    r23 = self.R23(th23)
    delta = self.Delta(d)

    self.pmns = np.ascontiguousarray(
        np.dot(np.dot(np.dot(r23, delta), np.dot(r13, delta.conjugate())), r12),
        dtype=np.complex128,
    )

    # Reduced mixing matrix U = R_{13} R_{12}
    self.U = np.ascontiguousarray(np.dot(r13, r12), dtype=np.complex128)

  # These are the orthogonal/unitary matrices factorising the PMNS matrix,
  # U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12}
  def R23(self, th):
      return np.array([
          [1.0, 0.0, 0.0],
          [0.0, cos(th), sin(th)],
          [0.0, -sin(th), cos(th)]
      ], dtype=np.complex128)

  def R13(self, th):
      return np.array([
          [cos(th), 0.0, sin(th)],
          [0.0, 1.0, 0.0],
          [-sin(th), 0.0, cos(th)]
      ], dtype=np.complex128)

  def R12(self, th):
      return np.array([
          [cos(th), sin(th), 0.0],
          [-sin(th), cos(th), 0.0],
          [0.0, 0.0, 1.0]
      ], dtype=np.complex128)

  def Delta(self, d):
      delta = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0, 0.0, 1.0]], dtype=np.complex128)
      delta[2][2] = exp(1j*d)
      return delta

  def __getitem__(self, i):
      return self.pmns[i]

  def transpose(self):
      return self.pmns.transpose()

  def conjugate(self):
      return self.pmns.conjugate()
