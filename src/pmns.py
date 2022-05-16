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
from numpy.linalg import multi_dot

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
  
    self.pmns = multi_dot([r23, delta, r13, delta.conjugate(), r12])

    self.U = np.dot(r13, r12)


  # These are the orthogonal/unitary matrices factorising the PMNS matrix, 
  # U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12}
  def R23(self,th):
      return np.matrix([
          [1, 0, 0],
          [0, cos(th), sin(th)],
          [0, -sin(th), cos(th)]
      ])

  def R13(self,th):
      return np.matrix([
          [cos(th), 0, sin(th)],
          [0, 1, 0],
          [-sin(th), 0, cos(th)]
      ])

  def R12(self,th):
      return np.matrix([
          [cos(th), sin(th), 0],
          [-sin(th), cos(th), 0],
          [0, 0, 1]
      ])

  def Delta(self,d):
      return np.matrix([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, exp(1j*d)]
      ])

  def __get_item__(self, i):
      return self.pmns[i]

  def transpose(self):
      return self.pmns.transpose()
  
  def conjugate(self):
      return self.pmns.conjugate()
