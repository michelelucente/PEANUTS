#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 7 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

import numpy as np
from numpy.linalg import multi_dot
from math import sin, cos, sqrt, pi
from scipy.integrate import complex_ode

from src.potentials import k, MatterPotential, R_E
from src.evolutor import FullEvolutor

def PnuenueEarth(density, pmns, DeltamSq21, DeltamSq31, eta, E, H):
  """
  PnuenueEarth(density, pmns, DeltamSq21, DeltamSq31, eta, E, H) computes
  numerically the probability of survival of an incident electron neutrino
  - density is the Earth density object
  - pmns is the PMNS matrix
  - DeltamSq21, Deltamq31 are the mass squared differences
  - eta is the nadir angle
  - E is the neutrino energy
  - H is the detector depth below the surface of the Earth
  """

  # Extract from pmns matrix
  U = pmns.U
  r23 = pmns.R23(pmns.theta23)
  delta = pmns.Delta(pmns.delta)
 
  Hk = multi_dot([U, np.diag(k(np.array([0, DeltamSq21, DeltamSq31]), E)), U.transpose()])

  h = H/R_E
  r_d = 1 - h
  x_d = sqrt(r_d**2 - sin(eta)**2)
  Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)
  n_1 = density(1-h/2,0)
  #n_1 = EarthDensity(x=1 - h/2) # TODO: Is eta=0 here? why?

  params = density.parameters(eta)
  x1, x2 = (-params[-1][1], x_d) if 0 <= eta < pi/2 else (0, Deltax)

  def model(t, y):
    nue, numu, nutau = y
    dnudt = - 1j * np.dot(multi_dot([r23, delta.conjugate(), Hk + np.diag([
        MatterPotential(density(t, eta)) if 0 <= eta < pi/2 else n_1
        ,0,0]), delta, r23.transpose()]), [nue, numu, nutau])
    return dnudt

  nu0 = (pmns.transpose()[1, :]).conjugate()

  nu = complex_ode(model)

  nu.set_integrator("Isoda")
  nu.set_initial_value(nu0, x1)


  x = np.linspace(x1, x2, 10**3)
  sol = [nu.integrate(xi) for xi in x[1::]]
  sol.insert(0, np.array(nu0)[0])

  return sol[-1]


def PnuenueEarth_analytical(density, pmns, DeltamSq21, DeltamSq31, eta, E, H):
  """
  PnuenueEarth_analytical(density, pmns, DeltamSq21, DeltamSq31, eta, E, H) computes
  analytically the probability of survival of an incident electron neutrino
  - density is the Earth density object
  - pmns is the PMNS matrix
  - DeltamSq21, Deltamq31 are the mass squared differences
  - eta is the nadir angle
  - E is the neutrino energy
  - H is the detector depth below the surface of the Earth
  """
  nu0 = (pmns.transpose()[1, :]).conjugate()

  return np.dot(FullEvolutor(density, 0, DeltamSq21, DeltamSq31, pmns, E, eta, H), nu0.transpose()).transpose()


