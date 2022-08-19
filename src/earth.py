#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 10 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import os
import numpy as np
from numpy.linalg import multi_dot
from math import sin, cos, sqrt, pi, asin
from scipy.integrate import complex_ode

import src.files as f
from src.potentials import k, MatterPotential, R_E
from src.evolutor import FullEvolutor


class EarthDensity:
  """
  See hep-ph/9702343 for the definition of trajectory coordinate and Earth density parametrisation.
  """

  def __init__(self, density_file=None):
    """
    Read the Earth density parametrisation, in units of mol/cm^3, following hep-ph/9702343
    """

    # TODO: This is specific for the example file, make general
    path = os.path.dirname(os.path.realpath( __file__ ))
    self.density_file = path + "/../Data/Earth_Density.csv" if density_file == None else density_file
    earth_density = f.read_csv(self.density_file, names=["rj", "alpha", "beta", "gamma"], skiprows=3)

    self.rj = earth_density.rj.to_numpy()
    self.alpha = earth_density.alpha.to_numpy()
    self.beta = earth_density.beta.to_numpy()
    self.gamma = earth_density.gamma.to_numpy()

  def parameters(self, eta):
    """
    Returns the values of the density parameters for the shells crossed by the path
    with nadir angle = eta as a list of lists, where each element [[a, b, c], x_i] refers to a
    Earth shell  (from inner to outer layers) having density profile n_e(x) = a + b x^2 + c x^4,
    with shell external boundary at x == x_i.
    """

    # Select the index "idx_shells" in rj such that for i >= idx_shells => rj[i] > sin(eta)
    # The shells having rj[i] > sin(eta) are the ones crossed by a path with nadir angle = eta
    idx_shells = np.searchsorted(self.rj, sin(eta))

    # Keep only the parameters for the shells crossed by the path with nadir angle eta
    alpha_prime = self.alpha[idx_shells::] + self.beta[idx_shells::] * sin(eta)**2 + self.gamma[idx_shells::] * sin(eta)**4
    beta_prime = self.beta[idx_shells::] + 2 * self.gamma[idx_shells::] * sin(eta)**2
    gamma_prime = self.gamma[idx_shells::]

    # Compute the value of the trajectory coordinates xj at each shell crossing
    xj = np.sqrt( (self.rj[idx_shells::])**2 - sin(eta)**2 )

    return [ [ [alpha_prime[i], beta_prime[i], gamma_prime[i]], xj[i] ] for i in range(len(alpha_prime))]

  def shells(self):
    """
    Returns the value of the radius for each shell
    """
    return self.rj

  def shells_eta(self):
    """
    Returns the value of the nadir angles corresponding to each shell
    """
    return np.arcsin(self.rj)/pi


  def __call__(self, x, eta):
    """
    Computes the value of Earth electron density in units of mol/cm^3 for trajectory coordinate
    x and nadir angle eta;
    """

    # The density profile is symmetric with respect to x=0
    x = np.abs(x)

    # If x > cos(eta) the trajectory coordinate is beyond Earth surface, thus density is zero.
    if x > cos(eta):
      return 0

    # Get the parameters
    param = self.parameters(eta)
    alpha_prime = [x[0][0] for x in param]
    beta_prime = [x[0][1] for x in param]
    gamma_prime = [x[0][2] for x in param]
    xj = [x[1] for x in param]

    # The index "idx" determines within which shell xj[idx] the point x is
    idx = np.searchsorted(xj, x)

    return alpha_prime[idx] + beta_prime[idx] * x**2 + gamma_prime[idx] * x**4


def Pearth_numerical(nustate, density, pmns, DeltamSq21, DeltamSq31, E, eta, H, basis="flavour", full_oscillation=False):
  """
  Pearth_numerical(nustate, density, pmns, DeltamSq21, DeltamSq31, E, eta, H) computes
  numerically the probability of survival of an incident electron neutrino spectrum
  - nustate is the array of weights of the incoherent neutrino flux
  - density is the Earth density object
  - pmns is the PMNS matrix
  - DeltamSq21, Deltamq31 are the mass squared differences
  - E is the neutrino energy
  - eta is the nadir angle
  - H is the detector depth below the surface of the Earth
  """

  # Extract from pmns matrix
  U = pmns.U
  r23 = pmns.R23(pmns.theta23)
  delta = pmns.Delta(pmns.delta)

  Hk = multi_dot([U, np.diag(k(np.array([0, DeltamSq21, DeltamSq31]), E)), U.transpose()])

  h = H/R_E
  r_d = 1 - h
  x_d = r_d * cos(eta)
  Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)
  n_1 = density(1-h/2,0)
  eta_prime = asin(r_d * sin(eta))

  params = density.parameters(eta_prime)
  x1, x2 = (-params[-1][1], x_d) if 0 <= eta < pi/2 else (0, Deltax)

  def model(t, y):
    nue, numu, nutau = y
    dnudt = - 1j * np.dot(multi_dot([r23, delta.conjugate(), Hk + np.diag([
        MatterPotential(density(t, eta=eta_prime) if 0 <= eta < pi/2 else n_1)
        ,0,0]), delta, r23.transpose()]), [nue, numu, nutau])
    return dnudt

  num_evol = []

  x = np.linspace(x1, x2, 10**3)

  successful_integration = True

  for col in range(3):
    if successful_integration:
        nu0 = np.identity(3)[:, col]

        nu = complex_ode(model)

        nu.set_integrator("Isoda")
        nu.set_initial_value(nu0, x1)

        sol = [nu.integrate(xi) for xi in x[1::]]

        successful_integration = successful_integration and nu.successful()

        sol.insert(0, np.array(nu0))

        num_evol.append(sol)

  if len(num_evol) < 3:
    print("Error: numerical integration failed")
    exit()

  num_solution = [np.column_stack((num_evol[0][k], num_evol[1][k], num_evol[2][k])) for k in range(len(x))]

  if basis == "flavour":
      evolution = [np.array(np.square(np.abs(np.dot(num_solution[i].transpose(), nustate))) ) for i in range(len(x))]
  elif basis == "mass":
      evolution = [np.array(np.dot(np.square(np.abs(np.dot(num_solution[i].transpose(), pmns.pmns))),nustate))[0] for i in range(len(x))]
  else:
      print("Error: unrecognised neutrino basis, please choose either \"flavour\" or \"mass\".")
      exit()

  if full_oscillation:
    return evolution, x
  else:
    return evolution[-1]



def Pearth_analytical(nustate, density, pmns, DeltamSq21, DeltamSq31, E, eta, H, basis="flavour"):
  """
  Pearth_analytical(nustate, density, pmns, DeltamSq21, DeltamSq31, E, eta, H) computes
  analytically the probability of survival of an incident electron neutrino spectrum
  - nustate is the array of weights of the incoherent neutrino flux
  - density is the Earth density object
  - pmns is the PMNS matrix
  - DeltamSq21, Deltamq31 are the mass squared differences
  - E is the neutrino energy
  - eta is the nadir angle
  - H is the detector depth below the surface of the Earth
  """

  evol = FullEvolutor(density, 0, DeltamSq21, DeltamSq31, pmns, E, eta, H)
  if basis == "flavour":
      return np.array(np.square(np.abs(np.dot(evol.transpose(), nustate))))
  elif basis == "mass":
      return np.array(np.dot(np.square(np.abs(np.dot(evol.transpose(), pmns.pmns))),nustate))

  else:
      print("Error: unrecognised neutrino basis, please choose either \"flavour\" or \"mass\".")
      exit()


def Pearth(nustate, density, pmns, DeltamSq21, DeltamSq31, E, eta, H, mode="analytical", basis="flavour", full_oscillation=False):
  """
  Pearth(nustate, density, pmns, DeltamSq21, DeltamSq21, E, eta, H), computes with a given mode
  the probability of survival of an incident electron neutrino spectrum
  - nustate is the array of weights of the incoherent neutrino flux
  - density is the Earth density object
  - pmns is the PMNS matrix
  - DeltamSq21, Deltamq31 are the mass squared differences
  - E is the neutrino energy
  - eta is the nadir angle
  - H is the detector depth below the surface of the Earth
  """

  # Make sure nustate has the write format
  if len(nustate) != 3:
    print("Error: neutrino state provided has the wrong format, it must be a vector of size 3.")
    exit()
  #if np.abs(np.sum(np.square(np.abs(nustate))) - 1) > 1e-3:
  #  print("Error: neutrino state provided has the wrong format, it elements square must sum to 1.")
  #  exit()

  if mode == "analytical":
    return Pearth_analytical(nustate, density, pmns, DeltamSq21, DeltamSq31, E, eta, H, basis=basis)

  elif mode == "numerical":
    return Pearth_numerical(nustate, density, pmns, DeltamSq21, DeltamSq31, E, eta, H, basis=basis, full_oscillation=full_oscillation)

  else:
    raise Exception("Error: Unkown mode for the computation of evoulutor")

