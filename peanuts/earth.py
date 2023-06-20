#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 10 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import os
import time
import numpy as np
import numba as nb
from numba.experimental import jitclass
from numpy.linalg import multi_dot
from math import sin, cos, sqrt, pi, asin
from scipy.integrate import complex_ode

import peanuts.files as f
from peanuts.potentials import k, MatterPotential, R_E
from peanuts.evolutor import FullEvolutor
from peanuts.time_average import NadirExposure

earthdensity =  [
  ('density_file', nb.types.string),
  ('rj', nb.float64[:]),
  ('alpha', nb.float64[:]),
  ('beta', nb.float64[:]),
  ('gamma', nb.float64[:])
]

@jitclass(earthdensity)
class EarthDensity:
  """
  See hep-ph/9702343 for the definition of trajectory coordinate and Earth density parametrisation.
  """

  def __init__(self, density_file=None):
    """
    Read the Earth density parametrisation, in units of mol/cm^3, following hep-ph/9702343
    """

    # TODO: This is specific for the example file, make general
    with nb.objmode(density_file='string', rj='float64[:]', alpha='float64[:]', beta='float64[:]', gamma='float64[:]'):
      path = os.path.dirname(os.path.realpath( __file__ ))
      density_file = path + "/../Data/Earth_Density.csv" if density_file == None else density_file
      earth_density = f.read_csv(density_file, names=["rj", "alpha", "beta", "gamma"], skiprows=3)

      rj = earth_density.rj.to_numpy()
      alpha = earth_density.alpha.to_numpy()
      beta = earth_density.beta.to_numpy()
      gamma = earth_density.gamma.to_numpy()

    self.density_file = density_file
    self.rj = rj
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma

  def parameters(self, eta):
    """
    Returns the values of the density parameters for the shells crossed by the path
    with nadir angle = eta as a list of lists, where each element [a, b, c, x_i] refers to a
    Earth shell  (from inner to outer layers) having density profile n_e(x) = a + b x^2 + c x^4,
    with shell external boundary at x == x_i.
    """

    # Select the index "idx_shells" in rj such that for i >= idx_shells => rj[i] > sin(eta)
    # The shells having rj[i] > sin(eta) are the ones crossed by a path with nadir angle = eta
    idx_shells = np.searchsorted(self.rj, sin(eta))

    # Keep only the parameters for the shells crossed by the path with nadir angle eta
    alpha_prime = self.alpha[idx_shells::] + self.beta[idx_shells::] * sin(eta)**2 + self.gamma[idx_shells::] * np.sin(eta)**4
    beta_prime = self.beta[idx_shells::] + 2 * self.gamma[idx_shells::] * sin(eta)**2
    gamma_prime = self.gamma[idx_shells::]

    # Compute the value of the trajectory coordinates xj at each shell crossing
    xj = np.sqrt( (self.rj[idx_shells::])**2 - sin(eta)**2 )

    return np.stack((alpha_prime, beta_prime, gamma_prime, xj), axis=1)

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


  def call(self, x, eta):
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
    alpha_prime = [y[0] for y in param]
    beta_prime = [y[1] for y in param]
    gamma_prime = [y[2] for y in param]
    xj = [y[3] for y in param]

    # The index "idx" determines within which shell xj[idx] the point x is
    idx = np.searchsorted(xj, x)

    return alpha_prime[idx] + beta_prime[idx] * x**2 + gamma_prime[idx] * x**4


def numerical_solution(density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height, antinu):
  """
  numerical_solution(density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu) computes
  numerically the probability of survival of an incident electron neutrino spectrum
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy, in units of MeV;
  - eta: the nadir angle
  - depth: the detector depth below the surface of the Earth, in meters
  - height: the altitudde production point of neutrios, in meters above the Earth surface
  - antinu: False for neutrinos, True for antineutrinos
  """

  # TODO: Missing height implementation in numerical solution

  # Extract from pmns matrix
  U = pmns.U
  r23 = pmns.R23(pmns.theta23)
  delta = pmns.Delta(pmns.delta)

  # Conjugate for antineutrinos
  if antinu:
    U = U.conjugate()
    r23 = r23.conjugate()
    delta = delta.conjugate()

  if DeltamSq3l > 0: # NO, l = 1
    ki = k(np.array([0, DeltamSq21, DeltamSq3l]), E)
  else: # IO, l = 2
    ki = k(np.array([-DeltamSq21, 0, DeltamSq3l]), E)
  Hk = multi_dot([U, np.diag(ki), U.transpose()])

  h = depth/R_E
  r_d = 1 - h
  x_d = r_d * cos(eta)
  Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)
  n_1 = density.call(1-h/2,0)
  eta_prime = asin(r_d * sin(eta))

  params = density.parameters(eta_prime)
  x1, x2 = (-params[-1][3], x_d) if 0 <= eta < pi/2 else (0, Deltax)

  def model(t, y):
    nue, numu, nutau = y
    dnudt = - 1j * np.dot(multi_dot([r23, delta.conjugate(), Hk + np.diag([
        MatterPotential(density.call(t, eta_prime) if 0 <= eta < pi/2 else n_1, antinu)
        ,0,0]), delta, r23.transpose()]), [nue, numu, nutau])
    return dnudt

  num_evol = []

  x = np.linspace(x1, x2, 10**3)

  successful_integration = True

  for col in range(3):
    if successful_integration:
        nu0 = np.identity(3)[:, col]

        nu = complex_ode(model)

        nu.set_integrator("lsoda")
        nu.set_initial_value(nu0, x1)

        sol = [nu.integrate(xi) for xi in x[1::]]

        successful_integration = successful_integration and nu.successful()

        sol.insert(0, np.array(nu0))

        num_evol.append(sol)

  if len(num_evol) < 3:
    print("Error: numerical integration failed")
    exit()

  num_solution = [np.column_stack((num_evol[0][k], num_evol[1][k], num_evol[2][k])) for k in range(len(x))]

  return num_solution, x


def evolved_state_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=0, full_oscillation=False, antinu=False):
  """
  evolved_state_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, full_oscillation, antinu) computes
  numerically the probability of survival of an incident electron neutrino spectrum
  - nustate: the array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E is the neutrino energy, in units of MeV;
  - eta: the nadir angle
  - depth: detector depth below the surface of the Earth, in meters
  - height: the altitudde production point of neutrios, in meters above the Earth surface
  - full_oscillation: return full oscillation along path (def. False))
  - antinu: False for neutrinos, True for antineutrinos
  """

  num_solution, x = numerical_solution(density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height, antinu)

  state = [np.array(np.dot(num_solution[i].transpose(), nustate)) for i in range(len(x))]

  if full_oscillation:
    return state, x
  else:
    return state[-1]

def Pearth_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=0, massbasis=True, full_oscillation=False, antinu=False):
  """
  Pearth_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, massbasis, full_oscillation, antinu) computes
  numerically the probability of survival of an incident electron neutrino spectrum
  - nustate: the array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E is the neutrino energy, in units of MeV;
  - eta: the nadir angle
  - depth: the detector depth below the surface of the Earth, in meters
  - height: the altitudde production point of neutrios, in meters above the Earth surface
  - massbasis: the basis of the neutrino eigenstate, True: mass, False: flavour (def. True)
  - full_oscillation: return full oscillation along path (def. False))
  - antinu: False for neutrinos, True for antineutrinos
  """

  num_solution, x = numerical_solution(density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height, antinu)

  if not massbasis:
      evolution = [np.array(np.square(np.abs(np.dot(num_solution[i].transpose(), nustate))) ) for i in range(len(x))]
  elif massbasis:
      if not antinu:
        evolution = [np.array(np.real(np.dot(np.square(np.abs(np.dot(num_solution[i].transpose(), pmns.pmns))), nustate))) for i in range(len(x))]
      else:
        evolution = [np.array(np.real(np.dot(np.square(np.abs(np.dot(num_solution[i].transpose(), pmns.pmns.conjugate()))), nustate))) for i in range(len(x))]
  else:
      print("Error: unrecognised neutrino basis, please choose either \"flavour\" or \"mass\".")
      exit()

  if full_oscillation:
    return evolution, x
  else:
    return evolution[-1]


@nb.njit
def evolved_state_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=0, antinu=False):
  """
  evolved_state_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu) computes
  analytically the evolved flavour state after matter regeneration
  - nustate: the array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy, in units of MeV;
  - eta: the nadir angle
  - height: the altitude production point of neutrinos, in meters above the Earth surface
  - depth: the detector depth below the surface of the Earth, in meters
  - antinu: False for neutrinos, True for antineutrinos
  """

  evol = FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, depth, height, antinu)
  return np.dot(evol.transpose(), nustate.astype(nb.complex128))


@nb.njit
def Pearth_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=0, massbasis=True, antinu=False):
  """
  Pearth_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, massbasis, antinu) computes
  analytically the probability of survival of an incident electron neutrino spectrum
  - nustate: the array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy, in units of MeV;
  - eta: the nadir angle
  - depth: the detector depth below the surface of the Earth, in meters
  - height: the altitude production point of neutrinos, in meters above the Earth surface
  - massbasis: the basis of the neutrino eigenstate, True: mass, False: flavour (def. True)
  - antinu: False for neutrinos, True for antineutrinos
  """
  evol = FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, depth, height, antinu)
  if not massbasis:
      return np.square(np.abs(np.dot(evol.transpose(), nustate.astype(nb.complex128))))
  elif massbasis:
      if not antinu:
        return np.real(np.dot(np.square(np.abs(np.dot(evol.transpose(), pmns.pmns)).astype(nb.complex128)), nustate.astype(nb.complex128)))
      else:
        return np.real(np.dot(np.square(np.abs(np.dot(evol.transpose(), pmns.pmns.conjugate())).astype(nb.complex128)), nustate.astype(nb.complex128)))

  else:
      raise Exception("Error: unrecognised neutrino basis, please choose either \"flavour\" or \"mass\".")


def evolved_state(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=0, mode="analytical", full_oscillation=False, antinu=False):
  """
  evolved_state(nustate, density, pmns, DeltamSq21, DeltamSq21, E, eta, depth, mode, full_oscillation, antinu), computes with a given mode
  the probability of survival of an incident electron neutrino spectrum
  - nustate: array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy, in units of MeV;
  - eta: the nadir angle
  - depth:  the detector depth below the surface of the Earth, in meters
  - height: the altitude production point of neutrinos, in meters above the Earth surface
  - mode: either analytical or numerical computation of the evolutor (def. analytical)
  - full_oscillation: return full oscillation along path (def. False))
  - antinu: False for neutrinos, True for antineutrinos
  """

  # Make sure nustate has the write format
  if len(nustate) != 3:
    print("Error: neutrino state provided has the wrong format, it must be a vector of size 3.")
    exit()

  if mode == "analytical":
    if full_oscillation:
      print("Warning: full oscillation only available in numerical mode. Result will be only final probability values")

    return evolved_state_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=height, antinu=antinu)

  elif mode == "numerical":
    # TODO: Missing implementation of height for numerical solution
    return evolved_state_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, full_oscillation=full_oscillation, antinu=antinu)

  else:
    raise Exception("Error: Unkown mode for the computation of evoulutor")



def Pearth(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=0, mode="analytical", massbasis=True, full_oscillation=False, antinu=False):
  """
  Pearth(nustate, density, pmns, DeltamSq21, DeltamSq21, E, eta, depth, mode, massbasis, full_oscillation, antinu), computes with a given mode
  the probability of survival of an incident electron neutrino spectrum
  - nustate: array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy, in units of MeV;
  - eta: the nadir angle
  - depth:  the detector depth below the surface of the Earth, in meters
  - height: the altitude production point of neutrinos, in meters above the Earth surface
  - mode: either analytical or numerical computation of the evolutor (def. analytical)
  - massbasis: the basis of the neutrino eigenstate, True: mass, False: flavour (def. True)
  - full_oscillation: return full oscillation along path (def. False))
  - antinu: False for neutrinos, True for antineutrinos
  """

  # Make sure nustate has the write format
  if len(nustate) != 3:
    print("Error: neutrino state provided has the wrong format, it must be a vector of size 3.")
    exit()
  #if np.abs(np.sum(np.square(np.abs(nustate))) - 1) > 1e-3:
  #  print("Error: neutrino state provided has the wrong format, it elements square must sum to 1.")
  #  exit()

  if mode == "analytical":
    if full_oscillation:
      print("Warning: full oscillation only available in numerical mode. Result will be only final probability values")

    return Pearth_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=height, massbasis=massbasis, antinu=antinu)

  elif mode == "numerical":
    return Pearth_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=height, massbasis=massbasis, full_oscillation=full_oscillation, antinu=antinu)

  else:
    raise Exception("Error: Unkown mode for the computation of evoulutor")


def Pearth_integrated(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, depth, height=0, mode="analytical", full_oscillation=False, antinu=False, lam=-1, d1=0, d2=365, ns=1000, normalized=False, from_file=None, angle="Nadir",daynight=None):
  """
  Pearth(nustate, density, pmns, DeltamSq21, DeltamSq21, E, lam, depth, mode, full_oscillation, antinu, d1, d2, ns, normalized, from_file, angle),
  computes the probability of survival of an incident electron neutrino spectrum integrated over the spectrum
  - nustate: array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy, in units of MeV;
  - depth:  the detector depth below the surface of the Earth, in meters
  - height: the altitude production point of neutrinos, in meters above the Earth surface
  - lam: the latitude of the experiment (def. -1)
  - mode: either analytical or numerical computation of the evolutor (def. analytical)
  - full_oscillation: return full oscillation along path (def. False))
  - antinu: False for neutrinos, True for antineutrinos
  - d1: lower limit of day interval
  - d2: upper limit of day interval
  - ns: number of nadir angle samples
  - normalized: normalization of exposure
  - from_file: file with experiments exposure
  - angle: angle of samples is exposure file
  """

  exposure = NadirExposure(lam=lam, normalized=normalized, d1=d1, d2=d2, ns=ns, from_file=from_file, angle=angle)

  day = True if daynight != "night" else False
  night = True if daynight != "day" else False


  prob = 0
  deta = pi/ns
  for eta, exp in exposure:
    if eta < pi/2 and night:
      prob += Pearth(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=height, mode=mode, massbasis=True, full_oscillation=full_oscillation, antinu=antinu) * exp * deta
    elif eta >= pi/2 and day:
      prob += Pearth(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, height=height, mode=mode, massbasis=True, full_oscillation=full_oscillation, antinu=antinu) * exp * deta


  return prob
