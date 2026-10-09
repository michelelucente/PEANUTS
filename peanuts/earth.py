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
from math import sin, cos, sqrt, pi, asin, floor, copysign
from scipy.integrate import complex_ode

import peanuts.files as f
from peanuts.potentials import k, MatterPotential, R_E
from peanuts.evolutor import FullEvolutor, evolutor_setup, crossing_evolutor, average_density, spectral_decomposition, evolutor_from_spectrum
from peanuts.time_average import NadirExposure

@nb.njit
def binom(n, k):
  """
  Returns the binomial number (n, k)
  """

  if k==0 or k==n:
    return 1

  return np.prod(np.array([(n + 1 - i)/i for i in range(1,k+1)]))

earthdensity =  [
  ('density_file', nb.types.string),
  ('use_custom_density', nb.boolean),
  ('use_tabulated_density', nb.boolean),
  ('rj', nb.float64[:]),
  ('alpha', nb.float64[:]),
  ('beta', nb.float64[:]),
  ('gamma', nb.float64[:]),
  ('deltas', nb.float64[:,:])
]

@jitclass(earthdensity)
class EarthDensity:
  """
  See hep-ph/9702343 for the definition of trajectory coordinate and Earth density parametrisation.
  """

  def __init__(self, density_file=None, tabulated_density=False, custom_density=False):
    """
    Read the Earth density parametrisation, in units of mol/cm^3, following hep-ph/9702343, if
    provided in a file (density_file), where it must consist of a table for each layer with the
    radius as the first entry and the rest the coefficients of the density as
    n_e(r) = alpha + beta r^2 + gamma r^4 + delta_n r^2n, with n>3
    If instead, the numerical flag is on, the density value will be taken from some user-provided
    function and the parameters extracted from a polynomial expansion
    """

    self.use_tabulated_density = tabulated_density

    if custom_density:
      self.use_custom_density = True

      # For simplicity create just one shell
      self.rj = np.ones(1)

    else:

      with nb.objmode(density_file='string', rj='float64[:]', alpha='float64[:]', beta='float64[:]', gamma='float64[:]', deltas='float64[:,:]'):
        path = os.path.dirname(os.path.realpath( __file__ ))

        # The default density file is for parametric density, not tabulated
        if density_file == None and tabulated_density:
          print("Error: tabulated density was selected, but no density file was provided.")
          exit()

        density_file = path + "/../Data/Earth_Density.csv" if density_file == None else density_file

        # Parse the density file to get starting row and columns
        skiprows, columns, sep  = f.parse_csv(density_file)

        earth_density = f.read_csv(density_file, names=columns, skiprows=skiprows, sep=sep)

        rj = earth_density.rj.to_numpy()
        alpha = earth_density.alpha.to_numpy()

        # If the density is tabulated we treat it as multiple layers with constant density alpha
        # So higher order coefficients are zero
        if tabulated_density:
          beta = np.zeros((len(rj)))
          gamma = np.zeros((len(rj)))
          deltas = np.zeros((0,len(rj)))
        else:
          beta = earth_density.beta.to_numpy() if "beta" in columns else np.zeros((len(rj)))
          gamma = earth_density.gamma.to_numpy() if "gamma" in columns else np.zeros((len(rj)))
          deltas = np.empty((len(columns)-4,len(rj)))
          for col in range(len(columns)-4):
            deltas[col] = getattr(earth_density,"delta"+str(col+1)).to_numpy()

      self.density_file = density_file
      self.rj = rj
      self.alpha = alpha
      self.beta = beta
      self.gamma = gamma
      self.deltas = deltas

  def parameters(self, eta):
    """
    Returns the values of the density parameters for the shells crossed by the path
    with nadir angle = eta as a list of lists, where each element [a, b, c, x_i] refers to a
    Earth shell  (from inner to outer layers) having density profile
    n_e(x) = alpha' + beta' x^2 + gamma' x^4 + delta_n' x^2n, with n>3
    with shell external boundary at x == x_i.
    """

    if not self.use_custom_density:

      # Make copy of class parameters to make sure no memory is touched
      alpha = self.alpha.copy()
      beta = self.beta.copy()
      gamma = self.gamma.copy()
      deltas = self.deltas.copy()

    else:
      # If using a custom density profile, we still need alpha, beta and gamma for the analytical computation
      # So approximate them using a Taylor expansion. We do not need the deltas.
      h = 0.001
      alpha = np.array([self.custom_density(0)])
      beta  = np.array([(self.custom_density(2*h) + self.custom_density(0) - 2*self.custom_density(h))/(2*h**2)])
      gamma = np.array([(self.custom_density(4*h) + self.custom_density(0) + 6*self.custom_density(2*h) - 4*self.custom_density(3*h) - 4*self.custom_density(h))/(24*h**4)])
      deltas = np.empty((0,1))

    # Select the index "idx_shells" in rj such that for i >= idx_shells => rj[i] > sin(eta)
    # The shells having rj[i] > sin(eta) are the ones crossed by a path with nadir angle = eta
    idx_shells = np.searchsorted(self.rj, sin(eta))

    # Keep only the parameters for the shells crossed by the path with nadir angle eta
    alpha_prime = alpha[idx_shells::] + beta[idx_shells::] * np.sin(eta)**2 + gamma[idx_shells::] * np.sin(eta)**4
    beta_prime = beta[idx_shells::] + 2 * gamma[idx_shells::] * np.sin(eta)**2
    gamma_prime = gamma[idx_shells::]

    # Add contribution from deltas
    deltas_prime = deltas[:,idx_shells::]
    for n in range(len(deltas)):
      alpha_prime += deltas[n,idx_shells::] * np.sin(eta)**(2*(n+3))
      beta_prime += (n+3) * deltas[n,idx_shells::] * np.sin(eta)**(2*(n+3-1))
      gamma_prime += binom(n+3, 2) * deltas[n,idx_shells::] * np.sin(eta)**(2*(n+3-2))
      for k in range(n+1,len(self.deltas)):
        deltas_prime[n] += binom(k+3,n+3) * deltas[k,idx_shells::] * np.sin(eta)**(2*(k-n))

    # Compute the value of the trajectory coordinates xj at each shell crossing
    xj = np.sqrt( (self.rj[idx_shells::])**2 - sin(eta)**2 )

    result = np.zeros((len(deltas_prime)+4,len(xj)))
    result[0] = alpha_prime
    result[1] = beta_prime
    result[2] = gamma_prime
    for i in range(len(deltas_prime)):
      result[3+i] = deltas_prime[i]
    result[-1] = xj
    result = result.transpose()

    return result



  def shells(self):
    """
    Returns the value of the radius for each shell
    """
    return self.rj

  def shells_x(self, eta):
    """
    Returns the value of the path coordinate for each shell
    crossed by the neutrino path
    """

    # Select the index "idx_shells" in rj such that for i >= idx_shells => rj[i] > sin(eta)
    # The shells having rj[i] > sin(eta) are the ones crossed by a path with nadir angle = eta
    idx_shells = np.searchsorted(self.rj, sin(eta))

    # Compute the value of the trajectory coordinates xj at each shell crossing
    xj = np.sqrt( (self.rj[idx_shells::])**2 - sin(eta)**2 )

    return xj

  def shells_eta(self):
    """
    Returns the value of the nadir angles corresponding to each shell
    """
    return np.arcsin(self.rj)/pi

  def custom_density(self, r):
    """
    Placeholder function to implement custom density functions
    """

    # Replace this example with a custom density function
    a = 6.1
    b = -1.7
    c = 2.6
    return a + b * r**2 + c * r**4

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

    # Use custom density profile
    if self.use_custom_density:
      # Switch to radial coordinate to evaluate density
      r = np.sqrt(x**2 + np.sin(eta)**2)
      return self.custom_density(r)

    # Get the parameters
    param = self.parameters(eta)
    alpha_prime = param[:,0]
    beta_prime = param[:,1]
    gamma_prime = param[:,2]
    deltas_prime = np.transpose(param[:,3:-1])
    xj = param[:,-1]

    # The index "idx" determines within which shell xj[idx] the point x is
    idx = np.searchsorted(xj, x)

    return alpha_prime[idx] + beta_prime[idx] * x**2 + gamma_prime[idx] * x**4 + np.sum(np.array([deltas_prime[i][idx] * x**(6+2*i) for i in range(len(deltas_prime))]))


def numerical_solution(density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu):
  """
  numerical_solution(density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu) computes
  numerically the probability of survival of an incident electron neutrino spectrum
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy
  - eta: the nadir angle
  - depth: the detector depth below the surface of the Earth
  - antinu: False for neutrinos, True for antineutrinos
  """

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

  x1, x2 = (-density.shells_x(eta_prime)[-1], x_d) if 0 <= eta < pi/2 else (0, Deltax)

  def model(t, y):
    nue, numu, nutau = y
    dnudt = - 1j * np.dot(multi_dot([r23, delta, Hk + np.diag([
        MatterPotential(density.call(t, eta_prime) if 0 <= eta < pi/2 else n_1, antinu)
        ,0,0]), delta.conjugate().transpose(), r23.transpose()]), [nue, numu, nutau])
    return dnudt

  num_evol = []

  if E<10:
    x = np.linspace(x1, x2, floor(10**(3+np.log10(10/E))))
  else:
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


def evolved_state_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, full_oscillation=False, antinu=False):
  """
  evolved_state_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, full_oscillation, antinu) computes
  numerically the probability of survival of an incident electron neutrino spectrum
  - nustate: the array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy
  - eta: the nadir angle
  - depth: detector depth below the surface of the Earth
  - full_oscillation: return full oscillation along path (def. False))
  - antinu: False for neutrinos, True for antineutrinos
  """

  num_solution, x = numerical_solution(density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu)

  state = [np.array(np.dot(num_solution[i], nustate)) for i in range(len(x))]

  if full_oscillation:
    return state, x
  else:
    return state[-1]

def Pearth_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, massbasis=True, full_oscillation=False, antinu=False):
  """
  Pearth_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, massbasis, full_oscillation, antinu) computes
  numerically the probability of survival of an incident electron neutrino spectrum
  - nustate: the array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy
  - eta: the nadir angle
  - depth: the detector depth below the surface of the Earth
  - massbasis: the basis of the neutrino eigenstate, True: mass, False: flavour (def. True)
  - full_oscillation: return full oscillation along path (def. False))
  - antinu: False for neutrinos, True for antineutrinos
  """

  num_solution, x = numerical_solution(density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu)

  if not massbasis:
      evolution = [np.array(np.square(np.abs(np.dot(num_solution[i], nustate))) ) for i in range(len(x))]
  elif massbasis:
      if not antinu:
        evolution = [np.array(np.real(np.dot(np.square(np.abs(np.dot(num_solution[i], pmns.pmns))), nustate))) for i in range(len(x))]
      else:
        evolution = [np.array(np.real(np.dot(np.square(np.abs(np.dot(num_solution[i], pmns.pmns.conjugate()))), nustate))) for i in range(len(x))]
  else:
      print("Error: unrecognised neutrino basis, please choose either \"flavour\" or \"mass\".")
      exit()

  if full_oscillation:
    return evolution, x
  else:
    return evolution[-1]


@nb.njit
def evolved_state_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu=False):
  """
  evolved_state_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu) computes
  analytically the evolved flavour state after matter regeneration
  - nustate: the array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy
  - eta: the nadir angle
  - depth: the detector depth below the surface of the Earth
  - antinu: False for neutrinos, True for antineutrinos
  """

  evol = FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, depth, antinu)
  return np.dot(evol, nustate.astype(nb.complex128))


@nb.njit
def Pearth_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, massbasis=True, antinu=False):
  """
  Pearth_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, massbasis, antinu) computes
  analytically the probability of survival of an incident electron neutrino spectrum
  - nustate: the array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy
  - eta: the nadir angle
  - depth: the detector depth below the surface of the Earth
  - massbasis: the basis of the neutrino eigenstate, True: mass, False: flavour (def. True)
  - antinu: False for neutrinos, True for antineutrinos
  """

  evol = FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, depth, antinu)
  if not massbasis:
      return np.square(np.abs(np.dot(evol, nustate.astype(nb.complex128))))
  elif massbasis:
      if not antinu:
        return np.real(np.dot(np.square(np.abs(np.dot(evol, pmns.pmns)).astype(nb.complex128)), nustate.astype(nb.complex128)))
      else:
        return np.real(np.dot(np.square(np.abs(np.dot(evol, pmns.pmns.conjugate())).astype(nb.complex128)), nustate.astype(nb.complex128)))

  else:
      raise Exception("Error: unrecognised neutrino basis, please choose either \"flavour\" or \"mass\".")


def evolved_state(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, mode="analytical", full_oscillation=False, antinu=False):
  """
  evolved_state(nustate, density, pmns, DeltamSq21, DeltamSq21, E, eta, depth, mode, full_oscillation, antinu), computes with a given mode
  the probability of survival of an incident electron neutrino spectrum
  - nustate: array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy
  - eta: the nadir angle
  - depth:  the detector depth below the surface of the Earth
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

    return evolved_state_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu=antinu)

  elif mode == "numerical":
    return evolved_state_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, full_oscillation=full_oscillation, antinu=antinu)

  else:
    raise Exception("Error: Unkown mode for the computation of evoulutor")



def Pearth(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, mode="analytical", massbasis=True, full_oscillation=False, antinu=False):
  """
  Pearth(nustate, density, pmns, DeltamSq21, DeltamSq21, E, eta, depth, mode, massbasis, full_oscillation, antinu), computes with a given mode
  the probability of survival of an incident electron neutrino spectrum
  - nustate: array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy
  - eta: the nadir angle
  - depth:  the detector depth below the surface of the Earth
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

    return Pearth_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, massbasis=massbasis, antinu=antinu)

  elif mode == "numerical":
    return Pearth_numerical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, massbasis=massbasis, full_oscillation=full_oscillation, antinu=antinu)

  else:
    raise Exception("Error: Unkown mode for the computation of evoulutor")

def Pearth_integrated(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, depth, mode="analytical", full_oscillation=False, antinu=False, lam=-1, d1=0, d2=365, ns=1000, normalized=False, from_file=None, angle="Nadir",daynight=None):
  """
  Pearth(nustate, density, pmns, DeltamSq21, DeltamSq21, E, lam, depth, mode, full_oscillation, antinu, d1, d2, ns, normalized, from_file, angle),
  computes the probability of survival of an incident electron neutrino spectrum integrated over the spectrum
  - nustate: array of weights of the incoherent neutrino flux
  - density: the Earth density object
  - pmns: the PMNS matrix
  - DeltamSq21, Deltamq3l: the mass squared differences
  - E: the neutrino energy
  - depth:  the detector depth below the surface of the Earth
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

  In analytical mode the sum over nadir angles is evaluated by compiled kernels, with the exposure,
  the path geometry and the mass-state projectors cached (see integrated_projectors). The result is
  identical to the sum of Pearth over the exposure samples computed by Pearth_integrated_reference.
  """

  if mode != "analytical" or full_oscillation or not isinstance(nustate, np.ndarray) or nustate.ndim != 1:
    return Pearth_integrated_reference(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, depth, mode=mode, full_oscillation=full_oscillation, antinu=antinu, lam=lam, d1=d1, d2=d2, ns=ns, normalized=normalized, from_file=from_file, angle=angle, daynight=daynight)

  exposure = cached_exposure(lam=lam, normalized=normalized, d1=d1, d2=d2, ns=ns, from_file=from_file, angle=angle, daynight=daynight)

  if len(exposure.etas) == 0:
    return 0

  # Make sure nustate has the write format
  if len(nustate) != 3:
    print("Error: neutrino state provided has the wrong format, it must be a vector of size 3.")
    exit()

  deta = pi/ns
  projectors = integrated_projectors(density, pmns, DeltamSq21, DeltamSq3l, E, depth, antinu, exposure)

  return integrate_projectors(projectors, exposure.used, exposure.weights, nustate, deta)


def Pearth_integrated_reference(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, depth, mode="analytical", full_oscillation=False, antinu=False, lam=-1, d1=0, d2=365, ns=1000, normalized=False, from_file=None, angle="Nadir",daynight=None):
  """
  Pearth_integrated_reference(...) is the direct sum of Pearth over the exposure samples, with the
  same arguments as Pearth_integrated. It is used for the numerical mode and as the reference of
  the compiled path.
  """

  exposure = NadirExposure(lam=lam, normalized=normalized, d1=d1, d2=d2, ns=ns, from_file=from_file, angle=angle, daynight=daynight)

  day = True if daynight != "night" else False
  night = True if daynight != "day" else False


  prob = 0
  deta = pi/ns
  for eta, exp in exposure:
    prob += Pearth(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, mode=mode, massbasis=True, full_oscillation=full_oscillation, antinu=antinu) * exp * deta


  return prob


class ExposureSamples:
  """
  Nadir exposure samples as returned by NadirExposure, split into angles and weights, with the
  mask of the samples that carry a non-zero weight (a zero weight contributes exactly 0 to the
  sum over nadir angles, so its probability is not computed).
  """

  def __init__(self, exposure):
    exposure = np.asarray(exposure, dtype=np.float64)
    self.etas = np.ascontiguousarray(exposure[:,0]) if len(exposure) else np.zeros(0)
    self.weights = np.ascontiguousarray(exposure[:,1]) if len(exposure) else np.zeros(0)
    self.used = self.weights != 0
    self.geometry = {}


_exposure_cache = {}

def cached_exposure(lam=-1, d1=0, d2=365, ns=1000, normalized=False, from_file=None, angle="Nadir", daynight=None):
  """
  cached_exposure(...) returns the ExposureSamples of NadirExposure with the same arguments. The
  exposure depends on the experiment only, so it is computed once; an exposure file is identified
  by its path, size and modification time.
  """

  stamp = None
  if from_file is not None and os.path.isfile(from_file):
    st = os.stat(from_file)
    stamp = (st.st_size, st.st_mtime_ns)
  key = (lam, d1, d2, ns, normalized, from_file, stamp, angle, daynight)
  if key not in _exposure_cache:
    _exposure_cache[key] = ExposureSamples(NadirExposure(lam=lam, normalized=normalized, d1=d1, d2=d2, ns=ns, from_file=from_file, angle=angle, daynight=daynight))
  return _exposure_cache[key]


@nb.njit
def path_geometry(density, etas, used, depth):
  """
  path_geometry(density, etas, used, depth) computes the energy-independent part of FullEvolutor
  for every used nadir angle: the kind of path (0: detector on the surface and neutrino from above,
  1: crossing the Earth, 2: from above the horizon, -1: invalid angle), the detector coordinate x_d
  and the parameters (a, b, c, x_i) of the crossed shells for kind 1, the path length for kind 2,
  and the density n_1 used for kind 2.
  """

  n = len(etas)
  h = depth / R_E
  r_d = 1 - h
  maxsh = len(density.rj)

  kind = np.zeros(n, dtype=nb.int64)
  nsh = np.zeros(n, dtype=nb.int64)
  xd = np.zeros(n)
  dx = np.zeros(n)
  a = np.zeros((n, maxsh))
  b = np.zeros((n, maxsh))
  c = np.zeros((n, maxsh))
  xs = np.zeros((n, maxsh))
  n_1 = 0.
  above = False

  for i in range(n):
    if not used[i]:
      continue
    eta = etas[i]
    if depth == 0 and (pi/2 <= eta <= pi):
      kind[i] = 0
    elif 0 <= eta < pi/2:
      kind[i] = 1
      eta_prime = asin(r_d * sin(eta))
      xd[i] = r_d * cos(eta)
      params = density.parameters(eta_prime)
      xshells = density.shells_x(eta_prime)
      nsh[i] = len(xshells)
      for j in range(len(xshells)):
        a[i,j] = params[j,0]
        b[i,j] = params[j,1]
        c[i,j] = params[j,2]
        xs[i,j] = xshells[j]
    elif pi/2 <= eta <= pi:
      kind[i] = 2
      dx[i] = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)
      above = True
    else:
      kind[i] = -1

  if above:
    n_1 = density.call(1 - h/2, 0)

  return kind, nsh, xd, dx, a, b, c, xs, n_1


@nb.njit
def mass_projectors(pmns, DeltamSq21, DeltamSq3l, E, antinu, used, kind, nsh, xd, dx, a, b, c, xs, n_1):
  """
  mass_projectors(...) returns, for every used nadir angle, the matrix |S U_{PMNS}|^2 (as complex
  numbers, the form used by Pearth_analytical) where S is the FullEvolutor at that angle; the
  probability of a mass-state mixture nustate is real(|S U|^2 . nustate). Paths from above the
  horizon at depth > 0 differ only by their length, so their spectral decomposition is shared by
  all the angles with the same average density (compared bit by bit).
  """

  n = len(kind)
  out = np.zeros((n,3,3), dtype=nb.complex128)

  ki, Hk, r23, r23delta, deltac, right = evolutor_setup(pmns, DeltamSq21, DeltamSq3l, E, antinu)
  th12 = pmns.theta12
  th13 = pmns.theta13
  pm = pmns.pmns
  if antinu:
    pm = pmns.pmns.conjugate()

  # Spectral decompositions already computed for paths from above the horizon, by average density
  ncache = 8
  nc = 0
  cache_nav = np.zeros(ncache)
  cache_lam = np.zeros((ncache,3), dtype=nb.complex128)
  cache_M = np.zeros((ncache,3,3,3), dtype=nb.complex128)
  cache_tr = np.zeros(ncache, dtype=nb.complex128)

  for i in range(n):
    if not used[i]:
      continue

    if kind[i] == 0:
      evol = (1+0.j)*np.identity(3)

    elif kind[i] == 1:
      m = nsh[i]
      evol = crossing_evolutor(ki, Hk, th12, th13, r23delta, right, xd[i], a[i,:m], b[i,:m], c[i,:m], xs[i,:m], antinu)

    elif kind[i] == 2:
      naverage = average_density(dx[i], 0, n_1, 0, 0)
      atilde = n_1 - naverage
      j = -1
      for q in range(nc):
        if cache_nav[q] == naverage and copysign(1., cache_nav[q]) == copysign(1., naverage):
          j = q
          break
      if j >= 0:
        lam = cache_lam[j].copy()
        M = cache_M[j].copy()
        tr = cache_tr[j]
      else:
        lam, M, tr = spectral_decomposition(ki, Hk, th12, th13, naverage, antinu)
        if nc < ncache:
          cache_nav[nc] = naverage
          cache_lam[nc] = lam
          cache_M[nc] = M
          cache_tr[nc] = tr
          nc += 1
      u = evolutor_from_spectrum(lam, M, tr, dx[i], 0, atilde, 0, 0, antinu)
      evol = np.dot(np.dot(r23delta, np.dot(u, deltac.transpose())), r23.transpose())

    else:
      raise ValueError('eta must be comprised between 0 and pi.')

    out[i] = np.square(np.abs(np.dot(evol, pm)).astype(nb.complex128))

  return out


@nb.njit
def integrate_projectors(projectors, used, weights, nustate, deta):
  """
  integrate_projectors(projectors, used, weights, nustate, deta) returns the sum over the used
  nadir angles of real(projector . nustate) * weight * deta, in the order of the angles.
  """

  prob = np.zeros(3)
  nus = nustate.astype(nb.complex128)
  for i in range(len(used)):
    if used[i]:
      prob += np.real(np.dot(projectors[i], nus)) * weights[i] * deta
  return prob


_projector_cache = [None, None]

def integrated_projectors(density, pmns, DeltamSq21, DeltamSq3l, E, depth, antinu, exposure):
  """
  integrated_projectors(...) returns mass_projectors for the exposure samples. The path geometry
  is cached per density profile and depth; the projectors of the last call are kept, so that
  several neutrino states at the same energy and parameters (e.g. the three mass states) share
  one evaluation of the evolutors.
  """

  pmns_mat = pmns.pmns
  U = pmns.U
  key = (pmns.theta12, pmns.theta13, pmns.theta23, pmns.delta, pmns_mat.tobytes(), U.tobytes(),
         DeltamSq21, DeltamSq3l, E, depth, antinu)
  last_key, last = _projector_cache
  if last is not None and last_key == key and last[0] is density and last[1] is exposure:
    return last[2]

  geometry = exposure.geometry.get(depth)
  if geometry is None or geometry[0] is not density:
    geometry = (density, path_geometry(density, exposure.etas, exposure.used, depth))
    exposure.geometry[depth] = geometry
  kind, nsh, xd, dx, a, b, c, xs, n_1 = geometry[1]

  projectors = mass_projectors(pmns, DeltamSq21, DeltamSq3l, E, antinu, exposure.used, kind, nsh, xd, dx, a, b, c, xs, n_1)
  _projector_cache[0] = key
  _projector_cache[1] = (density, exposure, projectors)
  return projectors
