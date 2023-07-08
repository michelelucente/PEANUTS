#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 21 2023

@author: Michele Lucente <michele.lucente@unibo.it>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import numpy as np
import numba as nb
from numba.experimental import jitclass
from math import sqrt, cos, sin, pi, asin

from peanuts.evolutor import Upert, ExponentialEvolution
from peanuts.potentials import R_E, R_S

# Atmosphere maximum height
Hmax = 2e4

atmospheredensity = [
   ('density_file', nb.types.string),
   ('rho0', nb.float64),
   ('H', nb.float64),
]

@jitclass(atmospheredensity)
class AtmosphereDensity:
  """
  Exponential profile of atmospheric density
  """

  def __init__(self, density_file=None):
    """
    Construct the exponential profiel of atmospheric density
    """

    if density_file is not None:
      print("Error: Custom atmospheric profiles are not supported yet")
      exit()

    else:

      # Mean atmospheric temperature, in K
      Tmean = 250 # K

      # Mean mass of one mol of atmospheric particles = 0.029 kg/mol for Earth
      Mmean = 0.029 # kg/mol

      # Boltzmann constant
      kB = 1.391e-23 # J / K = kg m^2 / s^2 / K

      # acceleration due to gravity (assumed constant)
      g = 9.8 # m/s^2

      # Avogadro number
      NA = 6.02214076e23 # mol^-1

      # Atmospheric density on surface (1.2 kg/m^3)
      self.rho0 = 1.2e-6 / Mmean # mol/cm^3

      # Scale height, in m
      self.H = NA * kB * Tmean / Mmean / g # m

  def call(self, h):
    """
    Computes the value of the density from the exponential profile
    for a value of h (in m), in units of mol/cm^3
    """

    return self.rho0 * np.exp(- h / self.H)

    # TODO: This is the atmospheric density, in principle we need the electron density


@nb.njit
def DL(eta, height):
    """
    DL(eta, height) compute length DL of trajectory from production point to Earth surface, normalized to Earth radius
    - eta: the nadir angle
    - height: the altitude production point of neutrinos, in meters above the Earth surface
    """

    r_surface = (R_E)/(R_E + height) # Relative radius of Earth surface relative to production sphere

    if eta >= 0 and eta <= pi/2:
        delta_relative = - r_surface * cos(eta) + sqrt(1 - r_surface**2 * sin(eta)**2) # length of path normalised to production radius
    elif eta > pi/2 and eta <= pi:
        delta_relative = r_surface * cos(eta) + sqrt(1 - r_surface**2 * sin(eta)**2) # length of path normalised to production radius

    return delta_relative * (R_E + height) / R_E # Returns the length normlised to Earth radius

class SolarDensityTemp:
    """
    Class containing the solar density"
    """

    def __init__(self):

      # Fill in the exponential profile of the density
      self.V0 = 0.0458e-3 # 1/m
      self.r0 = 0.1*R_S

    def call(self, r):
      """
      Return the the exponential profile density at radius r
      """

      return self.V0*np.exp(-r/self.r0)


def evolved_state_atmosphere(nustate, density, DeltamSq21, DeltamSq3l, pmns, E, eta, height, depth=0, massbasis=True, antinu=False):
    """
    evolved_state_atmosphere() computes the evolved neutrino state on the surface of the Earth produced at some height in the atmosphere:
    - nustate: incoming neutrino state
    - DeltamSq21: the solar mass splitting
    - DeltamSq3l: the atmospheric mass splitting (l=1 for NO, l=2 for IO)
    - pmns: the PMNS matrix
    - E: the neutrino energy, in units of MeV;
    - eta: the nadir angle;
    - height: the altitude production point of neutrinos, in meters above the Earth surface
    - depth: depth of the detector, default 0
    - massbasis: the basis of the neutrino eigenstate, True: mass, False: flavour (def. True)
    - antinu: False for neutrinos, True for antineutrinos
    """

    # If the detector is underneath the Earth, compute eta_prime
    if depth > 0:
      h = depth/R_E
      r_d = 1 - h
      eta_prime = asin(r_d * sin(eta))
    else:
      eta_prime = eta

    # If initial state is given as a mass eigenstate, compute the flavour eigenstate
    if massbasis:
      if not antinu:
        initialstate = np.dot(pmns.transpose(),nustate)
      else:
        initialstate = np.dot(pmns.conjugatetranpose(),nustate)
    else:
      initialstate = nustate

    # Compute the evolved state in the flavour basis
    evolved_state = ExponentialEvolution(initialstate, density, DeltamSq21, DeltamSq3l, pmns, E, R_E*DL(eta_prime, height), 0, antinu=False)

    return evolved_state


def Patmosphere(nustate, density, DeltamSq21, DeltamSq3l, pmns, E, eta, height, depth=0, massbasis=True, antinu=False):
    """
    Patmosphere() computes the probability of survival of a neutrino state on the surface of the Earth produced at some height in the atmosphere:
    - nustate: incoming neutrino state
    - DeltamSq21: the solar mass splitting
    - DeltamSq3l: the atmospheric mass splitting (l=1 for NO, l=2 for IO)
    - pmns: the PMNS matrix
    - E: the neutrino energy, in units of MeV;
    - eta: the nadir angle;
    - height: the altitude production point of neutrinos, in meters above the Earth surface
    - depth: depth of the detector, default 0
    - massbasis: the basis of the neutrino eigenstate, True: mass, False: flavour (def. True)
    - antinu: False for neutrinos, True for antineutrinos
    """

    # If the detector is underneath the Earth, compute eta_prime
    if depth > 0:
      h = depth/R_E
      r_d = 1 - h
      eta_prime = asin(r_d * sin(eta))
    else:
      eta_prime = eta

    # If initial state is given as a mass eigenstate, compute the flavour eigenstate
    if massbasis:
      if not antinu:
        initialstate = np.dot(pmns.transpose(),nustate)
      else:
        initialstate = np.dot(pmns.conjugatetranpose(),nustate)
    else:
      initialstate = nustate

    # Compute the evolved state in the flavour basis
    evolved_state = ExponentialEvolution(initialstate, density, DeltamSq21, DeltamSq3l, pmns, E, R_E*DL(eta_prime, height), 0)

    return np.square(np.abs(evolved_state))
