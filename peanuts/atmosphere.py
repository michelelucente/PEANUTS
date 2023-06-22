#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on June 21 2023

@author: Michele Lucente <michele.lucente@unibo.it>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import numpy as np
import numba as nb
from math import sqrt, cos, sin, pi, asin

from peanuts.evolutor import Upert
from peanuts.potentials import R_E

@nb.njit
def DL(eta, height):
    """
    DL(eta, height) compute lenght DL of trajectory from production point to Earth surface, normalized to Earth radius
    - eta: the nadir angle
    - height: the altitude production point of neutrinos, in meters above the Earth surface
    """

    r_surface = (R_E)/(R_E + height) # Relative radius of Earth surface relative to production sphere

    if eta >= 0 and eta <= pi/2:
        delta_relative = - r_surface * cos(eta) + sqrt(1 - r_surface**2 * sin(eta)**2) # lenght of path normalised to production radius
    elif eta > pi/2 and eta <= pi:
        delta_relative = r_surface * cos(eta) + sqrt(1 - r_surface**2 * sin(eta)**2) # lenght of path normalised to production radius

    return delta_relative * (R_E + height) / R_E # Returns the lenght normlised to Earth radius


@nb.njit
def evolved_state_atmosphere(nustate, DeltamSq21, DeltamSq3l, pmns, E, eta, height, depth=0, massbasis=True, antinu=False):
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

    # 3d identity matrix of complex numbers
    id3 = np.eye(3, dtype=nb.complex128)

    # Compute the factorised matrices R_{23} and \Delta
    # (remember that U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12})
    r23 = pmns.R23(pmns.theta23)
    delta = pmns.Delta(pmns.delta)

    # Conjugate for antineutrinos
    if antinu:
      r23 = r23.conjugate()
      delta = delta.conjugate()

    # If the detector is underneath the Earth, compute eta_prime
    if depth > 0:
      h = depth/R_E
      r_d = 1 - h
      eta_prime = asin(r_d * sin(eta))
    else:
      eta_prime = eta

    evolutor_atm = Upert(DeltamSq21, DeltamSq3l, pmns, E, DL(eta_prime,height), 0, 0, 0, 0, antinu) if height > 0 else id3
    evolutor = np.dot(np.dot(np.dot(r23, delta.conjugate()), np.dot(evolutor_atm , delta)), r23.transpose())

    if not massbasis:
      return np.dot(evolutor.transpose(), nustate.astype(nb.complex128))
    else:
      if not antinu:
        return np.dot(np.dot(evolutor.transpose(), pmns.pmns), nustate.astype(nb.complex128))
      else:
        return np.dot(np.dot(evolutor.transpose(), pmns.pmns.conjugate()), nustate.astype(nb.complex128))


@nb.njit
def Patmosphere(nustate, DeltamSq21, DeltamSq3l, pmns, E, eta, height, depth=0, massbasis=True, antinu=False):
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

    # 3d identity matrix of complex numbers
    id3 = np.eye(3, dtype=nb.complex128)

    # Compute the factorised matrices R_{23} and \Delta
    # (remember that U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12})
    r23 = pmns.R23(pmns.theta23)
    delta = pmns.Delta(pmns.delta)

    # Conjugate for antineutrinos
    if antinu:
      r23 = r23.conjugate()
      delta = delta.conjugate()

    # If the detector is underneath the Earth, compute eta_prime
    if depth > 0:
      h = depth/R_E
      r_d = 1 - h
      eta_prime = asin(r_d * sin(eta))
    else:
      eta_prime = eta

    evolutor_atm = Upert(DeltamSq21, DeltamSq3l, pmns, E, DL(eta_prime,height), 0, 0, 0, 0, antinu) if height > 0 else id3
    evolutor = np.dot(np.dot(np.dot(r23, delta.conjugate()), np.dot(evolutor_atm , delta)), r23.transpose())

    if not massbasis:
        return np.square(np.abs(np.dot(evolutor.transpose(), nustate.astype(nb.complex128))))
    elif massbasis:
        if not antinu:
            return np.real(np.dot(np.square(np.abs(np.dot(evolutor.transpose(), pmns.pmns)).astype(nb.complex128)), nustate.astype(nb.complex128)))
        else:
            return np.real(np.dot(np.square(np.abs(np.dot(evolutor.transpose(), pmns.pmns.conjugate())).astype(nb.complex128)), nustate.astype(nb.complex128)))
