#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 18 2023

@author: Michele Lucente <michele.lucente@unibo.it>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import numpy as np
from peanuts.evolutor import Upert
from peanuts.potentials import R_E

def vacuum_evolved_state(nustate, pmns, DeltamSq21, DeltamSq3l, E, L, antinu=False):
    """
    evolved_state_analytical(nustate, density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, antinu) computes
    analytically the evolved flavour state after matter regeneration
    - nustate: the array of weights of the incoherent neutrino flux
    - density: the Earth density object
    - pmns: the PMNS matrix
    - DeltamSq21, Deltamq3l: the mass squared differences
    - L: baseline, in km
    - antinu: False for neutrinos, True for antineutrinos
    """

    # Use dimensionless distances
    l = L*1e3 / R_E

    # Compute the factorised matrices R_{23} and \Delta
    # (remember that U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12})
    r23 = pmns.R23(pmns.theta23)
    delta = pmns.Delta(pmns.delta)

    # Conjuagate for antineutrinos
    if antinu:
      r23 = r23.conjugate()
      delta = delta.conjugate()

    # Compute the vacuum evolutor
    evol = np.dot(np.dot(np.dot(r23, delta.conjugate()), np.dot(Upert(DeltamSq21, DeltamSq3l, pmns, E, 0, l, 0, 0, 0, antinu), delta)), r23.transpose())

    return np.dot(evol, nustate)

def Pvacuum(nustate, pmns, DeltamSq21, DeltamSq3l, E, L, antinu=False, massbasis=True):
    """
    Probability of oscillating in vacuum
    - pmns is the PMNS matrix
    - DeltamSq21: the solar mass splitting
    - DeltamSq3l: the atmospheric mass splitting (l=1 for NO, l=2 for IO)
    - E: the neutrino energy, in units of MeV
    - L: baseline, in km
    - antinu: False for neutrinos, True for antineutrinos
    - massbasis: True for mass basis, False for flavour basis
    """

    # Use dimensionless distances
    l = L*1e3 / R_E

    # Compute the factorised matrices R_{23} and \Delta
    # (remember that U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12})
    r23 = pmns.R23(pmns.theta23)
    delta = pmns.Delta(pmns.delta)

    # Conjuagate for antineutrinos
    if antinu:
      r23 = r23.conjugate()
      delta = delta.conjugate()

    # Compute the vacuum evolutor
    evol = np.dot(np.dot(np.dot(r23, delta.conjugate()), np.dot(Upert(DeltamSq21, DeltamSq3l, pmns, E, l, 0, 0, 0, 0, antinu), delta)), r23.transpose())

    if not massbasis:
      return np.square(np.abs(np.dot(evol, nustate)))
    elif massbasis:
      if not antinu:
        return np.real(np.dot(np.square(np.abs(np.dot(evol, pmns.pmns))), nustate))
      else:
        return np.real(np.dot(np.square(np.abs(np.dot(evol, pmns.pmns.conjugate())).astype(nb.complex128)), nustate.astype(nb.complex128)))
