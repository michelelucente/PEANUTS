#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 23 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import time
import numpy as np
import numba as nb
from math import sin, asin, cos, sqrt, pi
from cmath import exp

from src.potentials import k, MatterPotential, R_E
from src.integration import c0, c1, lambdas, Iab

@nb.njit
def Upert (DeltamSq21, DeltamSq3l, pmns, E, x2, x1, a, b, c):
    """
    Upert(DeltamSq21, DeltamSq3l, pmns, E,  x2, x1, a, b, c) computes the evolutor
    for an ultrarelativistic neutrino state in flavour basis, for a reduced mixing matrix U = R_{13} R_{12}
    (the dependence on th_{23} and CP-violating phase \delta_{CP} can be factorised) for a density profile
    parametrised by a 4th degree even poliomial in the trajectory coordinate, to 1st order corrections around
    the mean density value:
    - DeltamSq21: the solar mass splitting
    - DeltamSq3l: the atmospheric mass splitting (l=1 for NO, l=2 for IO)
    - pmns is the PMNS matrix;
    - E is the neutrino energy, in units of MeV;
    - x1 (x2) is the starting (ending) point in the path;
    - a, b, c parametrise the density profile on the path, n_e(x) = a + b x^2 + c x^4.
    See hep-ph/9702343 for the definition of the perturbative expansion of the evolutor in a 2-flavours case.
    """

    # 3d identity matrix of complex numbers
    id3 = np.eye(3, dtype=nb.complex128)

    # Kinetic terms of the Hamiltonian
    if DeltamSq3l > 0: # NO, l = 1
      ki = k(np.array([0, DeltamSq21, DeltamSq3l], dtype=nb.complex128), E)
    else: # IO, l = 2
      ki = k(np.array([-DeltamSq21, 0, DeltamSq3l], dtype=nb.complex128), E)

    # Average matter density along the path
    naverage = (a * (x2 - x1) + b * (x2**3 - x1**3)/3 + c * (x2**5 - x1**5)/5) / (x2 - x1)

    # Matter potential for the 0th order evolutor
    V = MatterPotential(naverage)

    # Parameter for the density perturbation around the mean density value:
    atilde = a - naverage

    # Travelled distance
    L = (x2 - x1)

    # Reduced mixing matrix U = R_{13} R_{12}
    U = pmns.U

    # Hamiltonian in the reduced flavour basis
    H = np.dot(np.dot(U, np.diag(ki)), U.transpose()) + np.diag(np.array([V, 0, 0]))

    # Traceless Hamiltonian T = H - Tr(H)/3
    tr = np.sum(ki) + V
    T = H - tr/3 * id3

    # Coefficients of the characteristic equation for T
    c0_loc = c0(ki, pmns.theta12, pmns.theta13, naverage)
    c1_loc = c1(ki, pmns.theta12, pmns.theta13, naverage)

    # Roots of the characteristic equation for T
    lam = lambdas(c0_loc, c1_loc)

    # Matrices M_a, not depending on x
    M = np.zeros((len(lam),3,3), dtype=nb.complex128)
    for i in nb.prange(len(lam)):
      M[i] = (1 / (3*lam[i]**2 + c1_loc)) * ((lam[i]**2 + c1_loc) * id3 + lam[i] * T + np.dot(T,T))

    # 0th order evolutor (i.e. for constant matter density), following Eq. (46) in hep-ph/9910546
    u0 = np.zeros((3,3), dtype=nb.complex128)
    u1 = np.zeros((3,3), dtype=nb.complex128)
    for i in range(len(lam)):
      u0 += np.exp(-1j * (lam[i] + tr/3) * L) * M[i]

    # Compute correction to evolutor, taking into account 1st order terms in \delta n_e(x)
    if (b != 0) | (c != 0):
      for idx_a in range(3) :
        for idx_b in range(3) :
          u1 += np.dot(np.dot(M[idx_a], np.diag(np.array([-1j * MatterPotential(Iab(lam[idx_a] + tr/3, lam[idx_b] + tr/3, atilde, b, c, x2, x1)), 0, 0]))), M[idx_b])

    u = u0 + u1

    # Return the full evolutor
    return u


@nb.njit
def FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, H):
    """
    FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, H) computes the full evolutor for an ultrarelativistic
    neutrino crossing the Earth:
    - density is the Earth density object
    - DeltamSq21: the solar mass splitting
    - DeltamSq3l: the atmospheric mass splitting (l=1 for NO, l=2 for IO)
    - pmns is the PMNS matrix
    - E is the neutrino energy, in units of MeV;
    - d is the CP-violating PMNS phase;
    - eta is the nadir angle;
    - H is the underground detector depth, in units of meters.
    """

    # If the detector is on the surface and neutrinos are coming from above the horizon, there is no
    # matter effect
    if H == 0 and (pi/2 <= eta <= pi):
        return (1+0.j)*np.identity(3)

    # Detector depth normalised to Earth radius
    h = H / R_E

    # Position of detector the on a radial path
    r_d = 1 - h # This is valid for eta = 0

    # Compute the factorised matrices R_{23} and \Delta
    # (remember that U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12})
    r23 = pmns.R23(pmns.theta23)
    delta = pmns.Delta(pmns.delta)

    # If 0 <= eta < pi/2 we compute the evolutor taking care of matter density perturbation around the
    # density mean value at first order
    if 0 <= eta < pi/2:
        # Nadir angle if the detector was on surface
        eta_prime = asin(r_d * sin(eta))

        # Position of the detector along the trajectory coordinate
        # x_d = sqrt(r_d**2 - sin(eta)**2) -- wrong old definition
        x_d = r_d * cos(eta)

        # params is a list of lists, each element [a, b, c, x_i] contains the parameters of the density
        # profile n_e(x) = a + b x^2 + c x^4 along the crossed shell, with each shell ending at x == x_i
        params = density.parameters(eta_prime)
        params2 = np.flipud(params)

        # Compute the evolutors for the path from Earth entry point to trajectory mid-point at x == 0
        evolutors_full_path = [Upert(DeltamSq21, DeltamSq3l,pmns, E, params2[i][3], params2[i+1][3] if i < len(params2)-1 else 0, params2[i][0], params2[i][1], params2[i][2]) for i in range(len(params))]

        # Multiply the single evolutors
        evolutor_half_full = evolutors_full_path[0]
        for i in range(len(evolutors_full_path)-1):
          evolutor_half_full = np.dot(evolutor_half_full, evolutors_full_path[i+1])

        # Compute the evolutors for the path from the trajectory mid-point at x == 0 to the detector point x_d
        # Only the evolutor for the most external shell needs to be computed
        evolutors_to_detectors = evolutors_full_path.copy()

        evolutors_to_detectors[0] = Upert(DeltamSq21, DeltamSq3l, pmns, E, x_d, params[-2][3] if len(params) > 1 else 0, params[-1][0], params[-1][1], params[-1][2])

        # Multiply the single evolutors
        evolutor_half_detector = evolutors_to_detectors[0]
        for i in range(len(evolutors_to_detectors)-1):
          evolutor_half_detector = np.dot(evolutor_half_detector, evolutors_to_detectors[i+1])

        # Combine the two half-paths evolutors and include the factorised dependence on th23 and d to
        # obtain the full evolutor
        evolutor =  np.dot(np.dot(np.dot(r23, delta.conjugate()), np.dot(evolutor_half_detector, evolutor_half_full.transpose())), np.dot(delta, r23.transpose()))
        return evolutor

    # If pi/2 <= eta <= pi we approximate the density to the constant value taken at r = 1 - h/2
    elif pi/2 <= eta <= pi:

        #n_1 = EarthDensity(x = 1 - h / 2) TODO: eta = 0?
        n_1 = density.call(1 - h/2, 0)

        # Deltax is the lenght of the crossed path
        Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)

        # Compute the evolutor for constant density n_1 and traveled distance Deltax,
        # and include the factorised dependence on th23 and d to obtain the full evolutor
        evolutor = np.dot(np.dot(np.dot(r23, delta.conjugate()), np.dot(Upert(DeltamSq21, DeltamSq3l, pmns, E, Deltax, 0, n_1, 0, 0), delta)), r23.transpose())
        return evolutor

    else:
        raise ValueError('eta must be comprised between 0 and pi.')
