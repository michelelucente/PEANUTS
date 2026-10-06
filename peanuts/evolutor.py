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

from peanuts.potentials import k, MatterPotential, R_E
from peanuts.integration import c0, c1, lambdas, Iab


@nb.njit
def kinetic_terms(DeltamSq21, DeltamSq3l, E):
    """
    kinetic_terms(DeltamSq21, DeltamSq3l, E) returns the dimensionless kinetic terms k_i of the
    Hamiltonian, for normal (DeltamSq3l > 0, l = 1) or inverted (l = 2) ordering.
    """

    if DeltamSq3l > 0: # NO, l = 1
      return k(np.array([0, DeltamSq21, DeltamSq3l], dtype=nb.complex128), E)
    else: # IO, l = 2
      return k(np.array([-DeltamSq21, 0, DeltamSq3l], dtype=nb.complex128), E)


@nb.njit
def kinetic_hamiltonian(U, ki):
    """
    kinetic_hamiltonian(U, ki) returns the kinetic part of the Hamiltonian in the reduced flavour
    basis, U diag(k_i) U^T. It depends on the energy and on the mixing only, not on the matter
    density, so it can be computed once and shared by all the shells and paths at a given energy.
    """

    return np.dot(np.dot(U, np.diag(ki)), U.transpose())


@nb.njit
def average_density(x2, x1, a, b, c):
    """
    average_density(x2, x1, a, b, c) returns the average of n_e(x) = a + b x^2 + c x^4 on [x1, x2].
    """

    return (a * (x2 - x1) + b * (x2**3 - x1**3)/3 + c * (x2**5 - x1**5)/5) / (x2 - x1)


@nb.njit
def spectral_decomposition(ki, Hk, th12, th13, naverage, antinu):
    """
    spectral_decomposition(ki, Hk, th12, th13, naverage, antinu) returns the eigenvalues lam of the
    traceless Hamiltonian T = H - Tr(H)/3 for the constant density naverage, the matrices M_a of
    Eq. (46) in hep-ph/9910546 and the trace Tr(H). They depend on the path only through naverage.
    """

    # The element-wise matrix expressions below are written as loops over the matrix elements, with
    # the same scalar operations on the same operands as the array expressions
    #   H = Hk + diag(V, 0, 0),  T = H - Tr(H)/3 * 1,
    #   M_a = (1 / (3 lam_a^2 + c1)) * ((lam_a^2 + c1) * 1 + lam_a T + T^2),
    # so that no temporary arrays are allocated; the matrix product T^2 is left to BLAS.

    # Matter potential for the 0th order evolutor
    V = MatterPotential(naverage, antinu)

    # Hamiltonian in the reduced flavour basis and its trace
    tr = np.sum(ki) + V
    tr3 = tr/3

    # Traceless Hamiltonian T = H - Tr(H)/3
    T = np.empty((3,3), dtype=nb.complex128)
    for j in range(3):
      for k in range(3):
        Hjk = Hk[j,k] + (V if j == 0 and k == 0 else 0.)
        T[j,k] = Hjk - tr3 * ((1.+0.j) if j == k else 0.j)

    # Coefficients of the characteristic equation for T
    c0_loc = c0(ki, th12, th13, naverage, antinu)
    c1_loc = c1(ki, th12, th13, naverage, antinu)

    # Roots of the characteristic equation for T
    lam = lambdas(c0_loc, c1_loc)

    # Matrices M_a, not depending on x (T^2 does not depend on a either, so it is computed once)
    TT = np.dot(T,T)
    M = np.empty((len(lam),3,3), dtype=nb.complex128)
    for i in range(len(lam)):
      s1 = (1 / (3*lam[i]**2 + c1_loc))
      s2 = (lam[i]**2 + c1_loc)
      for j in range(3):
        for k in range(3):
          M[i,j,k] = s1 * ((s2 * ((1.+0.j) if j == k else 0.j) + lam[i] * T[j,k]) + TT[j,k])

    return lam, M, tr


@nb.njit
def evolutor_from_spectrum(lam, M, tr, x2, x1, atilde, b, c, antinu):
    """
    evolutor_from_spectrum(lam, M, tr, x2, x1, atilde, b, c, antinu) assembles the evolutor of Upert
    from the output of spectral_decomposition.
    """

    # Travelled distance
    L = (x2 - x1)

    # 0th order evolutor (i.e. for constant matter density), following Eq. (46) in hep-ph/9910546
    # (u0 += exp(-i (lam_a + Tr(H)/3) L) M_a, element by element)
    u0 = np.zeros((3,3), dtype=nb.complex128)
    u1 = np.zeros((3,3), dtype=nb.complex128)
    for i in range(len(lam)):
      phase = np.exp(-1j * (lam[i] + tr/3) * L)
      for j in range(3):
        for k in range(3):
          u0[j,k] += phase * M[i,j,k]

    # Compute correction to evolutor, taking into account 1st order terms in \delta n_e(x):
    # u1 = sum_{a != b} M_a diag(d_ab, 0, 0) M_b, with d_ab = -i V(I_ab). The matrix in the middle has
    # rank one, so each term is the outer product of the first column of M_a, times d_ab, and the
    # first row of M_b. The terms with a == b vanish identically (Iab returns 0 for la == lb).
    if (b != 0) | (c != 0):
      for idx_a in range(3) :
        for idx_b in range(3) :
          if idx_a != idx_b:
            d = -1j * MatterPotential(Iab(lam[idx_a] + tr/3, lam[idx_b] + tr/3, atilde, b, c, x2, x1), antinu)
            for i in range(3):
              Mad = M[idx_a,i,0] * d
              for j in range(3):
                u1[i,j] += Mad * M[idx_b,0,j]

    u = u0 + u1

    # Return the full evolutor
    return u


@nb.njit
def Upert_kinetic (ki, Hk, th12, th13, x2, x1, a, b, c, antinu):
    """
    Upert_kinetic(ki, Hk, th12, th13, x2, x1, a, b, c, antinu) is Upert with the kinetic terms ki and
    the kinetic Hamiltonian Hk (see kinetic_terms and kinetic_hamiltonian) supplied by the caller.
    """

    # Average matter density along the path
    naverage = average_density(x2, x1, a, b, c)

    # Parameter for the density perturbation around the mean density value:
    atilde = a - naverage

    lam, M, tr = spectral_decomposition(ki, Hk, th12, th13, naverage, antinu)

    return evolutor_from_spectrum(lam, M, tr, x2, x1, atilde, b, c, antinu)


@nb.njit
def Upert (DeltamSq21, DeltamSq3l, pmns, E, x2, x1, a, b, c, antinu):
    """
    Upert(DeltamSq21, DeltamSq3l, pmns, E,  x2, x1, a, b, c, antinu) computes the evolutor
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
    - antinu: False for neutrinos, True for antineutrinos
    See hep-ph/9702343 for the definition of the perturbative expansion of the evolutor in a 2-flavours case.
    """

    # Kinetic terms of the Hamiltonian
    ki = kinetic_terms(DeltamSq21, DeltamSq3l, E)

    # Reduced mixing matrix U = R_{13} R_{12}
    U = pmns.U
    if antinu:
      U = U.conjugate()

    # Kinetic Hamiltonian in the reduced flavour basis
    Hk = kinetic_hamiltonian(U, ki)

    return Upert_kinetic(ki, Hk, pmns.theta12, pmns.theta13, x2, x1, a, b, c, antinu)


@nb.njit
def evolutor_setup(pmns, DeltamSq21, DeltamSq3l, E, antinu):
    """
    evolutor_setup(pmns, DeltamSq21, DeltamSq3l, E, antinu) returns the quantities of FullEvolutor
    that depend on the energy and on the mixing but not on the path: the kinetic terms ki, the
    kinetic Hamiltonian Hk, R_{23}, the product R_{23} \Delta, \Delta^* and the product
    (\Delta^*)^T R_{23}^T (remember that U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12}).
    """

    # Kinetic terms of the Hamiltonian and kinetic Hamiltonian in the reduced flavour basis
    ki = kinetic_terms(DeltamSq21, DeltamSq3l, E)
    U = pmns.U
    if antinu:
      U = U.conjugate()
    Hk = kinetic_hamiltonian(U, ki)

    # Compute the factorised matrices R_{23} and \Delta
    r23 = pmns.R23(pmns.theta23)
    delta = pmns.Delta(pmns.delta)

    # Conjuagate for antineutrinos
    if antinu:
      r23 = r23.conjugate()
      delta = delta.conjugate()

    r23delta = np.dot(r23, delta)
    deltac = delta.conjugate()
    right = np.dot(deltac.transpose(), r23.transpose())

    return ki, Hk, r23, r23delta, deltac, right


@nb.njit
def crossing_evolutor(ki, Hk, th12, th13, r23delta, right, x_d, a, b, c, xshells, antinu):
    """
    crossing_evolutor(ki, Hk, th12, th13, r23delta, right, x_d, a, b, c, xshells, antinu) computes the
    full evolutor for a path crossing the Earth (0 <= eta < pi/2), given the output of evolutor_setup
    and, for each crossed shell from inner to outer, the parameters a, b, c of the density profile
    n_e(x) = a + b x^2 + c x^4 along the path and the shell end coordinate xshells.
    """

    nsh = len(xshells)

    # The matrix products are written into preallocated buffers (np.dot(a, b, out) is the BLAS
    # call that np.dot(a, b) makes after allocating its result)
    evolutors_full_path = np.empty((nsh,3,3), dtype=nb.complex128)
    buf = np.empty((2,3,3), dtype=nb.complex128)

    # Compute the evolutors for the path from Earth entry point to trajectory mid-point at x == 0
    for i in range(nsh):
      evolutors_full_path[i] = Upert_kinetic(ki, Hk, th12, th13, xshells[nsh-1-i], xshells[nsh-2-i] if i < nsh-1 else 0, a[nsh-1-i], b[nsh-1-i], c[nsh-1-i], antinu)

    # Multiply the single evolutors
    evolutor_half_full = evolutors_full_path[0]
    for i in range(nsh-1):
      np.dot(evolutor_half_full, evolutors_full_path[i+1], buf[i % 2])
      evolutor_half_full = buf[i % 2]
    evolutor_half_full = evolutor_half_full.copy()

    # Compute the evolutors for the path from the trajectory mid-point at x == 0 to the detector point x_d
    # Only the evolutor for the most external shell needs to be computed
    evolutor_half_detector = Upert_kinetic(ki, Hk, th12, th13, x_d, xshells[-2] if nsh > 1 else 0, a[-1], b[-1], c[-1], antinu)

    # Multiply the single evolutors
    for i in range(nsh-1):
      np.dot(evolutor_half_detector, evolutors_full_path[i+1], buf[i % 2])
      evolutor_half_detector = buf[i % 2]

    # Combine the two half-paths evolutors and include the factorised dependence on th23 and d to
    # obtain the full evolutor
    return np.dot(np.dot(r23delta, np.dot(evolutor_half_detector, evolutor_half_full.transpose())), right)


@nb.njit
def FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, depth, antinu):
    """
    FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, depth, antinu) computes the full evolutor for an ultrarelativistic
    neutrino crossing the Earth:
    - density is the Earth density object
    - DeltamSq21: the solar mass splitting
    - DeltamSq3l: the atmospheric mass splitting (l=1 for NO, l=2 for IO)
    - pmns is the PMNS matrix
    - E is the neutrino energy, in units of MeV;
    - d is the CP-violating PMNS phase;
    - eta is the nadir angle;
    - depth is the underground detector depth, in units of meters.
    - antinu: False for neutrinos, True for antineutrinos
    """

    # If the detector is on the surface and neutrinos are coming from above the horizon, there is no
    # matter effect
    if depth == 0 and (pi/2 <= eta <= pi):
        return (1+0.j)*np.identity(3)

    # Detector depth normalised to Earth radius
    h = depth / R_E

    # Position of detector the on a radial path
    r_d = 1 - h # This is valid for eta = 0

    # Energy and mixing dependent quantities, common to every shell
    ki, Hk, r23, r23delta, deltac, right = evolutor_setup(pmns, DeltamSq21, DeltamSq3l, E, antinu)

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

        # xshells contains the end coordinate x == x_i for each crossed earth density shell
        xshells = density.shells_x(eta_prime)

        return crossing_evolutor(ki, Hk, pmns.theta12, pmns.theta13, r23delta, right, x_d, params[:,0], params[:,1], params[:,2], xshells, antinu)

    # If pi/2 <= eta <= pi we approximate the density to the constant value taken at r = 1 - h/2
    elif pi/2 <= eta <= pi:

        n_1 = density.call(1 - h/2, 0)

        # Deltax is the length of the crossed path
        Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)

        # Compute the evolutor for constant density n_1 and traveled distance Deltax,
        # and include the factorised dependence on th23 and d to obtain the full evolutor
        evolutor = np.dot(np.dot(r23delta, np.dot(Upert_kinetic(ki, Hk, pmns.theta12, pmns.theta13, Deltax, 0, n_1, 0, 0, antinu), deltac.transpose())), r23.transpose())
        return evolutor

    else:
        raise ValueError('eta must be comprised between 0 and pi.')
