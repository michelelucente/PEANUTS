#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 23 2022

@author: Michele Lucente <michele.lucente@unibo.it>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import time
import numpy as np
import numba as nb
from math import sin, asin, cos, sqrt, pi, floor
from cmath import exp
from mpmath import hyp2f2
from scipy.linalg import inv

from peanuts.potentials import k, MatterPotential, R_E, R_S
from peanuts.integration import c0, c1, lambdas, Iab

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
    - pmns: the PMNS matrix;
    - E: the neutrino energy, in units of MeV;
    - x1 (x2): the starting (ending) point in the path;
    - a, b, c parametrise the density profile on the path, n_e(x) = a + b x^2 + c x^4.
    - antinu: False for neutrinos, True for antineutrinos
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
    V = R_E*MatterPotential(naverage, antinu)

    # Parameter for the density perturbation around the mean density value:
    atilde = a - naverage

    # Travelled distance
    L = (x2 - x1)

    # Reduced mixing matrix U = R_{13} R_{12}
    U = pmns.U
    if antinu:
      U = U.conjugate()

    # Hamiltonian in the reduced flavour basis
    H = np.dot(np.dot(U, np.diag(ki)), U.transpose()) + np.diag(np.array([V, 0, 0]))

    # Traceless Hamiltonian T = H - Tr(H)/3
    tr = np.sum(ki) + V
    T = H - tr/3 * id3

    # Coefficients of the characteristic equation for T
    c0_loc = c0(ki, pmns.theta12, pmns.theta13, naverage, antinu)
    c1_loc = c1(ki, pmns.theta12, pmns.theta13, naverage, antinu)

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
          u1 += np.dot(np.dot(M[idx_a], np.diag(np.array([-1j * R_E*MatterPotential(Iab(lam[idx_a] + tr/3, lam[idx_b] + tr/3, atilde, b, c, x2, x1), antinu), 0, 0]))), M[idx_b])

    u = u0 + u1

    # Return the full evolutor
    return u


@nb.njit
def FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, depth, antinu):
    """
    FullEvolutor(density, DeltamSq21, DeltamSq3l, pmns, E, eta, depth, antinu) computes the full evolutor for an ultrarelativistic
    neutrino crossing the Earth:
    - density: the Earth density object
    - DeltamSq21: the solar mass splitting
    - DeltamSq3l: the atmospheric mass splitting (l=1 for NO, l=2 for IO)
    - pmns: the PMNS matrix
    - E: the neutrino energy, in units of MeV;
    - eta: the nadir angle;
    - depth: the underground detector depth, in units of meters.
    - antinu: False for neutrinos, True for antineutrinos
    """

    # 3d identity matrix of complex numbers
    id3 = np.eye(3, dtype=nb.complex128)

    # If the detector is on the surface and neutrinos are coming from above the horizon, there is no
    # matter effect
    if depth == 0 and (pi/2 <= eta <= pi):
        return (1+0.j)*np.identity(3)

    # Detector depth normalised to Earth radius
    h = depth / R_E

    # Position of detector the on a radial path
    r_d = 1 - h # This is valid for eta = 0

    # Compute the factorised matrices R_{23} and \Delta
    # (remember that U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12})
    r23 = pmns.R23(pmns.theta23)
    delta = pmns.Delta(pmns.delta)

    # Conjugate for antineutrinos
    if antinu:
      r23 = r23.conjugate()
      delta = delta.conjugate()


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
        evolutors_full_path = [Upert(DeltamSq21, DeltamSq3l,pmns, E, params2[i][3], params2[i+1][3] if i < len(params2)-1 else 0, params2[i][0], params2[i][1], params2[i][2], antinu) for i in range(len(params))]

        # Multiply the single evolutors
        evolutor_half_full = evolutors_full_path[0]
        for i in range(len(evolutors_full_path)-1):
          evolutor_half_full = np.dot(evolutor_half_full, evolutors_full_path[i+1])

        # Compute the evolutors for the path from the trajectory mid-point at x == 0 to the detector point x_d
        # Only the evolutor for the most external shell needs to be computed
        evolutors_to_detectors = evolutors_full_path.copy()

        evolutors_to_detectors[0] = Upert(DeltamSq21, DeltamSq3l, pmns, E, x_d, params[-2][3] if len(params) > 1 else 0, params[-1][0], params[-1][1], params[-1][2], antinu)

        # Multiply the single evolutors
        evolutor_half_detector = evolutors_to_detectors[-1]
        for i in range(len(evolutors_to_detectors)-1):
          evolutor_half_detector = np.dot(evolutor_half_detector, evolutors_to_detectors[i+1])

        # Combine the two half-paths evolutors and include the factorised dependence on th23 and d to
        # obtain the full evolutor
        evolutor =  np.dot(np.dot(np.dot(r23, delta.conjugate()), np.dot(evolutor_half_detector, evolutor_half_full.transpose()) ), np.dot(delta, r23.transpose()))
        return evolutor

    # If pi/2 <= eta <= pi we approximate the density to the constant value taken at r = 1 - h/2
    elif pi/2 <= eta <= pi:
        # Nadir angle if the detector was on surface
        eta_prime = pi -  asin(r_d * sin(eta))

        #n_1 = EarthDensity(x = 1 - h / 2) TODO: eta = 0?
        n_1 = density.call(1 - h/2, 0)

        # Deltax is the lenght of the crossed path
        Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)

        # Compute the evolutor for constant density n_1 and traveled distance Deltax,
        # and include the factorised dependence on th23 and d to obtain the full evolutor
        evolutor = np.dot(np.dot(np.dot(r23, delta.conjugate()), np.dot(Upert(DeltamSq21, DeltamSq3l, pmns, E, Deltax, 0, n_1, 0, 0, antinu) , delta)), r23.transpose())
        return evolutor

    else:
        raise ValueError('eta must be comprised between 0 and pi.')


def ExponentialEvolution(initialstate, density, DeltamSq21, DeltamSq3l, pmns, E, xi, xf, antinu=False):
  """
  """

  # Extract from pmns matrix
  U = pmns.U
  r23 = pmns.R23(pmns.theta23)
  delta = pmns.Delta(pmns.delta)

  # 0onjugate for antineutrinos
  if antinu:
    U = U.conjugate()
    r23 = r23.conjugate()
    delta = delta.conjugate()

  # Kinetic terms of the Hamiltonian
  if DeltamSq3l > 0: # NO, l = 1
    ki = k(np.array([0, DeltamSq21, DeltamSq3l]), E)
  else: # IO, l = 2
    ki = k(np.array([-DeltamSq21, 0, DeltamSq3l]), E)

  # Change variables for simplicity
  V0 = MatterPotential(density.call(0), antinu) # 1/m
  r0 = -1./np.log(density.call(1)/density.call(0)) # m
  u0 = -np.log(r0*V0)
  # TODO: Should u be taken as the final coordinate in the path?
  u = xf/r0 + u0
  # TODO: I'm not 100% sure, but I think that this is the initial coordinate in u
  ui = xi/r0 + u0

  # Reduced hamiltonian
  Htilde = r0 * np.dot(U.conjugate(), np.dot(np.diag(ki), U.transpose()))

  # Extract (2,3) block matrix from the Hamiltonian, and define its components
  A = Htilde[1:,1:]
  a = A[0,0]
  b = A[1,1]
  c = A[0,1]
  d = np.sqrt(4*c**2 + (a-b)**2)

  # Construct elements of rotated matrix
  w1 = Htilde[0,0]
  w2 = 0.5*(a+b-d)
  w3 = 0.5*(a+b+d)
  v2 = [(w2-b)/c, 1]/np.sqrt(1+(w2-b)**2/c**2)
  v3 = [(w3-b)/c, 1]/np.sqrt(1+(w3-b)**2/c**2)
  R23a = np.array([v2,v3]).transpose()

  chi2 = Htilde[0,1]
  chi3 = Htilde[0,2]
  [chi2, chi3] = np.dot(R23a.transpose(),[chi2,chi3])

  # Write rotation matrix as 3x3
  r23a = np.block([[1,np.zeros((1,2))],[np.zeros((2,1)),R23a]])

  # Get eigenvalues of rotated hamiltonian
  X1 = w1**2 + w2**2 + w3**2 - w1*w2 - w1*w3 - w2*w3 + 3*(chi2**2 + chi3**2)
  X2 = -(2*w1 - w2 - w3)*(2*w2 - w1 - w3)*(2*w3 - w1 - w2) + 9*((2*w3 - w1 - w2)*chi2**2 + (2*w2 - w1 - w3)*chi3**2)
  phi = np.arctan(np.sqrt(4*X1*(X1/X2)**2-1)) if X2>0 else np.arctan(-np.sqrt(4*X1*(X1/X2)**2-1))+pi if X2<0 else pi/2
  mu1 = (w1 + w2 + w3 - 2*np.sqrt(X1)*np.cos(phi/3))/3
  mu2 = (w1 + w2 + w3 + 2*np.sqrt(X1)*np.cos((phi-pi)/3))/3
  mu3 = (w1 + w2 + w3 + 2*np.sqrt(X1)*np.cos((phi+pi)/3))/3
  [mu1,mu2,mu3] = np.sort([mu1,mu2,mu3])

  # Coefficients of solutions
  [K1, K2, K3] = [1/np.sqrt((w2-mu1)*(w3-mu1)*(mu2-mu1)*(mu3-mu1)),
                  1/np.sqrt((w2-mu2)*(w3-mu2)*(mu1-mu2)*(mu3-mu2)),
                  1/np.sqrt((w2-mu3)*(w3-mu3)*(mu1-mu3)*(mu2-mu3))]

  # Solution at end
  psi1 = np.array([K1 * (mu1-w2)*(mu1-w3) * np.exp(-1j*mu1*u) * hyp2f2(1-1j*(w2-mu1), 1-1j*(w3-mu1), 1-1j*(mu2-mu1), 1-1j*(mu3-mu1), 1j*np.exp(-u)),
                   K2 * (mu2-w2)*(mu2-w3) * np.exp(-1j*mu2*u) * hyp2f2(1-1j*(w2-mu2), 1-1j*(w3-mu2), 1-1j*(mu1-mu2), 1-1j*(mu3-mu2), 1j*np.exp(-u)),
                   K3 * (mu3-w2)*(mu3-w3) * np.exp(-1j*mu3*u) * hyp2f2(1-1j*(w2-mu3), 1-1j*(w3-mu3), 1-1j*(mu1-mu3), 1-1j*(mu2-mu3), 1j*np.exp(-u))],
          dtype=np.complex128)

  psi2 = np.array([K1 * chi2 * (mu1-w3) * np.exp(-1j*mu1*u) * hyp2f2(-1j*(w2-mu1), 1-1j*(w3-mu1), 1-1j*(mu2-mu1), 1-1j*(mu3-mu1), 1j*np.exp(-u)),
                   K2 * chi2 * (mu2-w3) * np.exp(-1j*mu2*u) * hyp2f2(-1j*(w2-mu2), 1-1j*(w3-mu2), 1-1j*(mu1-mu2), 1-1j*(mu3-mu2), 1j*np.exp(-u)),
                   K3 * chi2 * (mu3-w3) * np.exp(-1j*mu3*u) * hyp2f2(-1j*(w2-mu3), 1-1j*(w3-mu3), 1-1j*(mu1-mu3), 1-1j*(mu2-mu3), 1j*np.exp(-u))],
         dtype=np.complex128)

  psi3 = np.array([K1 * chi3 * (mu1-w2) * np.exp(-1j*mu1*u) * hyp2f2(1-1j*(w2-mu1), -1j*(w3-mu1), 1-1j*(mu2-mu1), 1-1j*(mu3-mu1), 1j*np.exp(-u)),
                   K2 * chi3 * (mu2-w2) * np.exp(-1j*mu2*u) * hyp2f2(1-1j*(w2-mu2), -1j*(w3-mu2), 1-1j*(mu1-mu2), 1-1j*(mu3-mu2), 1j*np.exp(-u)),
                   K3 * chi3 * (mu3-w2) * np.exp(-1j*mu3*u) * hyp2f2(1-1j*(w2-mu3), -1j*(w3-mu3), 1-1j*(mu1-mu3), 1-1j*(mu2-mu3), 1j*np.exp(-u))],
         dtype=np.complex128)

  #print("-- running exponential evolution")

  # Solutions at beginning
  psi = []
  uI = ui if (isinstance(ui, list) or isinstance(ui, np.ndarray)) else [ui]
  for i,I in enumerate(uI):
    psi10 = np.array([K1 * (mu1-w2)*(mu1-w3) * np.exp(-1j*mu1*I) * hyp2f2(1-1j*(w2-mu1), 1-1j*(w3-mu1), 1-1j*(mu2-mu1), 1-1j*(mu3-mu1), 1j*np.exp(-I)),
                    K2 * (mu2-w2)*(mu2-w3) * np.exp(-1j*mu2*I) * hyp2f2(1-1j*(w2-mu2), 1-1j*(w3-mu2), 1-1j*(mu1-mu2), 1-1j*(mu3-mu2), 1j*np.exp(-I)),
                    K3 * (mu3-w2)*(mu3-w3) * np.exp(-1j*mu3*I) * hyp2f2(1-1j*(w2-mu3), 1-1j*(w3-mu3), 1-1j*(mu1-mu3), 1-1j*(mu2-mu3), 1j*np.exp(-I))],
            dtype=np.complex128)


    psi20 = np.array([K1 * chi2 * (mu1-w3) * np.exp(-1j*mu1*I) * hyp2f2(-1j*(w2-mu1), 1-1j*(w3-mu1), 1-1j*(mu2-mu1), 1-1j*(mu3-mu1), 1j*np.exp(-I)),
                    K2 * chi2 * (mu2-w3) * np.exp(-1j*mu2*I) * hyp2f2(-1j*(w2-mu2), 1-1j*(w3-mu2), 1-1j*(mu1-mu2), 1-1j*(mu3-mu2), 1j*np.exp(-I)),
                    K3 * chi2 * (mu3-w3) * np.exp(-1j*mu3*I) * hyp2f2(-1j*(w2-mu3), 1-1j*(w3-mu3), 1-1j*(mu1-mu3), 1-1j*(mu2-mu3), 1j*np.exp(-I))],
            dtype=np.complex128)

    psi30 = np.array([K1 * chi3 * (mu1-w2) * np.exp(-1j*mu1*I) * hyp2f2(1-1j*(w2-mu1), -1j*(w3-mu1), 1-1j*(mu2-mu1), 1-1j*(mu3-mu1), 1j*np.exp(-I)),
                    K2 * chi3 * (mu2-w2) * np.exp(-1j*mu2*I) * hyp2f2(1-1j*(w2-mu2), -1j*(w3-mu2), 1-1j*(mu1-mu2), 1-1j*(mu3-mu2), 1j*np.exp(-I)),
                    K3 * chi3 * (mu3-w2) * np.exp(-1j*mu3*I) * hyp2f2(1-1j*(w2-mu3), -1j*(w3-mu3), 1-1j*(mu1-mu3), 1-1j*(mu2-mu3), 1j*np.exp(-I))],
            dtype=np.complex128)

    # Get coefficients from boundary conditions (rotate initial state to match the rotated hamiltonian)
    Mboundary = np.matrix([psi10, psi20, psi30])
    phi0 = np.dot(r23a.transpose(), np.dot(delta, np.dot(r23.transpose(), initialstate)))
    C = np.dot(inv(Mboundary),phi0)

    # Get final solution
    psiI = np.array([np.dot(C,psi1), np.dot(C,psi2), np.dot(C,psi3)],dtype=np.complex128)

    # Undo rotation of 2-3 components
    psiI = np.dot(r23a, psiI)

    # Reintroduce the theta23 and delta matrices
    psiI = np.dot(r23,np.dot(delta.conjugate(), psiI))

    if len(uI) > 1 and not (i+1)*100/len(uI) % 1:
      print("---- completed ", floor((i+1)/len(uI)*100), "%")

    # Fill solution array
    psi.append(psiI)

  if len(psi) == 1:
    return psi[0]
  else:
    return np.array(psi)

