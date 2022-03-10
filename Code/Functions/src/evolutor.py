#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 23 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
"""

import numpy as np
from numpy.linalg import multi_dot
from math import sin, cos, sqrt, pi
from cmath import exp

from src.potentials import k, MatterPotential, R_E
from src.integration import c0, c1, lambdas, Iab

def Upert (m1Sq, m2Sq, m3Sq, pmns, E, x2=1, x1=0, a=0, b=0, c=0):
    """Upert(m1Sq, m2Sq, m3Sq, pmns, E,  x2=1, x1=0, a=0, b=0, c=0, order = 1) computes the evolutor
    for an ultrarelativistic neutrino state in flavour basis, for a reduced mixing matrix U = R_{13} R_{12}
    (the dependence on th_{23} and CP-violating phase \delta_{CP} can be factorised) for a density profile 
    parametrised by a 4th degree even poliomial in the trajectory coordinate, to 1st order corrections around
    the mean density value:
    - miSq are the squared masses (or mass differences) in units of eV^2;
    - pmns is the PMNS matrix;
    - E is the neutrino energy, in units of MeV;
    - x1 (x2) is the starting (ending) point in the path;
    - a, b, c parametrise the density profile on the path, n_e(x) = a + b x^2 + c x^4.
See hep-ph/9702343 for the definition of the perturbative expansion of the evolutor in a 2-flavours case."""
    
    # Kinetic terms of the Hamiltonian
    [k1, k2, k3] = k(np.array([m1Sq, m2Sq, m3Sq]), E)
    
    # Average matter density along the path
    naverage = (a * (x2 - x1) + b * (x2**3 - x1**3)/3 + c * (x2**5 - x1**5)/5) / (x2 - x1)
    
    # Matter potential for the 0th order evolutor
    V = MatterPotential(naverage)
    
    # Parameter for the density perturbation around the mean density value: 
    # \delta n_e(x) = atilde + b x^2 + c x^4
    atilde = a - naverage
    
    # Travelled distance
    L = (x2 - x1)

    # Reduced mixing matrix U = R_{13} R_{12}
    U = pmns.U
    
    # Hamiltonian in the reduced flavour basis
    H = multi_dot([U, np.diag([k1, k2, k3]), U.transpose()]) + np.diag([V, 0, 0])
    
    # Traceless Hamiltonian T = H - Tr(H)/3
    tr = np.sum([k1, k2, k3, V])
    T = H - tr/3 * np.identity(3)

    # Coefficients of the characteristic equation for T
    c0_loc = c0(m1Sq, m2Sq, m3Sq, pmns.theta12, pmns.theta13, E, naverage)
    c1_loc = c1(m1Sq, m2Sq, m3Sq, pmns.theta12, pmns.theta13, E, naverage)

    # Roots of the characteristic equation for T
    lam = lambdas(c0_loc, c1_loc)

    # Matrices M_a, not depending on x
    M = [(1 / (3*la**2 + c1_loc)) * 
                    ((la**2 + c1_loc) * np.identity(3) + la * T + np.dot(T,T)) for la in lam]

    # 0th order evolutor (i.e. for constant matter density), following Eq. (46) in hep-ph/9910546
    u0 = np.sum([exp(-1j * (lam[i] + tr/3) * L) * M[i] for i in range(len(lam))], 0)

    # Compute correction to evolutor, taking into account 1st order terms in \delta n_e(x)
    if (b != 0) | (c != 0):
        u1 = np.sum([multi_dot([M[idx_a], np.diag([-1j * MatterPotential(Iab(lam[idx_a] + tr/3, lam[idx_b] + tr/3, 
                                                                             atilde, b, c, x2, x1)), 0, 0]),
                            M[idx_b]]) for idx_a in range(3) for idx_b in range(3)], 0)
    # If density profile is constant the 1st order correction is identically zero
    else:
        u1 = 0

    # Return the full evolutor
    return u0 + u1



def FullEvolutor (density, m1Sq, m2Sq, m3Sq, pmns, E, eta, H):
    """FullEvolutor(density, m1Sq, m2Sq, m3Sq, pmns, E, eta, H) computes the full evolutor for an ultrarelativistic
    neutrino crossing the Earth:
    - density is the Earth density object
    - miSq are the squared masses (or mass differences) in units of eV^2;
    - pmns is the PMNS matrix
    - E is the neutrino energy, in units of MeV;
    - d is the CP-violating PMNS phase;
    - eta is the nadir angle;
    - H is the underground detector depth, in units of meters."""
    
    # Detector depth normalised to Earth radius
    h = H/R_E
    
    # Position of the detector along the trajectory coordinate
    r_d = 1 - h # This is valid for eta = 0
    x_d = sqrt(r_d**2 - sin(eta)**2) # For generic values of eta
    
    # Compute the factorised matrices R_{23} and \Delta 
    # (remember that U_{PMNS} = R_{23} \Delta R_{13} \Delta^* R_{12})
    r23= pmns.R23(pmns.theta23)
    delta = pmns.Delta(pmns.delta)
    
    # If 0 <= eta < pi/2 we compute the evolutor taking care of matter density perturbation around the
    # density mean value at first order
    if 0 <= eta < pi/2:
        # params is a list of lists, each element [[a, b, c], x_i] contains the parameters of the density 
        # profile n_e(x) = a + b x^2 + c x^4 along the crossed shell, with each shell ending at x == x_i
        params = density.parameters(eta)

        # Compute the evolutors for the path from Earth entry point to trajectory mid-point at x == 0
        evolutors_full_path = [Upert(m1Sq, m2Sq, m3Sq, pmns, E, params[i][1], params[i-1][1] if i > 0 else 0, 
                           params[i][0][0], params[i][0][1], params[i][0][2]) for i in reversed(range(len(params)))]

        # Multiply the single evolutors
        evolutor_half_full = multi_dot(evolutors_full_path) if len(evolutors_full_path) > 1 else evolutors_full_path[0]
        
        
        # Compute the evolutors for the path from the trajectory mid-point at x == 0 to the detector point x_d
        # Only the evolutor for the most external shell needs to be computed
        evolutors_to_detectors = evolutors_full_path.copy()
        evolutors_to_detectors[0] = Upert(m1Sq, m2Sq, m3Sq, pmns, E, x_d, params[-2][1] if len(params) > 1 else 0, 
                           params[-1][0][0], params[-1][0][1], params[-1][0][2])

        # Multiply the single evolutors
        evolutor_half_detector = multi_dot(evolutors_to_detectors) if len(evolutors_to_detectors) > 1 else evolutors_to_detectors[0]

        # Combine the two half-paths evolutors and include the factorised dependence on th23 and d to
        # obtain the full evolutor
        evolutor = multi_dot([r23, delta.conjugate(), evolutor_half_detector, evolutor_half_full.transpose(), 
                              delta, r23.transpose()])
        return evolutor

    # If pi/2 <= eta <= pi we approximate the density to the constant value taken at r = 1 - h/2
    elif pi/2 <= eta <= pi:
        #n_1 = EarthDensity(x = 1 - h / 2) TODO: eta = 0?
        n_1 = density(1 - h/2, 0)

        # Deltax is the lenght of the crossed path
        Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)

        # Compute the evolutor for constant density n_1 and traveled distance Deltax,
        # and include the factorised dependence on th23 and d to obtain the full evolutor
        evolutor = multi_dot([r23, delta.conjugate(), 
                              Upert(m1Sq, m2Sq, m3Sq, pmns, E, Deltax, 0, n_1, 0, 0), 
                              delta, r23.transpose()])
        return evolutor

    else:
        raise ValueError('eta must be comprised between 0 and pi.')
