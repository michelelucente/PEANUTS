#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 23 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
"""

import numpy as np
from math import cos
from cmath import exp
from cmath import sqrt as csqrt

from src.potentials import k, MatterPotential


# Computes coefficients of the characteristic equation for the matrix T = H - Tr(H)/3, cf. hep-ph/9910546 
def c0 (m1Sq, m2Sq, m3Sq, th12, th13, E, n):
    """c0(m1Sq, m2Sq, m3Sq, E, n) computes the coefficient c_0 defined in hep-ph/9910546, for the specific case
    in which the mixing matrix is the reduced one U = R_{13} R_{12}:
    - miSq are the squared masses (or mass differences) in units of eV**2;
    - thij are the PMNS mixing angles;
    - E is the neutrino energy, in units of MeV;
    - n is the electron matted density, in units of mol/cm**3.
See hep-ph/9910546 for the full context of the definition."""
    [k1, k2, k3] = k(np.array([m1Sq, m2Sq, m3Sq]), E)
    V = MatterPotential(n)
    
    return (-4*(k1 + k2 - 2*k3)*(2*k1 - k2 - k3)*(k1 - 2*k2 + k3) + 
  3*(k1**2 - 4*k1*k2 + k2**2 + 2*(k1 + k2)*k3 - 2*k3**2)*V + 
  3*(k1 + k2 - 2*k3)*V**2 - 8*V**3 - 18*(k1 - k2)*V*(k1 + k2 - 2*k3 + V)*
   cos(2*th12)*cos(th13)**2 - 9*V*(k1**2 + k2**2 - 2*k3*(k3 + V) + 
    k2*(2*k3 + V) + k1*(-4*k2 + 2*k3 + V))*cos(2*th13) )/108


def c1(m1Sq, m2Sq, m3Sq, th12, th13, E, n):
    """c1(m1Sq, m2Sq, m3Sq, E, n) computes the coefficient c_1 defined in hep-ph/9910546, for the specific case
    in which the mixing matrix is the reduced one U = R_{13} R_{12}:
    - miSq are the squared masses (or mass differences) in units of eV**2;
    - thij are the PMNS mixing angles;
    - E is the neutrino energy, in units of MeV;
    - n is the electron matted density, in units of mol/cm**3.
See hep-ph/9910546 for the full context of the definition."""
    [k1, k2, k3] = k(np.array([m1Sq,m2Sq,m3Sq]), E)
    V = MatterPotential(n)
    
    return (-4*(k1**2 - k1*k2 + k2**2 - (k1 + k2)*k3 + k3**2) + (k1 + k2 - 2*k3)*V - 
  4*V**2 + 6*(-k1 + k2)*V*cos(2*th12)*cos(th13)**2 - 
  3*(k1 + k2 - 2*k3)*V*cos(2*th13))/12


# Computes the solutions of the characteristic equation for the matrix T = H - Tr(H)/3, cf. hep-ph/9910546 
def lambdas (c0, c1):
    """lambdas(c0, c1) computes the solutions (roots) of the characteristic equation for the matrix 
    T = H - Tr(H)/3, i.e. the quantities \lambda_i defined in Eq.s (33-35) in hep-ph/9910546:
    - c0, c1 are the coefficients of the characteristic equations for the matrix T = H - Tr(H)/3, they are
    computed by the functions c0(m1Sq, m2Sq, m3Sq, E, n) and c1(m1Sq, m2Sq, m3Sq, E, n) for the specific
    scenario in which the mixing matrix is the reduced one U = R_{13} R_{12}.
    
    The function returns a list containing the 3 roots.
See hep-ph/9910546 for the full context of the definition."""
    l1 = (-2*3**(1/3)*c1 + 2**(1/3)*(-9*c0 + csqrt(81*c0**2 + 12*c1**3))**(2/3))/(6**(2/3)*(-9*c0 + csqrt(81*c0**2 + 12*c1**3))**(1/3))
    l2 = ((-1)**(1/3)*(2*3**(1/3)*c1 + (-2)**(1/3)*(-9*c0 + csqrt(81*c0**2 + 12*c1**3))**(2/3)))/(6**(2/3)*(-9*c0 + csqrt(81*c0**2 + 12*c1**3))**(1/3))
    l3 = -(((-1)**(1/3)*(2*(-3)**(1/3)*c1 + 2**(1/3)*(-9*c0 + csqrt(81*c0**2 + 12*c1**3))**(2/3)))/(6**(2/3)*(-9*c0 + csqrt(81*c0**2 + 12*c1**3))**(1/3)))
    
    return [l1, l2, l3]



# Compute the integrals required for the first order correction in the evolutor
def Iab (la, lb, atilde, b, c, x2, x1):
    """Iab(la, lb, atilde, b, c, x2, x1) computes the definite integral: 
    \int_x1**x2 dx e**{- i la (x2-x)} (atilde + b x**2 + c x**4) e**{- i lb (x-x1)}.
    
    It is assumed that the integral of (atilde + b x**2 + c x**4) vanished on the considered interval, i.e.
    atilde (x2-x1) + b (x2**3 -x1**3)/3 + c (x2**5 - x1**5)/5 = 0.
    
    The integral is identically zero if la == lb.
    
    The analytic solution is numerically instable when la ~ lb, thus for the case in which 
    abs((la - lb) / (la + lb)) < 1e-2 we compute the 2nd order Taylor expansion around the point la == lb,
    which is numerically stable."""
    
    # The analytic solution depends on la, lb only via Dl = la - lb, except for a common phase factor.
    Dl = la - lb
    
    # If la == la the integral is identically zero.
    if Dl == 0:
        return 0
    
    # For small differences between la and lb we compute the 2nd order Taylor expansion around la == lb,
    # which is numerically stable
    elif np.abs(Dl / (la + lb)) < 1e-2:
        return exp(1j*lb*(-x2 + x1))*(Dl*((-1j/2)*atilde*(x2 - x1)**2 - (1j/12)*b*(x2**4 - 4*x2*x1**3 + 
          3*x1**4) - (1j/30)*c*(x2**6 - 6*x2*x1**5 + 5*x1**6)) + 
          Dl**2*(-(atilde*(x2 - x1)**3)/6 - 
        (b*(x2**5 - 10*x2**2*x1**3 + 15*x2*x1**4 - 6*x1**5))/60 - 
        (c*(x2**7 - 21*x2**2*x1**5 + 35*x2*x1**6 - 15*x1**7))/210))

    # In the other regions we compute the full analytic solution.
    else:
        return exp(1j*lb*(-x2 + x1))*((atilde*(-1j + 1j/exp(1j*Dl*(x2 - x1))))/Dl + 
                (b*(2*1j + 2*Dl*x2 - 1j*Dl**2*x2**2 + (1j*(-2 + (2*1j)*Dl*x1 + Dl**2*x1**2))/
          exp(1j*Dl*(x2 - x1))))/Dl**3 - 
          (1j*c*(24 + Dl*x2*(-24*1j + Dl*x2*(-12 + Dl*x2*(4*1j + Dl*x2))) - 
             (24 + Dl*x1*(-24*1j + Dl*x1*(-12 + Dl*x1*(4*1j + Dl*x1))))/
          exp(1j*Dl*(x2 - x1))))/Dl**5)