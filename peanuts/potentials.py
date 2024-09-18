#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 23 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import numba as nb

# Earth radius R_E: since we integrate over the dimensionless parameter r = R/R_E, the Hamiltonian must be
# multiplied by R_E
R_E = 6.371e6 # meters

# Sun raidus R_S
R_S = 6.96e8 # meters

# Matter potential
@nb.njit
def MatterPotential (n, antinu):
    """
    MatterPotential(n, antinu) computes the matter potential due to an electron density n, expressed in mol/cm^3
    - n: electron density, in mol / cm^3
    See Eq. 4.17 in 1802.05781.
    """
    return (-1 if antinu else 1) * 3.868e-7 * n # 1/m

# Kinetic potential
@nb.njit
def k(mSq, E):
    """
    k_E(mSq, E) computes the kinetic potential for an ultrarelativistic neutrino:
    - mSq: the squared mass (or mass difference) in units of eV^2;
    - E: the neutrino energy in MeV.
    See Eq. 4.18 in 1802.05781.
    """
    hbarc = 197.3269804e-15
    return 0.5 * 1e-12 * mSq / E / hbarc # 1/m
