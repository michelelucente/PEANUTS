#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 11:34:45 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
"""

# Earth radius R_E: since we integrate over the dimensionless parameter r = R/R_E, the Hamiltonian must be
# multiplied by R_E
R_E = 6.371e6 # meters

# Matter potential
def MatterPotential (n):
    """MatterPotential(n) computes the matter potential due to an electron density n, expressed in mol/cm^3
See Eq. 4.17 in 1802.05781."""
    # n in mol / cm^3
    return R_E * 3.868e-7 * n

# Kinetic potential
def k(mSq, E):
    """k(mSq, E) computes the kinetic potential for an ultrarelativistic neutrino:
    - mSq is the squared mass (or mass difference) in units of eV^2;
    - E is the neutrino energy in MeV.
See Eq. 4.18 in 1802.05781."""
    # mSq in eV^2, E in MeV
    return R_E * 2.533 * mSq / E
