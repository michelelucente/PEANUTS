#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:09:53 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""
import numba as nb
import numpy as np


# Define function for the ratio between matter and vacuum terms in neutrino oscillations
@nb.njit
def Vk(Deltam2, E, ne):
    """
    Vk(Deltam2, E, ne) computes the ratio V/k between the matter and vacuum terms in neutrino oscillations.
    - Deltam2: the squared mass difference, in units of eV^2;
    - E: the neutrino energy, in units of MeV;
    - ne: the electron matter density, in units of mol/cm^3;
    See also Eq.s 4.17, 4.18 in 1802.05781.
    """

    return (3.868e-7)/(2.533) * (ne * E / Deltam2)

@nb.njit
def DeltamSqee(th12, DeltamSq21, DeltamSq3l):
    """
    DeltamSqee(th12, DeltamSq21, DeltamSq3l) computes the dielectron invariant mass
    - th12: the vacuum mixing angle th_{12}
    - DeltamSq21, DeltamSq3l: the vacuum squared mass difference;
    """

    DeltamSq31 = DeltamSq3l if DeltamSq3l > 0 else DeltamSq3l + DeltamSq21
    DeltamSq32 = DeltamSq3l if DeltamSq3l < 0 else DeltamSq3l - DeltamSq21

    return np.cos(th12)**2*DeltamSq31 + np.sin(th12)**2*DeltamSq32

# Define mixing angles in matter
@nb.njit
def th13_M (th12, th13, DeltamSq21, DeltamSq3l, E, ne):
    """
    th13_M(th13, DeltamSq31, E, ne) computes the mixing angle \theta_{13} in matter.
    - thij: the vacuum mixing angles in radians;
    - DeltamSq21, DeltamSq3l: the vacuum squared mass difference;
    - E: the neutrino energy, in units of MeV;
    - ne: the electron matter density, in units of mol/cm^3.
    See arXiv:1604.08167 and arXiv:1801.06514.
    """

    vk = Vk(DeltamSqee(th12, DeltamSq21, DeltamSq3l), E, ne)

    return 0.5*np.arccos((np.cos(2*th13) - vk) / np.sqrt((np.cos(2*th13) - vk)**2 + np.sin(2*th13)**2)) % (np.pi/2)

@nb.njit
def th12_M (th12, th13, DeltamSq21, DeltamSq3l, E, ne):
    """
    th12_M(th12, th13, DeltamSq21, E, ne) computes the mixing angle \theta_{12} in matter.
    - th1j: the vacuum mixing angles in radians;
    - DeltamSq21, DeltamSq3l: the vacuum squared mass difference;
    - E: the neutrino energy, in units of MeV;
    - ne: the electron matter density, in units of mol/cm^3.
    See arXiv:1604.08167 and arXiv:1801.06514.
    """

    th13m = th13_M(th12, th13, DeltamSq21, DeltamSq3l, E, ne)
    Vkprime = Vk(DeltamSq21, E, ne)*np.cos(th13m)**2 + DeltamSqee(th12, DeltamSq21, DeltamSq3l)/DeltamSq21*np.sin(th13m-th13)**2

    return 0.5*np.arccos((np.cos(2*th12) - Vkprime) / np.sqrt((np.cos(2*th12) - Vkprime)**2 + np.sin(2*th12)**2*np.cos(th13m-th13)**2)) % (np.pi/2)
