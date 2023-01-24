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



# Define mixing angles in matter
@nb.njit
def th12_M (th12, th13, DeltamSq21, E, ne):
    """
    th12_M(th12, th13, DeltamSq21, E, ne) computes the mixing angle \theta_{12} in matter.
    - th1j: the vacuum mixing angles in radians;
    - DeltamSq21: the vacuum squared mass difference between mass eigenstates 2 and 1;
    - E: the neutrino energy, in units of MeV;
    - ne: the electron matter density, in units of mol/cm^3.
    See also Eq. 1.22 in FiuzadeBarros:2011qna.
    """

    return (np.arctan(np.tan(2*th12) / (1 - (np.cos(th13)**2)/(np.cos(2* th12)) * Vk(DeltamSq21, E, ne))) / 2) % (np.pi/2)

@nb.njit
def th13_M (th13, DeltamSq31, E, ne):
    """
    th13_M(th13, DeltamSq31, E, ne) computes the mixing angle \theta_{13} in matter.
    - th13: the vacuum mixing angles \theta_{13} in radians;
    - DeltamSq31: the vacuum squared mass difference between mass eigenstates 3 and 1;
    - E: the neutrino energy, in units of MeV;
    - ne: the electron matter density, in units of mol/cm^3.
    See also Eq. 1.22 in FiuzadeBarros:2011qna.
    """

    return (np.arcsin(np.sin(th13) * (1 + Vk(DeltamSq31, E, ne) * np.cos(th13)**2))) % (np.pi/2)
