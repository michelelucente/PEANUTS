#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 7 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""
# Import libraries
import numpy as np
from math import cos, sin
from scipy import integrate
from src.matter_mixing import th13_M, th12_M
from src.pmns import PMNS

import src.files as f

class SolarModel:
    """"
    Class containing the info of the solar model
    """

    def __init__(self, filename="./Data/bs2005agsopflux.csv"):
        """
        Constructor of the solar model. 
        Reads the solar model file and fills useful variables
        """

        # Set file name
        self.filename = "./Data/bs2005agsopflux.csv" if filename == "" else filename
          
        # Import data from solar model
        # TODO: This assumes that any solar model file is the same format, make it more general
        self.model = f.read_csv(filename, 
                                usecols=[1, 3, 7, 13],
                                names = ['radius', 'density_log_10', '8B fraction', 'hep fraction'],
                                sep=" ", skiprows=27, header=None)

        # Set useful variables
        self.radius = self.model['radius']
        self.density =  10**self.model['density_log_10']
        self.fraction = {'8B' : self.model['8B fraction'],
                         'hep': self.model['hep fraction']}

    def radius(self):
        """
        Returns the radius column of the solar model
        """

        return self.radius

    def density(self):
        """
        Returns the density column of the solar model
        """

        return self.density



    def fraction(self, name):
        """
        Returns the fraction of neutrinos for the requested column
        """

        return self.fraction[name]

    def has_fraction(self, name):
       """
       Returns whether the solar model contains the given neutrino sample fraction
       """

       return name in self.fraction.keys()

# Compute flux of incoherent mass eigenstates for fixed densitiy value
def Tei (th12, th13, DeltamSq21, DeltamSq31, E, ne):
    """Tei(th12, th13, DeltamSq21, DeltamSq31, E, ne) computes the weights composing an incoherent flux of
    neutrino mass eigenstates, for electron neutrinos produced in matter in the adiabatic approximation:
    - thij are the PMNS mixing angles;
    - DeltamSqj1 are the squared mass differences in units of eV^2;
    - E is the neutrino energy, in units of MeV;
    - ne is the electron density at production point, in units of mol/cm^3.
See Eq. (6.11) in FiuzadeBarros:2011qna for its derivation."""
    
    # Compute the mixing angles at neutrino production point
    th13m = th13_M(th13, DeltamSq31, E, ne)
    th12m = th12_M(th12, th13, DeltamSq21, E, ne)
    
    # Compute and return the weights
    c13M = cos(th13m)
    s13M = sin(th13m)
    c12M = cos(th12m)
    s12M = sin(th12m)
    
    return ((c13M * c12M)**2, (c13M * s12M)**2, s13M**2)



# Compute flux of inchoerent mass eigenstates integrated over production point in the Sun    
def solar_flux_mass (th12, th13, DeltamSq21, DeltamSq31, E, radius_samples, density, fraction):
    """solar_flux_mass(th12, th13, DeltamSq21, DeltamSq31, E, radius_samples, density, fraction) computes 
    the weights of mass eigenstates composing the incoherent flux of solar neutrinos in the adiabatic 
    approximation:
    - thij are the PMNS mixing angles;
    - DeltamSqj1 are the squared mass differences in units of eV^2;
    - E is the neutrino energy, in units of MeV;
    - radius_samples is a list of solar relative radius values, where density and fraction are sampled;
    - density is the list of electron densities at radii radius_samples, in units of mol/cm^3;
    - fraction is the relative fraction of neutrinos produced in the considered reaction, 
    sampled at radius_samples."""
    
    IntegratedFraction = integrate.trapezoid(y=fraction, x=radius_samples)
    Tei_radius = np.array([Tei(th12, th13, DeltamSq21, DeltamSq31, E, ne_r) for ne_r in density])
    
    [Te1, Te2, Te3] = [
        integrate.trapezoid(y=([Tei_radius[k][i] for k in range(len(Tei_radius))] * fraction), x = radius_samples) / IntegratedFraction 
        for i in range(3)]
    
    return [Te1, Te2, Te3]

# Compute the flavour probabilities for the solar neutrino flux
def Psolar (pmns, DeltamSq21, DeltamSq31, E, radius_samples, density, fraction):
    """Psolar(th12, th13, th23, d, DeltamSq21, DeltamSq31, E, radius_samples, density, fraction) computes the
    flavour probabilities of observing a solar neutrino as a given flavour. 
    The function returns a list for the probabilities [P(electron), P(muon), P(tau)].
    The neutrino has energy E and is produced in a specific reaction:
    - pmns is the PMNs matrix
    - DeltamSqj1 are the vacuum squared mass difference between mass eigenstates j and 1;
    - E is the neutrino energy, in units of MeV;
    - radius_samples is a list of solar relative radius values where density and fraction are sampled;
    - density is the list of electron densities at radii radius_samples, in units of mol/cm^3;
    - fraction is the relative fraction of neutrinos produced in the considered reaction,  
    sampled at radius_samples."""
    
    # Compute the weights in the uncoherent solar flux of mass eigenstates
    Tei = np.array(solar_flux_mass(pmns.theta12, pmns.theta13, DeltamSq21, DeltamSq31, E, radius_samples, density, fraction))
    
    # Compute the probabilities that a mass eigenstate is observed as a given flavour
    #P_i_to_a = np.square(np.abs(PMNS(th12, th13, th23, -d))) # TODO: Why negative -d? ANSWER: because the mass eigenstates are given by 
    # the conjugate of the PMNS mixing matrix, thus we can simply invert the CP phase instead of taking the conjugate of the full matrix
    P_i_to_a = np.square(np.abs(pmns.conjugate()))
    
    # Multiply probabilities by weights, and return the result
    return np.array(np.dot(P_i_to_a, Tei))[0]
