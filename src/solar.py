#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 7 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""
# Import libraries
import os
import numpy as np
import numba as nb
from math import cos, sin
from scipy import integrate
from scipy.interpolate import interp1d
from src.matter_mixing import th13_M, th12_M
from src.pmns import PMNS

import src.files as f

class SolarModel:
    """"
    Class containing the info of the solar model
    """

    def __init__(self, filename=None, spectrum_files={}):
        """
        Constructor of the solar model.
        Reads the solar model file and fills useful variables
        """

        # Set file name
        path = os.path.dirname(os.path.realpath( __file__ ))
        self.filename = path + "/../Data/bs2005agsopflux.csv" if filename == None else filename

        # Format for the various spectra are layered differently
        fluxrow = 0
        fractionrow = 0
        if "bs2005" in self.filename:
          fluxcols = [3, 5]
          fluxrow = 6
          fractioncols = [1, 3, 7, 13]
          fractionrow = 27
        elif "bp00" in self.filename:
          fluxcols=[3,5]
          fluxrow = 25
          fractioncols = [0, 2, 6, 12]
          fractionrow = 29
        else:
            print("Error: Unknown solar model file")
            exit()


        try:
          # Import fluxes
          self.fluxes = f.read_csv(self.filename,
                                   usecols = fluxcols,
                                   names = ['hep', '8B'],
                                   sep=" ", skiprows=fluxrow, nrows=1, header=None)

          # Import fraction data from solar model
          self.model = f.read_csv(self.filename,
                                  usecols = fractioncols,
                                  names = ['radius', 'density_log_10', '8B fraction', 'hep fraction'],
                                  sep=" ", skiprows=fractionrow, header=None)

        except:
          print("Error! The solar model file provided does not exists or it is corrupted")
          exit()

        # Set useful variables
        self.rad = self.model['radius']
        self.dens =  10**self.model['density_log_10']
        self.frac = {'8B' : self.model['8B fraction'],
                     'hep': self.model['hep fraction']}

        # Import spectral shapes
        spectrum_files["8B"] = path + "/../Data/8B_shape_Ortiz_et_al.csv" if "8B" not in spectrum_files else spectrum_files['8B']
        spectrum_files["hep"] = path + "/../Data/hep_shape.csv" if "hep" not in spectrum_files else spectrum_files['hep']
        self.spectra = {}
        for fraction, spectrum_file in spectrum_files.items():
          self.spectra[fraction] = f.read_csv(spectrum_file, usecols=[0, 1], names = ["Energy", "Spectrum"], skiprows=3, header=None)
          norm = 1000 if "Winter_et_al" in spectrum_file else 1
          self.spectra[fraction]['Spectrum'] /= norm


    def radius(self):
        """
        Returns the radius column of the solar model
        """

        return self.rad.to_numpy()

    def density(self):
        """
        Returns the density column of the solar model
        """

        return self.dens.to_numpy()



    def fraction(self, name):
        """
        Returns the fraction of neutrinos for the requested column
        """

        return self.frac[name].to_numpy()

    def has_fraction(self, name):
       """
       Returns whether the solar model contains the given neutrino sample fraction
       """

       return name in self.fraction.keys()

    def flux(self,name):
       """
       Returns the cumulative fluxes for each channel in cm^-2 s^-1
       """

       return self.fluxes[name][0]*1e10

    def spectrum(self, name, energy=None):
       """
       Returns the energy spectrum for each channel
       """

       if energy == None:
         return self.spectra[name]
       else:
         if energy < min(self.spectra[name].Energy) or energy > max(self.spectra[name].Energy):
           print("Error: selected energy is outside the valid range of the energy spectrum")
           exit()
         spec = interp1d(self.spectra[name].Energy, self.spectra[name].Spectrum)
         return float(spec(energy))

# Compute flux of incoherent mass eigenstates for fixed density value
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
    c13M = np.cos(th13m)
    s13M = np.sin(th13m)
    c12M = np.cos(th12m)
    s12M = np.sin(th12m)

    return np.array(((c13M * c12M)**2, (c13M * s12M)**2, s13M**2))


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


    IntegratedFraction = integrate.trapz(y=fraction, x=radius_samples)

    temp = Tei(th12, th13, DeltamSq21, DeltamSq31, E, density) * fraction
    [Te1, Te2, Te3] = [
        integrate.trapz(y=temp[i], x = radius_samples) / IntegratedFraction
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
    return np.array(np.dot(P_i_to_a, Tei))
