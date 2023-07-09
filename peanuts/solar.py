#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 7 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author: Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""
# Import libraries
import os
import time
import numpy as np
import numba as nb
from math import cos, sin
from scipy import integrate
from scipy.interpolate import interp1d
from peanuts.matter_mixing import th13_M, th12_M
from peanuts.potentials import R_S
from peanuts.evolutor import ExponentialEvolution
from peanuts.pmns import PMNS

import peanuts.files as f

class SolarDensity:
    """
    Class containing the solar density"
    """

    def __init__(self, density_table):

      # Fill in the tabulated density from a Solar Model
      self.adiabatic_density = density_table.to_numpy()

      # Fill in the exponential profile of the density
      self.V0 = 0.0458e-3 / 3.868e-7 # mol/cm^3
      self.r0 = 0.1*R_S

    def call(self, r):
      """
      Return the the exponential profile density at radius r
      """

      return self.V0*np.exp(-r/self.r0)

    def table(self):
      """
      Return the adiabatic density table
      """

      return self.adiabatic_density

class SolarModel:
    """"
    Class containing the info of the solar model
    """

    def __init__(self, solar_model_file=None, flux_file=None, spectrum_files=None, fluxrows=None, fluxcols=None, fluxscale=None, distrow=None, radiuscol=None, densitycol=None, fractioncols=None):
        """
        Constructor of the solar model.
        Reads the solar model file and fills useful variables
        """

        # Set file name
        path = os.path.dirname(os.path.realpath( __file__ ))
        # Default solar model file is the B16 from Vinyoles et al 2016, https://www.ice.csic.es/personal/aldos/Solar_Data.html
        self.solar_model_file = path + "/../Data/nudistr_b16_agss09.dat" if solar_model_file == None else solar_model_file
        # Default flux file is the B16 from Vinyoles et al 2016, https://www.ice.csic.es/personal/aldos/Solar_Data.html
        self.flux_file = path + "/../Data/fluxes_b16.dat" if (flux_file == None and "b16" in self.solar_model_file) else (solar_model_file if flux_file == None else flux_file)

        # Format for the various spectra are layered differently
        if "b16" in self.solar_model_file:
          if not "b16" in self.flux_file:
            print("Error: Flux file for B16 solar model should also be from B16")
            exit()
          if "b16_ags" in self.solar_model_file:
            fluxcols = 2
          else:
            fluxcols = 0
          fluxrows = {'pp':9, 'pep':10, 'hep':11, '7Be':12, '8B':13, '13N':14, '15O':15, '17F':16}
          fluxscale = {'pp':1e10, 'pep':1e8, 'hep':1e3, '7Be':1e9, '8B':1e6, '13N':1e8, '15O':1e8, '17F':1e6}
          distrow = 22
          radiuscol = 0
          densitycol = 2
          fractioncols = {'pp':4, 'pep':5, 'hep':6, '7Be':7, '8B':8, '13N':9, '15O':10, '17F':11}
        elif "bs2005" in self.solar_model_file:
          self.flux_file = self.solar_model_file
          fluxcols = {'hep':2, '8B':4}
          fluxrows = 6
          fluxscale = 1e10
          distrow = 27
          radiuscol = 0
          densitycol = 2
          fractioncols = {'8B':6, 'hep':12}
        elif "bp00" in self.solar_model_file:
          self.flux_file = self.solar_model_file
          fluxcols = {'hep':2, '8B':4}
          fluxrows = 25
          fluxscale = 1e10
          distrow = 29
          radiuscol = 0
          densitycol = 2
          fractioncols = {'8B':6, 'hep':12}
        elif fluxcols is None or fractioncols is None or fluxrows is None or distrow is None or radiuscol is None or densitycol is None:
          print("Error: Solar model not known to PEANUTS, you must provide the rows and columns for the fluxes and fractions")
          exit()

        # Make sure the shapes of optional variables are correct
        if isinstance(fluxcols, dict) and len(fluxcols) != len(fractioncols):
          print("Error: The number of selected fractions for the flux and fraction distributions must be the same.")
          exit()
        if isinstance(fluxrows, dict) and len(fluxrows) != len(fractioncols):
          print("Error: The number of selected fractions for the flux and fraction distributions must be the same.")
          exit()
        if isinstance(fluxcols, dict) and isinstance(fluxrows, dict) and len(fluxcols) != len(fluxrows):
          print("Error: The number of fraction rows and columns must be the same.")
          exit()
        if (not isinstance(fluxcols, dict) and not isinstance(fluxrows, dict)) or not isinstance(fractioncols, dict):
          print("Error: fluxcols (or fluxrows) and fractioncols should be dictionaries with fraction name and column number, e.g. {'8B': 2}, please change that.")
          exit()
        if not isinstance(fluxcols, dict):
          fluxcols = {frac : fluxcols for frac in fluxrows.keys()}
        if not isinstance(fluxrows, dict):
          fluxrows = {frac : fluxrows for frac in fluxcols.keys()}
        if fluxscale is not None:
          if isinstance(fluxscale, dict) and len(fluxscale) != len(fluxcols):
            print("Error: The number of scaling factors for the fluxes must match the number of fraction distributions.")
            exit()
          elif not isinstance(fluxscale, dict):
            fluxscale = {frac: fluxscale for frac in fluxcols.keys()}

        distcols = [radiuscol, densitycol] + list(fractioncols.values())
        distnames = ['radius', 'density_log_10'] + [fr + ' fraction' for fr in fractioncols.keys()]

        try:
          # Import fluxes
          self.fluxes = {}

          for key, val in fluxcols.items():
            self.fluxes[key] = f.read_csv(self.flux_file,
                                   usecols = [val],
                                   names = [key],
                                   delim_whitespace=True, skiprows=fluxrows[key], nrows=1, header=None)[key][0]
            if fluxscale is not None:
              self.fluxes[key] = self.fluxes[key]*fluxscale[key]


          # Import fraction data from solar model
          self.model = f.read_csv(self.solar_model_file,
                                  usecols = distcols,
                                  names = distnames,
                                  delim_whitespace=True, skiprows=distrow, header=None)

        except:
          print("Error! The solar model file provided does not exist or it is corrupted")
          exit()

        # Set useful variables
        self.rad = self.model['radius']
        self.dens = SolarDensity(10**self.model['density_log_10'])
        self.frac = {fr : self.model[fr + ' fraction'] for fr in fractioncols.keys()}

        # Import spectral shapes
        spectrum_files = {} if spectrum_files is None else spectrum_files
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

        return self.dens


    def fraction(self, name):
        """
        Returns the fraction of neutrinos for the requested column
        """

        return self.frac[name].to_numpy()

    def has_fraction(self, name):
       """
       Returns whether the solar model contains the given neutrino sample fraction
       """

       return name in self.frac.keys()

    def flux(self,name):
       """
       Returns the cumulative fluxes for each channel in cm^-2 s^-1
       """

       return self.fluxes[name]

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
@nb.njit
def Tei (th12, th13, DeltamSq21, DeltamSq3l, E, ne):
    """
    Tei(th12, th13, DeltamSq21, DeltamSq3l, E, ne) computes the weights composing an incoherent flux of
    neutrino mass eigenstates, for electron neutrinos produced in matter in the adiabatic approximation:
    - thij: the vacuum mixing angles in radians;
    - DeltamSq21. DeltamSq3l: the squared mass differences in units of eV^2;
    - E: the neutrino energy, in units of MeV;
    - ne: the electron density at production point, in units of mol/cm^3.
    See arXiv:1604.08167 and arXiv:1801.06514.
    """

    # Compute the mixing angles at neutrino production point
    th13m = th13_M(th12, th13, DeltamSq21, DeltamSq3l, E, ne)
    th12m = th12_M(th12, th13, DeltamSq21, DeltamSq3l, E, ne)

    # Compute and return the weights
    c13M = np.cos(th13m)
    s13M = np.sin(th13m)
    c12M = np.cos(th12m)
    s12M = np.sin(th12m)

    return [(c13M * c12M)**2, (c13M * s12M)**2, s13M**2]


# Compute flux of inchoerent mass eigenstates integrated over production point in the Sun
#@nb.njit
def solar_flux_mass (pmns, DeltamSq21, DeltamSq3l, E, radius_samples, density, fraction, adiabatic=True, exponential=False):
    """
    solar_flux_mass(th12, th13, DeltamSq21, DeltamSq3l, E, radius_samples, density, fraction) computes
    the weights of mass eigenstates composing the incoherent flux of solar neutrinos in the adiabatic
    approximation:
    - thij: the vacuum mixing angles in radians;
    - DeltamSq21, DeltamSq3l: the squared mass differences in units of eV^2;
    - E: the neutrino energy, in units of MeV;
    - radius_samples: a list of solar relative radius values, where density and fraction are sampled;
    - density: the list of electron densities at radii radius_samples, in units of mol/cm^3;
    - fraction: the relative fraction of neutrinos produced in the considered reaction,
    sampled at radius_samples.
    - adiabatic: whether to use the adiabatic approximation or not (def. True)
    - exponential: whether to use and exponential profile for the Sun density (def. False)
    """

    IntegratedFraction = np.trapz(y=fraction, x=radius_samples)

    # Use the adiabatic approximation for the neutrino evolution through the Sun
    if adiabatic:
      temp = Tei(pmns.theta12, pmns.theta13, DeltamSq21, DeltamSq3l, E, density.adiabatic_density)
    # Use the exponential profile for the neutrino evolution through the Sun
    elif exponential:
      nustate = [1,0,0]
      temp = ExponentialEvolution(nustate, density, DeltamSq21, DeltamSq3l, pmns, E, radius_samples*R_S, R_S)
      temp = np.array([np.array([np.square(np.abs(te)) for te in tem]) for tem in temp]).T
    else:
      print("Error: unknown solution for the evolution through the Sun, options are adiabatic or exponential")
      exit()

    temp = [temp[i]*fraction for i in range(len(temp))]

    Te = [np.trapz(y=temp[i], x = radius_samples) / IntegratedFraction
          for i in range(3)]

    return np.array(Te)

# Compute the flavour probabilities for the solar neutrino flux
def Psolar (pmns, DeltamSq21, DeltamSq3l, E, radius_samples, density, fraction, adiabatic=True, exponential=False):
    """
    Psolar(pmns, DeltamSq21, DeltamSq3l, E, radius_samples, density, fraction) computes the
    flavour probabilities of observing a solar neutrino as a given flavour.
    The function returns a list for the probabilities [P(electron), P(muon), P(tau)].
    The neutrino has energy E and is produced in a specific reaction:
    - pmns: the PMNs matrix
    - DeltamSq21, DeltamSq3l: the vacuum squared mass differences in units of eV^2;
    - E: the neutrino energy, in units of MeV;
    - radius_samples: a list of solar relative radius values where density and fraction are sampled;
    - density: the list of electron densities at radii radius_samples, in units of mol/cm^3;
    - fraction: the relative fraction of neutrinos produced in the considered reaction,
    sampled at radius_samples.
    """

    # Compute the weights in the uncoherent solar flux of mass eigenstates
    Tei = np.array(solar_flux_mass(pmns, DeltamSq21, DeltamSq3l, E, radius_samples, density, fraction, adiabatic=adiabatic, exponential=exponential))

    # Compute the probabilities that a mass eigenstate is observed as a given flavour
    #P_i_to_a = np.square(np.abs(PMNS(th12, th13, th23, -d))) # TODO: Why negative -d? ANSWER: because the mass eigenstates are given by
    # the conjugate of the PMNS mixing matrix, thus we can simply invert the CP phase instead of taking the conjugate of the full matrix
    P_i_to_a = np.square(np.abs(pmns.conjugate()))

    # Multiply probabilities by weights, and return the result
    return np.array(np.dot(P_i_to_a, Tei))
