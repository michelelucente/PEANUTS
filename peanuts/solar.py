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
from peanuts.matter_mixing import th13_M, th12_M, th12_M_given_th13_M
from peanuts.pmns import PMNS

import peanuts.files as f

# Column layouts known to SolarModel
SOLAR_MODEL_LAYOUTS = ("b16", "bs2005", "bp00", "sf3")

# Compositions of the B23 / SF-III standard solar models (Zenodo 10.5281/zenodo.10822316)
SF3_COMPOSITIONS = ("GS98", "AGSS09", "C11", "AAG21", "MB22m", "MB22p")

# Neutrino sources in the order of the SF3 columns nu_<source> and of the SF3 flux table
SF3_SOURCES = ("pp", "pep", "hep", "7Be", "8B", "13N", "15O", "17F")


def detect_layout(solar_model_file):
    """
    Returns the column layout of a solar model file from its name, or None if it is not known.
    The B23 / SF-III profiles are recognised by their file name, struct+nu_SF3_<composition>.dat;
    the other layouts by the substrings "b16", "bs2005", "bp00" of the path, in this order.
    """

    if os.path.basename(solar_model_file).lower().startswith("struct+nu_sf3_"):
      return "sf3"
    for layout in ("b16", "bs2005", "bp00"):
      if layout in solar_model_file:
        return layout
    return None


def sf3_flux_file(solar_model_file):
    """
    Returns the flux file fluxes_SF3_<composition>.dat that accompanies the SF3 profile file
    struct+nu_SF3_<composition>.dat in the same directory, or None if the name does not follow
    this convention or the file does not exist.
    """

    base = os.path.basename(solar_model_file)
    if not base.lower().startswith("struct+nu_"):
      return None
    candidate = os.path.join(os.path.dirname(solar_model_file), "fluxes_" + base[len("struct+nu_"):])
    return candidate if os.path.isfile(candidate) else None


def _header_tokens(filename, row):
    """
    Returns the whitespace-separated tokens of line number `row` (from 0) of a text file.
    """

    with open(filename) as fh:
      for i, line in enumerate(fh):
        if i == row:
          return line.split()
    return []


def sf3_columns(solar_model_file, flux_file):
    """
    Returns (fluxcols, fluxrows, fluxscale, distrow, radiuscol, densitycol, fractioncols) of the
    B23 / SF-III layout, with the column numbers read from the headers of the files:
    - struct+nu_SF3_<comp>.dat: one header line starting with "#", then one row per radius;
      columns R_sun (radius in solar units), nu_<source> (production fraction per unit R_sun,
      normalised to unit integral), log_ne (log10 of the electron density in N_A cm^-3, i.e. mol/cm^3);
    - fluxes_SF3_<comp>.dat: title line, line of source names, line of fluxes in cm^-2 s^-1.
    Exits with an error if a column is missing.
    """

    if not os.path.isfile(solar_model_file) or not os.path.isfile(flux_file):
      print("Error! The solar model file provided does not exist or it is corrupted")
      exit()
    if not os.path.basename(flux_file).lower().startswith("fluxes_sf3_"):
      print("Error: Flux file for SF3 solar model should also be from SF3 (fluxes_SF3_<composition>.dat)")
      exit()
    comp_model = os.path.basename(solar_model_file)[len("struct+nu_SF3_"):].rsplit(".", 1)[0]
    comp_flux = os.path.basename(flux_file)[len("fluxes_SF3_"):].rsplit(".", 1)[0]
    if comp_model.lower() != comp_flux.lower():
      print("Warning: SF3 solar model", os.path.basename(solar_model_file), "is used with the fluxes of", os.path.basename(flux_file))

    # Profile header: "#" followed by one name per column
    header = _header_tokens(solar_model_file, 0)
    if not header or header[0] != "#":
      print("Error: the SF3 solar model file", solar_model_file, "does not start with its header line")
      exit()
    names = header[1:]
    needed = ["R_sun", "log_ne"] + ["nu_" + s for s in SF3_SOURCES]
    missing = [n for n in needed if n not in names]
    if missing:
      print("Error: the SF3 solar model file", solar_model_file, "lacks the columns", ", ".join(missing))
      exit()

    # Flux table: the source names on line 1, the fluxes on line 2 (followed by the label "Flux")
    flux_names = _header_tokens(flux_file, 1)
    flux_values = _header_tokens(flux_file, 2)
    if flux_values[-1:] != ["Flux"] or any(s not in flux_names for s in SF3_SOURCES):
      print("Error: the SF3 flux file", flux_file, "does not have the expected layout")
      exit()

    fluxcols = {s: flux_names.index(s) for s in SF3_SOURCES}
    fluxrows = 2
    fluxscale = None
    distrow = 1
    radiuscol = names.index("R_sun")
    densitycol = names.index("log_ne")
    fractioncols = {s: names.index("nu_" + s) for s in SF3_SOURCES}
    return fluxcols, fluxrows, fluxscale, distrow, radiuscol, densitycol, fractioncols


class SolarModel:
    """"
    Class containing the info of the solar model
    """

    def __init__(self, solar_model_file=None, flux_file=None, spectrum_files=None, fluxrows=None, fluxcols=None, fluxscale=None, distrow=None, radiuscol=None, densitycol=None, fractioncols=None, layout=None):
        """
        Constructor of the solar model.
        Reads the solar model file and fills useful variables.
        The column layout of the file is detected from its name ("b16", "bs2005", "bp00", and the
        B23/SF-III "struct+nu_SF3_<composition>.dat") or selected with layout = "b16", "bs2005",
        "bp00" or "sf3"; any other file needs the explicit rows and columns.
        """

        # Set file name
        path = os.path.dirname(os.path.realpath( __file__ ))
        # Default solar model file is the B16 from Vinyoles et al 2016, https://www.ice.csic.es/personal/aldos/Solar_Data.html
        self.solar_model_file = path + "/../Data/nudistr_b16_agss09.dat" if solar_model_file == None else solar_model_file
        # Default flux file is the B16 from Vinyoles et al 2016, https://www.ice.csic.es/personal/aldos/Solar_Data.html
        self.flux_file = path + "/../Data/fluxes_b16.dat" if (flux_file == None and "b16" in self.solar_model_file) else (solar_model_file if flux_file == None else flux_file)

        # Column layout: explicit, or detected from the file name
        if layout is not None and layout not in SOLAR_MODEL_LAYOUTS:
          print("Error: unknown solar model layout", layout, "- known layouts:", ", ".join(SOLAR_MODEL_LAYOUTS))
          exit()
        self.layout = detect_layout(self.solar_model_file) if layout is None else layout
        if self.layout == "b16" and flux_file == None:
          # Explicit B16 layout for a file whose name does not contain "b16": default B16 fluxes as well
          self.flux_file = path + "/../Data/fluxes_b16.dat"

        # Format for the various spectra are layered differently
        if self.layout == "sf3":
          # B23 / SF-III standard solar models (Herrera & Serenelli, ICE-CSIC), Zenodo 10.5281/zenodo.10822316:
          # struct+nu_SF3_<comp>.dat (radial profiles) and fluxes_SF3_<comp>.dat (fluxes) in the same directory
          if flux_file == None:
            self.flux_file = sf3_flux_file(self.solar_model_file)
            if self.flux_file is None:
              print("Error: the flux file of an SF3 solar model cannot be inferred from the name", self.solar_model_file, "- provide flux_file")
              exit()
          fluxcols, fluxrows, fluxscale, distrow, radiuscol, densitycol, fractioncols = sf3_columns(self.solar_model_file, self.flux_file)
        elif self.layout == "b16":
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
        elif self.layout == "bs2005":
          self.flux_file = self.solar_model_file
          fluxcols = {'pp':0, 'pep':1, 'hep':2, '7Be':3, '8B':4, '13N':5, '15O':6, '17F':7}
          fluxrows = 6
          fluxscale = 1e10
          distrow = 27
          radiuscol = 0
          densitycol = 2
          fractioncols = {'pp':5, '8B':6, '13N':7, '15O':8, '17F':9, '7Be':10, 'pep':11, 'hep':12}
        elif self.layout == "bp00":
          self.flux_file = self.solar_model_file
          fluxcols = {'pp':0, 'pep':1, 'hep':2, '7Be':3, '8B':4, '13N':5, '15O':6, '17F':7}
          fluxrows = 25
          fluxscale = 1e10
          distrow = 29
          radiuscol = 0
          densitycol = 2
          fractioncols = {'pp':5, '8B':6, '13N':7, '15O':8, '17F':9, '7Be':10, 'pep':11, 'hep':12}
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
        # pandas assigns the names to the selected columns in file order, so both lists are put in that order
        # (the B16, BS05 and BP00 layouts already are; the SF3 layout has the density after the fractions)
        distcols, distnames = map(list, zip(*sorted(zip(distcols, distnames))))

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
        self.dens =  10**self.model['density_log_10']
        self.frac = {fr : self.model[fr + ' fraction'] for fr in fractioncols.keys()}

        # Import spectral shapes
        spectrum_files = {} if spectrum_files is None else spectrum_files
        spectrum_files["8B"] = path + "/../Data/8B_shape_Ortiz_et_al.csv" if "8B" not in spectrum_files else spectrum_files['8B']
        spectrum_files["hep"] = path + "/../Data/hep_shape.csv" if "hep" not in spectrum_files else spectrum_files['hep']
        spectrum_files["pp"] = path + "/../Data/pp_shape.csv" if "pp" not in spectrum_files else spectrum_files['pp']
        spectrum_files["17F"] = path + "/../Data/f17_shape.csv" if "17F" not in spectrum_files else spectrum_files['17F']
        spectrum_files["7Be"] = spectrum_files["7Beground"] = path + "/../Data/be7ground_shape.csv" if "7Beground" not in spectrum_files else spectrum_files['7Beground']
        spectrum_files["7Beexcited"] = path + "/../Data/be7excited_shape.csv" if "7Beexcited" not in spectrum_files else spectrum_files['7Beexcited']
        spectrum_files["13N"] = path + "/../Data/n13_shape.csv" if "13N" not in spectrum_files else spectrum_files['13N']
        spectrum_files["15O"] = path + "/../Data/o15_shape.csv" if "15O" not in spectrum_files else spectrum_files['15O']

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
    th12m = th12_M_given_th13_M(th12, th13, th13m, DeltamSq21, DeltamSq3l, E, ne)

    # Compute and return the weights
    c13M = np.cos(th13m)
    s13M = np.sin(th13m)
    c12M = np.cos(th12m)
    s12M = np.sin(th12m)

    return [(c13M * c12M)**2, (c13M * s12M)**2, s13M**2]


# Compute flux of inchoerent mass eigenstates integrated over production point in the Sun
@nb.njit
def solar_flux_mass (th12, th13, DeltamSq21, DeltamSq3l, E, radius_samples, density, fraction):
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
    """

    IntegratedFraction = np.trapz(y=fraction, x=radius_samples)

    temp = Tei(th12, th13, DeltamSq21, DeltamSq3l, E, density)
    temp = [temp[i]*fraction for i in range(len(temp))]

    Te = [np.trapz(y=temp[i], x = radius_samples) / IntegratedFraction
          for i in range(3)]

    return np.array(Te)

# Compute the flavour probabilities for the solar neutrino flux
def Psolar (pmns, DeltamSq21, DeltamSq3l, E, radius_samples, density, fraction):
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
    Tei = np.array(solar_flux_mass(pmns.theta12, pmns.theta13, DeltamSq21, DeltamSq3l, E, radius_samples, density, fraction))

    # Compute the probabilities that a mass eigenstate is observed as a given flavour
    #P_i_to_a = np.square(np.abs(PMNS(th12, th13, th23, -d))) # TODO: Why negative -d? ANSWER: because the mass eigenstates are given by
    # the conjugate of the PMNS mixing matrix, thus we can simply invert the CP phase instead of taking the conjugate of the full matrix
    P_i_to_a = np.square(np.abs(pmns.conjugate()))

    # Multiply probabilities by weights, and return the result
    return np.array(np.dot(P_i_to_a, Tei))
