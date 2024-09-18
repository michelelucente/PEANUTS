#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import os
import numpy as np
from optparse import OptionParser
from math import pi, radians

import peanuts.files as f
from peanuts.utils import get_comma_separated_floats, print_banner, print_inputs
from peanuts.pmns import PMNS
from peanuts.vacuum import Pvacuum, vacuum_evolved_state
from peanuts.solar import SolarModel, solar_flux_mass, Psolar
from peanuts.atmosphere import AtmosphereDensity, Patmosphere, evolved_state_atmosphere, Hmax
from peanuts.earth import EarthDensity, Pearth, Pearth_integrated, evolved_state

mainfilename = 'run_peanuts'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-f", help="Input file", type='string', action='store', dest="in_file")
(options, args) = parser.parse_args()
if len(args) > 0 or options.in_file == None:
  if len(args) > 0:
    print("Error: Wrong number of arguments")
  if options.in_file == None:
    print('Error: Missing input file')
  print('\n\
Usage: python '+mainfilename+'.py -f <in_file>\n\
       <in_file>                   Input file\n\
\n\
Options:\n\
       -h, --help                    Show this help message and exit\n\
       -v, --verbose                 Print debug output')
  exit()

# Import input file
settings = f.read_yaml(options.in_file)

# Print program banner and inputs
print_banner()
print_inputs(settings)
print("Running PEANUTS...")

# List of probabilities
outs = []

# Initialize values
if settings.solar:

  # Import data from solar model and spectrum files
  solar_model = SolarModel(solar_model_file=settings.solar_file, flux_file=settings.flux_file, spectrum_files=settings.spectra,
                           fluxrows=settings.fluxrows, fluxcols=settings.fluxcols, distrow=settings.distrow,
                           radiuscol=settings.radiuscol, densitycol=settings.densitycol, fractioncols=settings.fractioncols)

# If the solar probabilities will not be computed before, take state from settings
else:
  nustate = settings.nustate
  massbasis = True if settings.basis=="mass" else False

# Import atmospheric density, fixed for now
if settings.atmosphere:
  atmos_density = AtmosphereDensity()

# Import earth density
if settings.earth:
  earth_density = EarthDensity(density_file=settings.density_file,custom_density=settings.custom_density,tabulated_density=settings.tabulated_density)


# Loop over energy values
for param in settings.scan:

  out = {}

  pmns = PMNS(param.theta12, param.theta13, param.theta23, param.delta)

  if settings.vacuum:

    if settings.probabilities:
      # Calculate vacuum probabilities
      out["vacuum"] = Pvacuum(nustate, pmns, param.dm21, param.dm3l, param.energy, param.baseline, antinu=settings.antinu, massbasis=massbasis)

    if settings.evolved_state:
      out["evolved_state"] = vacuum_evolved_state(nustate, pmns, param.dm21, param.dm3l, param.energy, settings.baseline, antinu=settings.antinu)

  if settings.solar:

    # Add flux if requested
    if settings.flux:
      out["flux"] = solar_model.flux(settings.fraction)

    # Compute probability for the given sample fraction and energy
    if settings.solar_probabilities:
      out["solar"] = Psolar(pmns, param.dm21, param.dm3l, param.energy, solar_model.radius(), solar_model.density(), solar_model.fraction(settings.fraction),
                            adiabatic=settings.adiabatic, exponential=settings.exponential)

    # Add undistorted or distorted spectrum if requested
    if settings.undistorted_spectrum:
      out['spectrum'] = solar_model.spectrum(settings.fraction, energy=param.energy)
    elif settings.distorted_spectrum:
      if "solar" in out:
        out["spectrum"] = solar_model.spectrum(settings.fraction, energy=param.energy) * out["solar"]
      else:
        out['spectrum'] = solar_model.spectrum(settings.fraction, energy=param.energy) * Psolar(pmns, param.dm21, param.dm3l, param.energy, solar_model.radius(), solar_model.density(),
                                               solar_model.fraction(settings.fraction), adiabatic=settings.adiabatic, exponential=settings.exponential)

    # If the earth or atmospheric propbabilities are to be computed, or the solar weights are requested, compute the solar weights
    if settings.earth or settings.atmosphere or settings.solar_weights:
      mass_weights = solar_flux_mass(pmns, param.dm21, param.dm3l, param.energy,
                                     solar_model.radius(), solar_model.density(), solar_model.fraction(settings.fraction),
                                     adiabatic=settings.adiabatic, exponential=settings.exponential)
      if settings.solar_weights:
        out["solar_weights"] = mass_weights

    # If the earth or atmospheric probabilities are to be computed, the input state is a incoherent combination of mass eigenstates
    # So the probabilities need to be computed for all mass eigenstates
    if settings.earth or settings.atmosphere:
      nustate = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=complex)
      massbasis = True

  if settings.atmosphere:

    # If Earth oscillations are enabled, eta refers to the angle with the detector, so we need to pass the depth in order to compute eta_prime
    if settings.earth:
      depth = settings.depth;
    else:
      # If the nadir angle is < pi/2 (night), then the neutrino path crosses the Earth, so atmospheric oscillations are not enough
      if param.eta < pi/2:
        print("Error: Neutrinos with angles eta < pi/2 cross the Earth, so Earth oscillations should be enabled.\
               Please add an Earth node to your yaml file, with depth = 0 for a detector on the surface")
        exit()
      depth = 0

    # Compute the probability on Earth's surface
    if settings.atm_probabilities:
      out["atmosphere"] = Patmosphere(nustate, atmos_density, param.dm21, param.dm3l, pmns, param.energy, param.eta, param.height, depth=depth,
                                      massbasis=massbasis, antinu=settings.antinu)

      if settings.solar:
        out["atmosphere"] = np.dot(out["atmosphere"], mass_weights)

    # If the earth probabilities are to be computed, or the evolved state is requested, calculate the evolved state
    if settings.atm_evolved_state or settings.earth and not settings.exposure:
       nustate = evolved_state_atmosphere(nustate, atmos_density, param.dm21, param.dm3l, pmns, param.energy, param.eta, param.height, depth=depth,
                                          massbasis=massbasis, antinu=settings.antinu)
       massbasis = False

       if settings.atm_evolved_state:
         out["atm_evolved_state"] = nustate


  if settings.earth:

    # If the exposure is requested compute probability, integrating over exposure
    if settings.exposure:
      if settings.atmosphere:
        height = param.height
        height_file = settings.height_file
      else:
        height = Hmax
        height_file  = None

      out["earth"] = Pearth_integrated(nustate, earth_density, pmns, param.dm21, param.dm3l, param.energy, settings.depth,
                                       height=height, mode=settings.evolution, antinu=settings.antinu,
                                       lam=radians(settings.latitude), d1=settings.exposure_time[0], d2=settings.exposure_time[1],
                                       normalized=settings.exposure_normalized, ns=settings.exposure_samples,
                                       angle_file=settings.exposure_file, angle=settings.exposure_angle, height_file=height_file,
                                       solar=settings.solar, atmosphere=settings.atmosphere)

    else:
      # Compute probability of survival after propagation through Earth
      out["earth"] = Pearth(nustate, earth_density, pmns, param.dm21, param.dm3l, param.energy, param.eta, settings.depth,
                           mode=settings.evolution, massbasis=massbasis, antinu=settings.antinu)
    if settings.solar:
      out["earth"] = np.dot(out["earth"], mass_weights)

    # If the evolved state is requested, compute that too
    if settings.earth_evolved_state:
      if massbasis:
        print("Warning: The output evolved state is always a flavour eigenstate, even when the input state is a mass eigenstate.")
      out["earth_evolved_state"] = evolved_state(nustate, earth_density, pmns, param.dm21, param.dm3l, param.energy, param.eta, settings.depth, massbasis=massbasis,
                                             mode=settings.evolution, antinu=settings.antinu)

  # Append to list
  outs.append(out)

# Print results
f.output(settings, outs)


