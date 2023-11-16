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
from peanuts.solar import SolarModel, solar_flux_mass, Psolar
from peanuts.earth import EarthDensity, Pearth, Pearth_integrated, evolved_state
from peanuts.time_average import NadirExposure

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

if settings.earth:

  # If the solar probabilities will not be computed before, take state from settings
  if not settings.solar:
    nustate = settings.nustate
    massbasis = True if settings.basis=="mass" else False

  # Earth density
  earth_density = EarthDensity(density_file=settings.density_file,custom_density=settings.custom_density)


# Loop over energy values
for param in settings.scan:

  out = {}

  pmns = PMNS(param.theta12, param.theta13, param.theta23, param.delta)

  if settings.solar:

    # Add flux if requested
    if settings.flux:
      out["flux"] = solar_model.flux(settings.fraction)

    # Compute probability for the given sample fraction and energy
    if settings.probabilities:
      out["solar"] = Psolar(pmns, param.dm21, param.dm3l, param.energy, solar_model.radius(), solar_model.density(), solar_model.fraction(settings.fraction))

    # Add undistorted or distorted spectrum if requested
    if settings.undistorted_spectrum:
      out['spectrum'] = solar_model.spectrum(settings.fraction, energy=param.energy)
    elif settings.distorted_spectrum:
      out['spectrum'] = solar_model.spectrum(settings.fraction, energy=param.energy) * Psolar(pmns, param.dm21, param.dm3l, param.energy, solar_model.radius(), solar_model.density(), solar_model.fraction(settings.fraction))

    # If the earth propbabilities are to be computed, we need the mass weights
    if settings.earth:
      mass_weights = solar_flux_mass(pmns.theta12, pmns.theta13, param.dm21, param.dm3l, param.energy,
                                     solar_model.radius(), solar_model.density(), solar_model.fraction(settings.fraction))

  if settings.earth:

    # If the solar probabilities were computed before, use the precomputed mass weights as neutrino state
    if settings.solar:
      nustate = np.array(mass_weights, dtype=complex)
      massbasis = True

    # If the latitude is provided compute probability integrated over exposure
    if settings.exposure:
      out["earth"] = Pearth_integrated(nustate, earth_density, pmns, param.dm21, param.dm3l, param.energy, settings.depth, mode=settings.evolution, antinu=settings.antinu,
                                       lam=radians(settings.latitude), d1=settings.exposure_time[0], d2=settings.exposure_time[1],
                                       normalized=settings.exposure_normalized, ns=settings.exposure_samples,
                                       from_file=settings.exposure_file, angle=settings.exposure_angle)

    else:
      # Compute probability of survival after propagation through Earth
      out["earth"] = Pearth(nustate, earth_density, pmns, param.dm21, param.dm3l, param.energy, param.eta, settings.depth, mode=settings.evolution, massbasis=massbasis, antinu=settings.antinu)

    # If the evolved state is requested, compute that too
    if settings.evolved_state:
      if massbasis:
        print("Warning: The evolved state can only be computed from a neutrino state in the flavour basis, so it will not be provided")
        settings.evolved_state = False
      else:
        out["evolved_state"] = evolved_state(nustate, earth_density, pmns, param.dm21, param.dm3l, param.energy, param.eta, settings.depth, mode=settings.evolution, antinu=settings.antinu)

  # Append to list
  outs.append(out)

# Print results
f.output(settings, outs)


