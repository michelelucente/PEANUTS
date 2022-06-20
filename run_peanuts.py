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

import src.files as f
from src.utils import get_comma_separated_floats, print_banner, print_inputs
from src.pmns import PMNS
from src.solar import SolarModel, solar_flux_mass, Psolar
from src.earth import EarthDensity, Pearth
from src.time_average import NadirExposure

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

  # Import data from solar model
  solar_model = SolarModel(settings.solar_file)

if settings.earth:

  # If the solar probabilities will not be computed before, take state from settings
  if not settings.solar:
    nustate = settings.nustate
    basis = settings.basis

  # Earth density
  earth_density = EarthDensity(settings.density_file)


# Loop over energy values
for e in settings.energy:

  out = {}

  if settings.solar:

    # Add flux if requested
    if settings.flux:
      out["flux"] = solar_model.flux(settings.fraction)

    # Compute probability for the given sample fraction and energy
    if settings.probabilities:
      out["solar"] = Psolar(settings.pmns, settings.dm21, settings.dm31, e, solar_model.radius, solar_model.density, solar_model.fraction[settings.fraction])

    # Add undistorted or distorted spectrum if requested
    if settings.undistorted_spectrum: 
      out['spectrum'] = solar_model.spectrum(settings.fraction, energy=e)
    elif settings.distorted_spectrum:
      out['spectrum'] = solar_model.spectrum(settings.fraction, energy=e) * Psolar(settings.pmns, settings.dm21, settings.dm31, e, solar_model.radius, solar_model.density, solar_model.fraction[settings.fraction])

    # If the earth propbabilities are to be computed, we need the mass weights
    if settings.earth:
      mass_weights = solar_flux_mass(settings.pmns.theta12, settings.pmns.theta13, settings.dm21, settings.dm31, e, 
                                     solar_model.radius, solar_model.density, solar_model.fraction[settings.fraction])

  if settings.earth:

    # If the solar probabilities were computed before, use the precomputed mass weights as neutrino state
    if settings.solar:
      nustate = mass_weights
      basis = "mass"

    # If the latitude is provided compute exposure 
    if settings.exposure:
      exposure = NadirExposure(radians(settings.latitude), normalized=True, d1=settings.exposure_time[0], d2=settings.exposure_time[1], ns=settings.exposure_samples)

      out["earth"] = 0
      deta = pi/settings.exposure_samples
      for eta, exp in exposure:
        out["earth"] += Pearth(nustate, earth_density, settings.pmns, settings.dm21, settings.dm31, e, eta, settings.depth, mode=settings.evolution, basis=basis) * exp * deta

    else:
      # Compute probability of survival after propagation through Earth 
      out["earth"] = Pearth(nustate, earth_density, settings.pmns, settings.dm21, settings.dm31, e, settings.eta, settings.depth, mode=settings.evolution, basis=basis)

  # Append to list
  outs.append(out)

# Print results
f.output(settings, outs)


