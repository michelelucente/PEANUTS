#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import os
import numpy as np
from optparse import OptionParser

import src.files as f
from src.utils import print_banner, print_inputs
from src.settings import Settings
from src.pmns import PMNS
from src.solar import SolarModel, solar_flux_mass, Psolar

mainfilename = 'run_prob_sun'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-s", "--solar", help ="Add custom solar model", action='store', dest="solar", default="")
parser.add_option("-i", "--in_slha", help="SLHA input file", action='store', dest='in_slha', default="")
(options, args) = parser.parse_args()
if len(args) < 2 or (options.in_slha == "" and len(args) != 8):
  print('Wrong number of arguments \n\
        \n\
Usage: ./'+mainfilename+'.py <energy> <fraction> [<th12> <th13> <th23> <delta> <md21> <md3l>]\n\
       <energy>                    Energy\n\
       <fraction>                  Neutrino fraction sample\n\
       <th12>                      Mixing angle theta_12\n\
       <th13>                      Mixing angle theta_13\n\
       <th23>                      Mixing angle theta_23\n\
       <delta>                     CP phase delta\n\
       <md21>                      Mass splitting m^2_{21}\n\
       <md3l>                      Mass splitting m^2_{3l} (l=1 NO, l=2 IO)\n\
\n\
Options:\n\
       -h, --help                  Show this help message and exit\n\
       -v, --verbose               Print debug output\n\
       -s, --solar <solar_file>    Add custom solar model\n\
       -i, --in_slha <slha_file>   SLHA input file for neutrino parameters')

  exit()

# Read the solar model file
path = os.path.dirname(os.path.realpath( __file__ ))
solar_file = path + '/Data/bs2005agsopflux.csv' if options.solar == "" else options.solar

# Import data from solar model
solar_model = SolarModel(solar_file)

# Get arguments
E = float(args[0])
fraction = args[1]
if not solar_model.has_fraction(fraction):
   print("Error: The fraction ", fraction, " does not exist in the solar model.")
   exit()

# If the -i/--in_slha option is given, read the slha file
if options.in_slha != "":

  # If pyslha has not been imported throw error
  if not f.with_slha:
    print("Error: The module `pyslha` is needed to use SLHA input, please install it.")
    exit()

  # Read slha file
  slha_file = options.in_slha

  # Read example slha file and fill PMNS matrix
  nu_params = f.read_slha(slha_file)
  th12 = nu_params['theta12']
  th13 = nu_params['theta13']
  th23 = nu_params['theta23']
  d = nu_params['delta']
  pmns = PMNS(th12, th13, th23, d)

  DeltamSq21 = nu_params['dm21']
  DeltamSq3l = nu_params['dm3l']

# Otherwise, the parameters are given as arguments
else:

  th12 = float(args[2])
  th13 = float(args[3])
  th23 = float(args[4])
  d = float(args[5])
  pmns = PMNS(th12, th13, th23, d)

  DeltamSq21 = float(args[6])
  DeltamSq3l = float(args[7])

# Print program banner and inputs
print_banner()
print_inputs(Settings(pmns, DeltamSq21, DeltamSq3l, E, fraction, options))
print("Running PEANUTS...")

# Compute probability for the given sample fraction and energy
prob = Psolar(pmns, DeltamSq21, DeltamSq3l, E, solar_model.radius(), solar_model.density(), solar_model.fraction(fraction))

# Print results
print()
print("Probability to oscillate to an electron neutrino : ", prob[0])
print("Probability to oscillate to a muon neutrino      : ", prob[1])
print("Probability to oscillate to a tau neutrino       : ", prob[2])
print()
