#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

import os
import numpy as np
from optparse import OptionParser

import src.files as f
from src.utils import print_banner, print_inputs
from src.pmns import PMNS
from src.solar import SolarModel, solar_flux_mass, Psolar

mainfilename = 'run_prob_sun'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-s", "--solar", help ="Add custom solar model", action='store', dest="solar", default="")
(options, args) = parser.parse_args()
if len(args) < 3 :
  print('Wrong number of arguments \n\
        \n\
Usage: python '+mainfilename+'.py <in_file> <energy> <nu_fraction>\n\
       <in_file>                   Input file\n\
       <energy>                    Energy\n\
       <nu_fraction>               Neutrino fraction sample\n\
\n\
Options:\n\
       -h, --help                    Show this help message and exit\n\
       -v, --verbose                 Print debug output\n\
       -s, --solar                   Add custom solar model')

  exit()

# Read the input files
path = os.path.dirname(os.path.realpath( __file__ ))
slha_file = args[0]
solar_file = path + '/Data/bs2005agsopflux.csv' if options.solar == "" else options.solar

# Import data from solar model
solar_model = SolarModel(solar_file)

# Get arguments
E = float(args[1])
nu_fraction = args[2]
if not solar_model.has_fraction(nu_fraction):
   print("Error: The fraction ", nu_fraction, " does not exist in the solar model.")
   exit()


# Read example slha file and fill PMNS matrix
nu_params = f.read_slha(slha_file)
th12 = nu_params['theta12']
th13 = nu_params['theta13']
th23 = nu_params['theta23']
d = nu_params['delta']
pmns = PMNS(th12, th13, th23, d)

DeltamSq21 = nu_params['dm21']
DeltamSq31 = nu_params['dm31']

# Print program banner and inputs
print_banner()
print_inputs("solar", options, pmns, DeltamSq21, DeltamSq31, E, nu_fraction)

# Compute probability for the given sample fraction and energy
print("Running PEANUTS...")
prob = Psolar(pmns, DeltamSq21, DeltamSq31, E, solar_model.radius, solar_model.density, solar_model.fraction[nu_fraction])

# Print results
print()
print("Probability to oscillate to an electron neutrino : ", prob[0])
print("Probability to oscillate to a muon neutrino      : ", prob[1])
print("Probability to oscillate to a tau neutrino       : ", prob[2])

