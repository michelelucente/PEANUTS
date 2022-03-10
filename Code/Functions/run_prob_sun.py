#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 10 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

import numpy as np
from optparse import OptionParser

import src.files as f
from src.pmns import PMNS
from src.solar import SolarModel, Psolar

mainfilename = 'run_prob_sun'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-s", "--solar", help ="Add custom solar model", action='store', dest="solar", default="")
parser.add_option("-f", "--flavour", help="Neutrino flavour", action='store', dest='flav', default="")
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
slha_file = args[0]
solar_file = './Data/bs2005agsopflux.csv' if options.solar == "" else options.solar

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

# Compute probability for the given sample fraction and energy
prob = Psolar(pmns, DeltamSq21, DeltamSq31, E, solar_model.radius, solar_model.density, solar_model.fraction[nu_fraction])

# TODO: Which unit do we expect the energy?
if options.flav == '':
  print("Probabilities of neutrinos of sample fraction", nu_fraction, "with energy E =", E, "at Sun exit is", prob)
elif options.flav == 'e':
  print("Probabilities of electron neutrinos of sample fraction", nu_fraction, "with energy E =", E, "at Sun exit is", prob[0])
elif options.flav == 'mu':
  print("Probabilities of muon neutrinos of sample fraction", nu_fraction, "with energy E =", E, "at Sun exit is", prob[1])
elif options.flav == 'tau':
  print("Probabilities of tau neutrinos of sample fraction", nu_fraction, "with energy E =", E, "at Sun exit is", prob[2])
else:
  print("Error: Unknown neutrino flavour")
