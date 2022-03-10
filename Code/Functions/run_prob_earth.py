#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

import numpy as np
#from math import sqrt # TODO Not needed right now, re-add if needed or delete
#from numpy import arctan, arcsin # TODO: Not needed right now, re-add if needed or delete
from optparse import OptionParser

import src.files as f
from src.pmns import PMNS
from src.earth import EarthDensity, Pearth, Pearth_analytical

mainfilename = 'run_prob_earth'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-d", "--density", help ="Add custom earth density profile", action='store', dest="density", default="")
parser.add_option("-a", "--analytical", help="Perform analytical evolution", action='store_true', dest="analytical", default=False)
(options, args) = parser.parse_args()
if len(args) < 4 :
  print('Wrong number of arguments \n\
        \n\
Usage: python '+mainfilename+'.py <in_file> <energy> <eta> <depth>\n\
       <in_file>                   Input file\n\
       <energy>                    Energy of neutrinos\n\
       <eta>                       Nadir angle of the incident neutrinos\n\
       <depth>                     Depth of detector location below Earth\'s surface\n\
\n\
Options:\n\
       -h, --help                    Show this help message and exit\n\
       -v, --verbose                 Print debug output\n\
       -d, --density                 Add custom earth density profile')

  exit()

# Read the input files
slha_file = args[0]
density_file = './Data/Earth_Density.csv' if options.density == '' else options.density

# Get parameters
E = float(args[1])
eta = float(args[2])
H = float(args[3])

# Read example slha file and fill PMNS matrix
nu_params = f.read_slha(slha_file)
th12 = nu_params['theta12']
th13 = nu_params['theta13']
th23 = nu_params['theta23']
d = nu_params['delta']
pmns = PMNS(th12, th13, th23, d)

DeltamSq21 = nu_params['dm21']
DeltamSq31 = nu_params['dm31']

# Earth density
earth_density = EarthDensity(density_file)

# Earth regeneration

# Check if analytical solution was requested
if options.analytical:
  print("Using analytical computation")
  prob = Pearth_analytical(earth_density, pmns, DeltamSq21, DeltamSq31, eta, E, H)

# Otherwise use numerical solution
else:
 print("Using numerical computation")
 prob = Pearth(earth_density, pmns, DeltamSq21, DeltamSq31, eta, E, H)


print("Probability of survival of electron neutrinos with nadir angle eta =", eta, "and energy E =", E, "at ", H, "meters below the surface of the Earth is", prob)
