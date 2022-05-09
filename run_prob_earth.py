#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

import os
import numpy as np
from optparse import OptionParser

import src.files as f
from src.utils import get_comma_separated_floats, print_banner, print_inputs
from src.pmns import PMNS
from src.solar import SolarModel, solar_flux_mass
from src.earth import EarthDensity, Pearth

mainfilename = 'run_prob_earth'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-d", "--density", help ="Add custom earth density profile", action='store', dest="density", default="")
parser.add_option("-a", "--analytical", help="Perform analytical evolution", action='store_true', dest="analytical", default=True)
parser.add_option("-n", "--numerical", help="Perform numerical evolution", action='store_false', dest="analytical")
parser.add_option("-f", "--flavour", help="Input neutrino state, in flavour basis", type='string', action='callback', callback=get_comma_separated_floats, dest="flavour")
parser.add_option("-m", "--mass", help="Input neutrino state, in mass basis", type='string', action='callback', callback=get_comma_separated_floats, dest="mass")
(options, args) = parser.parse_args()
if len(args) < 4 :
  print('Wrong number of arguments \n\
        \n\
Usage: python '+mainfilename+'.py -f/-m <eigensate> <in_file> <energy> <eta> <depth>\n\
       <eigenstate>                Flavour (-f/--flavour) or mass (-m/--mass) input eigenstate\n\
       <in_file>                   Input file\n\
       <energy>                    Energy of neutrinos\n\
       <eta>                       Nadir angle of the incident neutrinos\n\
       <depth>                     Depth of detector location below Earth\'s surface\n\
\n\
Options:\n\
       -h, --help                    Show this help message and exit\n\
       -v, --verbose                 Print debug output\n\
       -d, --density                 Add custom earth density profile\n\
       -a, --analytical              Perform analytical evolution\n\
       -n, --numerical               Perform numerical evolution')
  exit()


# Check that we have a valid neutrino state
if options.flavour == None and options.mass == None:
  print("Error: Missing neutrino state, you need to provide a neutrino state in either the flavour or mass basis.")
  exit()
elif options.flavour != None and options.mass != None:
  print("Error: Neutrino state cannot be given simultaneously in the flavour and mass basis, choose one.")
  exit()

# Read the input files
path = os.path.dirname(os.path.realpath( __file__ ))
slha_file = args[0]
density_file = path +'/Data/Earth_Density.csv' if options.density == '' else options.density

# Read example slha file and fill PMNS matrix
nu_params = f.read_slha(slha_file)
th12 = nu_params['theta12']
th13 = nu_params['theta13']
th23 = nu_params['theta23']
d = nu_params['delta']
pmns = PMNS(th12, th13, th23, d)

DeltamSq21 = nu_params['dm21']
DeltamSq31 = nu_params['dm31']

# Parse neutrino state
nustate = np.zeros(3)
if options.flavour != None:
  nustate = np.array(options.flavour)
elif options.mass != None:
  if len(options.mass) != 3:
    print("Error: neutrino state provided has the wrong format, it must be a vector of size 3.")
    exit()
  nustate = np.array(np.dot(pmns.pmns, np.array(options.mass)))[0]

# Get parameters
E = float(args[1])
eta = float(args[2])
H = float(args[3])

# Earth density
earth_density = EarthDensity(density_file)

# Print program banner and inputs
print_banner()
print_inputs("earth", options, pmns, DeltamSq21, DeltamSq31, E, eta, H)

# Compute probability of survival after propagation through Earth 
print("Running PEANUTS...")

# Check if analytical solution was requested
if options.analytical:
  prob = Pearth(nustate, earth_density, pmns, DeltamSq21, DeltamSq31, E, eta, H)

# Otherwise use numerical solution
else:
 prob = Pearth(nustate, earth_density, pmns, DeltamSq21, DeltamSq31, E, eta, H, mode="numerical")


# Print results
print()
print("Probability to oscillate to an electron neutrino : ", prob[0])
print("Probability to oscillate to a muon neutrino      : ", prob[1])
print("Probability to oscillate to a tau neutrino       : ", prob[2])
