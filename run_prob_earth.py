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

import peanuts.files as f
from peanuts.utils import get_comma_separated_floats, print_banner, print_inputs
from peanuts.settings import Settings
from peanuts.pmns import PMNS
from peanuts.solar import SolarModel, solar_flux_mass
from peanuts.earth import EarthDensity, Pearth

mainfilename = 'run_prob_earth'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-i", "--in_slha", help="SLHA input file", action='store', dest='in_slha', default="")
parser.add_option("-d", "--density", help ="Add custom earth density profile", action='store', dest="density", default="")
parser.add_option("--antinu", help = "", action="store_true", dest="antinu", default=False)
parser.add_option("--analytical", help="Perform analytical evolution", action='store_true', dest="analytical", default=True)
parser.add_option("--numerical", help="Perform numerical evolution", action='store_false', dest="analytical")
parser.add_option("-f", "--flavour", help="Input neutrino state, in flavour basis", type='string', action='callback', callback=get_comma_separated_floats, dest="flavour")
parser.add_option("-m", "--mass", help="Input neutrino state, in mass basis", type='string', action='callback', callback=get_comma_separated_floats, dest="mass")
(options, args) = parser.parse_args()
if len(args) < 3 or (options.in_slha == "" and len(args) != 9):
  print('Wrong number of arguments \n\
        \n\
Usage: ./'+mainfilename+'.py [options] -f/-m <eigenstate> <energy> <eta> <depth> [<th12> <th13> <th23> <delta> <md21> <md3l>]\n\
       <eigenstate>                Flavour (-f/--flavour) or mass (-m/--mass) input eigenstate\n\
       <energy>                    Energy of neutrinos\n\
       <eta>                       Nadir angle of the incident neutrinos\n\
       <depth>                     Depth of detector location below Earth\'s surface\n\
       <th12>                      Mixing angle theta_12\n\
       <th13>                      Mixing angle theta_13\n\
       <th23>                      Mixing angle theta_23\n\
       <delta>                     CP phase delta\n\
       <md21>                      Mass splitting m^2_{21}\n\
       <md3l>                      Mass splitting m^2_{3} (l=1 NO, l=2 IO)\n\
\n\
Options:\n\
       -h, --help                  Show this help message and exit\n\
       -v, --verbose               Print debug output\n\
       -i, --in_slha <slha_file>   SLHA input file for neutrino parameters\n\
       -d, --density               Add custom earth density profile\n\
       --antinu                    Input state is an antineutrino\n\
       --analytical                Perform analytical evolution\n\
       --numerical                 Perform numerical evolution')
  exit()


# Check that we have a valid neutrino state
if options.flavour == None and options.mass == None:
  print("Error: Missing neutrino state, you need to provide a neutrino state in either the flavour or mass basis.")
  exit()
elif options.flavour != None and options.mass != None:
  print("Error: Neutrino state cannot be given simultaneously in the flavour and mass basis, choose one.")
  exit()

# Read the density file
path = os.path.dirname(os.path.realpath( __file__ ))
density_file = path +'/Data/Earth_Density.csv' if options.density == '' else options.density

# Get parameters
E = float(args[0])
eta = float(args[1])
depth = float(args[2])

# If the -i/--in_slha option is given, read the slha file
if options.in_slha != "":

  # If pyslha has not been imported throw error
  if not f.with_slha:
    print("Error: The module `pyslha` is needed to use SLHA input, please install it.")
    exit()

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

  th12 = float(args[3])
  th13 = float(args[4])
  th23 = float(args[5])
  d = float(args[6])
  pmns = PMNS(th12, th13, th23, d)

  DeltamSq21 = float(args[7])
  DeltamSq3l = float(args[8])

# Parse neutrino state
antinu = options.antinu
nustate = np.zeros(3)
if options.flavour != None:
  nustate = np.array(options.flavour)
  massbasis = False
elif options.mass != None:
  #if len(options.mass) != 3:
  #  print("Error: neutrino state provided has the wrong format, it must be a vector of size 3.")
  #  exit()
  #nustate = np.array(np.dot(pmns.pmns, np.array(options.mass)))[0]
  nustate = np.array(options.mass)
  massbasis = True

# Earth density
earth_density = EarthDensity(density_file=density_file)

# Print program banner and inputs
print_banner()
settings = Settings(pmns, DeltamSq21, DeltamSq3l, E, eta, depth, options)
print_inputs(settings)
print("Running PEANUTS...")

# Compute probability of survival after propagation through Earth

# Check if analytical solution was requested
if options.analytical:
  prob = Pearth(nustate, earth_density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, massbasis=massbasis, antinu=antinu)

# Otherwise use numerical solution
else:
 prob = Pearth(nustate, earth_density, pmns, DeltamSq21, DeltamSq3l, E, eta, depth, mode="numerical", massbasis=massbasis, antinu=antinu)


# Print results
print()
print("Probability to oscillate to an electron " + ("anti" if settings.antinu else "") + "neutrino : ", prob[0])
print("Probability to oscillate to a muon " + ("anti" if settings.antinu else "") + "neutrino      : ", prob[1])
print("Probability to oscillate to a tau " + ("anti" if settings.antinu else "") + "neutrino       : ", prob[2])
print()
