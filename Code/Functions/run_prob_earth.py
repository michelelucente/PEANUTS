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
from src.solar import SolarModel, solar_flux_mass
from src.earth import EarthDensity, Pearth

mainfilename = 'run_prob_earth'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-s", "--solar", help ="Add custom solar model", action='store', dest="solar", default="")
parser.add_option("-f", "--flavour", help="Neutrino flavour", action='store', dest='flav', default="")
parser.add_option("-d", "--density", help ="Add custom earth density profile", action='store', dest="density", default="")
parser.add_option("-a", "--analytical", help="Perform analytical evolution", action='store_true', dest="analytical", default=True)
parser.add_option("-n", "--numerical", help="Perform numerical evolution", action='store_false', dest="analytical")
(options, args) = parser.parse_args()
if len(args) < 5 :
  print('Wrong number of arguments \n\
        \n\
Usage: python '+mainfilename+'.py <in_file> <energy> <eta> <depth>\n\
       <in_file>                   Input file\n\
       <energy>                    Energy of neutrinos\n\
       <nu_fraction>               Neutrino fraction sample\n\
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
solar_file = './Data/bs2005agsopflux.csv' if options.solar == "" else options.solar
density_file = './Data/Earth_Density.csv' if options.density == '' else options.density

# Import data from solar model
solar_model = SolarModel(solar_file)

# Get parameters
E = float(args[1])
nu_fraction = args[2]
if not solar_model.has_fraction(nu_fraction):
   print("Error: The fraction ", nu_fraction, " does not exist in the solar model.")
   exit()
eta = float(args[3])
H = float(args[4])

# Read example slha file and fill PMNS matrix
nu_params = f.read_slha(slha_file)
th12 = nu_params['theta12']
th13 = nu_params['theta13']
th23 = nu_params['theta23']
d = nu_params['delta']
pmns = PMNS(th12, th13, th23, d)

DeltamSq21 = nu_params['dm21']
DeltamSq31 = nu_params['dm31']

# Compute solar neutrino flux
nustate = solar_flux_mass(th12, th13, DeltamSq21, DeltamSq31, E, solar_model.radius, solar_model.density, solar_model.fraction[nu_fraction])

# Earth density
earth_density = EarthDensity(density_file)

# Compute probability of survival after propagation through Earth 

# Check if analytical solution was requested
if options.analytical:
  print("Using analytical computation")
  prob = Pearth(nustate, earth_density, pmns, DeltamSq21, DeltamSq31, E, eta, H)

# Otherwise use numerical solution
else:
 print("Using numerical computation")
 prob = Pearth(nustate, earth_density, pmns, DeltamSq21, DeltamSq31, E, eta, H, mode="numerical")


# TODO: Which unit do we expect the energy?
if options.flav == '':
  print("Probabilities of survival of neutrinos of sample fraction", nu_fraction, "with energy E =", E, "and nair angle eta =", eta, "at", H, "meters below the surface of the Earth is", prob)
elif options.flav == 'e':
  print("Probability of survival of electron neutrinos of sample fraction", nu_fraction, "with energy E =", E, "and nair angle eta =", eta, "at", H, "meters below the surface of the Earth is", prob[0])
elif options.flav == 'mu':
  print("Probability of survival of muon neutrinos of sample fraction", nu_fraction, "with energy E =", E, "and nair angle eta =", eta, "at", H, "meters below the surface of the Earth is", prob[1])
elif options.flav == 'tau':
  print("Probability of survival of tau neutrinos of sample fraction", nu_fraction, "with energy E =", E, "and nair angle eta =", eta, "at", H, "meters below the surface of the Earth is", prob[2])
else:
  print("Error: Unknown neutrino flavour. Options: [e, mu, tau]")

