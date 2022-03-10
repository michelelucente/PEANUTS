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
from src.solar_model import SolarModel
from src.probability_sun_energy import PnuenueSunReaction

mainfilename = 'run_SNO_test'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-s", "--solar", help ="Add custom solar model", action='store', dest="solar", default="")
parser.add_option("-d", "--density", help ="Add custom earth density profile", action='store', dest="density", default="")
(options, args) = parser.parse_args()
if len(args) < 1 :
  print('Wrong number of arguments \n\
        \n\
Usage: python '+mainfilename+'.py <in_file>\n\
       <in_file>                   Input file\n\
\n\
Options:\n\
       -h, --help                    Show this help message and exit\n\
       -v, --verbose                 Print debug output\n\
       -s, --solar                   Add custom solar model\n\
       -d, --density                 Add custom earth density profile')

  exit()

# Read the input files
slha_file = args[0]
solar_file = './Data/bs2005agsopflux.csv' if options.solar == "" else options.solar
density_file = './Data/Earth_Density.csv' if options.density == '' else options.density

# Test Sun survival probabilities
#################################

# Import data from solar model
solar_model = SolarModel(solar_file)

# Read example slha file and fill PMNS matrix
nu_params = f.read_slha(slha_file)
th12 = nu_params['theta12']
th13 = nu_params['theta13']
th23 = nu_params['theta23']
d = nu_params['delta']
pmns = PMNS(th12, th13, th23, d)

DeltamSq21 = nu_params['dm21']
DeltamSq31 = nu_params['dm31']
#th12 = arctan(sqrt(0.469)) TODO: These values are different than the PDG's, why?
#th13 = arcsin(sqrt(0.01))
#DeltamSq21 = 7.9e-5
#DeltamSq31 = 2.46e-3
E = 10
ne = 100 # TODO: What is this for?

# Compute probability for the sample fractions '8B' and 'hep' in the energy ragnge E=[1,20]
xrange = np.arange(1,20,0.1)
ProbB8 = [PnuenueSunReaction(th12, th13, DeltamSq21, DeltamSq31, E, solar_model.radius, solar_model.density, solar_model.fraction['8B']) for E in xrange]
Probhep = [PnuenueSunReaction(th12, th13, DeltamSq21, DeltamSq31, E, solar_model.radius, solar_model.density, solar_model.fraction['hep']) for E in xrange]

# Use SNO's example data files for comparison
SNO_B8 = f.read_csv("./Data/B8.csv", names=['energy', 'Pnuenue'])
SNO_hep = f.read_csv("./Data/HEP.csv", names=['energy', 'Pnuenue'])

# Define paths to save plots
import matplotlib.pyplot as plt
from pathlib import Path
project_folder = str(Path(Path.cwd()).parents[1])
plots_folder = project_folder + "/TeX/figs/"


plt.plot(xrange, ProbB8, label="This code")
plt.plot(SNO_B8.energy, SNO_B8.Pnuenue, label='SNO_example')
plt.title("${}^8$B neutrinos")
plt.xlabel('Energy [MeV]')
plt.ylabel(r"$P_{\nu_e \rightarrow \nu_e}$")
plt.legend()
plt.savefig(plots_folder + "8B_SNO_cmparison.pdf")

plt.show()


plt.plot(xrange, Probhep, label="This code")
plt.plot(SNO_hep.energy, SNO_hep.Pnuenue, label='SNO_example')
plt.title("hep neutrinos")
plt.xlabel('Energy [MeV]')
plt.ylabel(r"$P_{\nu_e \rightarrow \nu_e}$")
plt.legend()
plt.savefig(plots_folder + "hep_SNO_comparison.pdf")

plt.show()


# Test Earth density profiles
#############################
from math import pi
from src.earth_density import EarthDensity

x = np.arange(0,1,0.001)
eta = [0, pi/6, pi/4, pi/3]
labels = ["0", "pi/6", "pi/4", "pi/3"]

earth_density = EarthDensity(density_file)
density = [ [earth_density(r, n) for r in x] for n in eta]

plt.xlabel("Nadir angle $\eta$")
plt.ylabel("Density [mol/cm${}^3$]")
for i in range(len(density)):
    plt.plot(x,density[i], label = "$\eta$ = %s" %labels[i])
plt.legend()
plt.savefig(plots_folder + "earth_density.pdf")
    
plt.show()


# Test Earth regeneration
#########################
from src.earth_regeneration import PneunueEarth, PnuenueEarth_analytical

# Use SNO location for comparison
H = 2e3 # meters

# Case 1: 0 <= eta <= pi/2
eta = np.random.uniform(0, pi/2)
E = np.random.uniform(1,20)

One_num = PnuenueEarth(earth_density, pmns, DeltamSq21, DeltamSq31, eta, E, H)

# Check analytical solution
One_an = PnuenueEarth_analytical(earth_density, pmns, DeltamSq21, DeltamSq31, eta, E, H)

err = np.linalg.norm(One_num - One_an)/np.linalg.norm(One_num + One_an)

print("For E = %.2f and eta = %.2f pi the relative error between analytic and numerical solutions is %f" % (E, eta/pi, err))


# Case 2: pi/2 <= eta <= pi
eta = np.random.uniform(pi/2, pi)
E = np.random.uniform(1,20)

One_num = PnuenueEarth(earth_density, pmns, DeltamSq21, DeltamSq31, eta, E, H)

# Check analytical solution
One_an = PnuenueEarth_analytical(earth_density, pmns, DeltamSq21, DeltamSq31, eta, E, H)

err = np.linalg.norm(One_num - One_an)/np.linalg.norm(One_num + One_an)

print("For E = %.2f and eta = %.2f pi the relative error between analytic and numerical solutions is %f" % (E, eta/pi, err))
