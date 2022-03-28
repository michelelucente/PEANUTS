#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author Tomas Gonzalo <gonzalo@physik.rwth-aachen.de>
"""

import numpy as np
from math import sqrt
from numpy import arctan, arcsin
from optparse import OptionParser

import src.files as f
from src.pmns import PMNS
from src.solar import SolarModel, Psolar

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
#slha_file = args[0]
slha_file = './Data/example_slha.slha2' # To load from interactive session
solar_file = './Data/bs2005agsopflux.csv' if options.solar == "" else options.solar
density_file = './Data/Earth_Density.csv' if options.density == '' else options.density


# Define paths to save plots
import matplotlib.pyplot as plt
from pathlib import Path
project_folder = str(Path(Path.cwd()).parents[1])
plots_folder = project_folder + "/TeX/figs/"



# Test Sun survival probabilities
#################################

# Import data from solar model
solar_model = SolarModel(solar_file)

# Plot solar model

plt.plot(solar_model.radius, solar_model.density)
plt.yscale("log")
plt.xlabel("r")
plt.ylabel("$n_e(r)$ [mol/cm${}^3$]")
plt.legend(labels=['Solar electron density'])

plt.savefig(plots_folder + "sun_density.pdf")

plt.show()



plt.plot(solar_model.radius, solar_model.fraction['8B'])
plt.plot(solar_model.radius, solar_model.fraction['hep'])
plt.xlabel('r')
plt.ylabel('Neutrino fraction')
plt.legend(labels=['${}^8$B', 'hep'])

plt.savefig(plots_folder + "reaction_fraction.pdf")

plt.show()


# Read example slha file and fill PMNS matrix
nu_params = f.read_slha(slha_file)
th12 = nu_params['theta12']
th13 = nu_params['theta13']
th23 = nu_params['theta23']
d = nu_params['delta']
pmns = PMNS(th12, th13, th23, d)

DeltamSq21 = nu_params['dm21']
DeltamSq31 = nu_params['dm31']

# Values to compare with SNO, uncomment for exact comparison
#th12 = arctan(sqrt(0.469))
#th13 = arcsin(sqrt(0.01))
#DeltamSq21 = 7.9e-5
#DeltamSq31 = 2.46e-3
#pmns = PMNS(th12, th13, th13, d)

E = 10

# Compute probability for the sample fractions '8B' and 'hep' in the energy ragnge E=[1,20]
xrange = np.arange(1,20,0.1)
ProbB8 = [Psolar(pmns, DeltamSq21, DeltamSq31, E, solar_model.radius, solar_model.density, solar_model.fraction['8B']) for E in xrange]
Probhep = [Psolar(pmns, DeltamSq21, DeltamSq31, E, solar_model.radius, solar_model.density, solar_model.fraction['hep']) for E in xrange]

# Use SNO's example data files for comparison
SNO_B8 = f.read_csv("./Data/B8.csv", names=['energy', 'Pnuenue'])
SNO_hep = f.read_csv("./Data/HEP.csv", names=['energy', 'Pnuenue'])

labels = ["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"]

for flavour in range(len(ProbB8[0])):
    plt.plot(xrange, [prob[flavour] for prob in ProbB8], label=labels[flavour])

plt.plot(SNO_B8.energy, SNO_B8.Pnuenue, label="SNO $\\nu_e \\rightarrow \\nu_e$", linestyle="dashed")
plt.title("${}^8$B neutrinos")
plt.xlabel('Energy [MeV]')
plt.ylabel(r"P(${\nu_e \rightarrow \nu_\alpha})$")
plt.legend()
plt.savefig(plots_folder + "8B_SNO_cmparison.pdf")

plt.show()




for flavour in range(len(Probhep[0])):
    plt.plot(xrange, [prob[flavour] for prob in Probhep], label=labels[flavour])

plt.plot(SNO_hep.energy, SNO_hep.Pnuenue, label="SNO $\\nu_e \\rightarrow \\nu_e$", linestyle="dashed")
plt.title("hep neutrinos")
plt.xlabel('Energy [MeV]')
plt.ylabel(r"P(${\nu_e \rightarrow \nu_\alpha})$")
plt.legend()
plt.savefig(plots_folder + "hep_SNO_comparison.pdf")

plt.show()



# Test Earth density profiles
#############################
from math import pi
from src.earth import EarthDensity

x = np.arange(0,1,0.001)
eta = [0, pi/6, pi/4, pi/3]
labels = ["0", "pi/6", "pi/4", "pi/3"]

earth_density = EarthDensity(density_file)
density = [ [earth_density(r, n) for r in x] for n in eta]

plt.xlabel("x")
plt.ylabel("Density [mol/cm${}^3$]")
for i in range(len(density)):
    plt.plot(x,density[i], label = "$\eta$ = %s" %labels[i])
plt.legend()
plt.savefig(plots_folder + "earth_density.pdf")
    
plt.show()


# Test analytical vs numerical solutions
########################################
from src.earth import Pearth

# Use SNO location for comparison
H = 2e3 # meters

# Sample neutrino state
state = np.array([0,1,0])

# Case 1: 0 <= eta <= pi/2
eta = np.random.uniform(0, pi/2)
E = np.random.uniform(1,20)

sol, x = Pearth(state, earth_density, pmns, DeltamSq21, DeltamSq31, E, eta, H, mode="numerical", full_oscillation=True)

One_num = sol[-1]
One_num = np.array([One_num])

# Check analytical solution
One_an = Pearth(state, earth_density, pmns, DeltamSq21, DeltamSq31, E, eta, H)

err = np.linalg.norm(One_num - One_an)/np.linalg.norm(One_num + One_an)

print("For E = %.2f and eta = %.2f pi the relative error between analytic and numerical solutions is %f" % (E, eta/pi, err))

probs = sol

plt.xlabel("Trajectory coordinate")
plt.ylabel("Probability")
plt.title("Energy = %.2f MeV, nadir $\eta$ = %.2f $\pi$ | error = %f" % (E, eta/pi, err))
plt.plot(x, probs, label=["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"])
plt.legend()
plt.savefig(plots_folder + "earth_regeneration090.pdf")

plt.show()


# Case 2: pi/2 <= eta <= pi
eta = np.random.uniform(pi/2, pi)
E = np.random.uniform(1,20)

sol, x = Pearth(state, earth_density, pmns, DeltamSq21, DeltamSq31, E, eta, H, mode="numerical", full_oscillation=True)

One_num = sol[-1]
One_num = np.array([One_num])

# Check analytical solution
One_an = Pearth(state, earth_density, pmns, DeltamSq21, DeltamSq31, E, eta, H)

err = np.linalg.norm(One_num - One_an)/np.linalg.norm(One_num + One_an)

print("For E = %.2f and eta = %.2f pi the relative error between analytic and numerical solutions is %f" % (E, eta/pi, err))

probs = np.square(np.abs(sol))

plt.xlabel("Trajectory coordinate")
plt.ylabel("Probability")
plt.title("Energy = %.2f MeV, nadir $\eta$ = %.2f $\pi$ | error = %f" % (E, eta/pi, err))
plt.plot(x, probs, label=["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"])
plt.legend()
plt.savefig(plots_folder + "earth_regeneration90180.pdf")

plt.show()



# Test Sun-Earth survival probability
#####################################
from src.solar import solar_flux_mass
from src.evolutor import FullEvolutor

E = np.random.uniform(1, 20)
eta = np.random.uniform(0, pi)

H = 2e3

radius_samples = solar_model.radius
density = solar_model.density
fraction = solar_model.fraction['8B']


mass_weights = solar_flux_mass(th12, th13, DeltamSq21, DeltamSq31, E, radius_samples, density, fraction)

#mass_to_flavour_probabilitites = np.square(np.abs(np.dot(FullEvolutor(earth_density, 0, DeltamSq21, DeltamSq31, pmns, E, eta, H), pmns.conjugate())))

#flavour_probabilities = np.array(np.dot(mass_to_flavour_probabilitites, mass_weights))
# TODO:  This should now be the same as using the Pearth function, so do that
flavour_probabilities = Pearth(mass_weights, earth_density, pmns, DeltamSq21, DeltamSq31, E, eta, H)

print("For E = %.2f and eta = %.2f pi the flavour probabilitites are %s" % (E, eta/pi, str(flavour_probabilities)) )

exit()

# Test time average
from math import radians
from src.time_average import IntegralDay
from scipy import integrate

lat = [0, 45, 89]
shells_eta = np.insert(np.arcsin(np.array([0.192, 0.546, 0.895, 0.937, 1]))/pi, 0, 0)
colors = ['b', 'g', 'r', 'c', 'm']

x = np.linspace(0, pi, 10**3)
dist = [[IntegralDay(eta, radians(lam)) for eta in x] for lam in lat]

plt.xlabel("Nadir angle $\eta$ / $\pi$")
plt.ylabel("Weight")
plt.title("Annual nadir exposure for an experiment at various latitudes")
ax = plt.gca()
ax.set_xlim([0,1])
#plt.vlines(np.arcsin(np.array([0.192, 0.546, 0.895, 0.937, 1]))/pi, ymin=0, ymax=5, linestyles='dashed')
for i in range(len(shells_eta) - 1):
    plt.axvspan(shells_eta[i], shells_eta[i+1], alpha=0.1, color=colors[i])

for lam in range(len(lat)):
    plt.plot(x/pi, np.array(dist[lam])/integrate.trapezoid(x=x,y=dist[lam]), label="$\lambda$ = %.f°" % lat[lam])
plt.legend()

plt.savefig(plots_folder + "eta_weights.pdf")
plt.show()



# Import 8B energy spectrum
import pandas as pd

B8_shape = pd.read_csv('./Data/8B_shape.csv', header=None, names=['Energy MeV', 'Fraction'])
B8_shape.Fraction[B8_shape.Fraction < 0] = 0
print(integrate.trapezoid(x=B8_shape['Energy MeV'], y=B8_shape['Fraction']))


B8_shape.plot(x='Energy MeV', y='Fraction', title='${}^8$B energy spectrum')
plt.show()



# Compute distorted flux
radius_samples = solar_model.radius
density = solar_model.density
fraction = solar_model.fraction['8B']

survival_prob = np.array([Psolar(pmns, DeltamSq21, DeltamSq31, E, radius_samples, density, fraction) for E in B8_shape['Energy MeV']])
distorted_shape = np.array([B8_shape.Fraction]).T * survival_prob

labels = ["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"]

plt.plot(B8_shape['Energy MeV'], B8_shape.Fraction, label='$\\nu_e$ undistorted', linestyle='dashed')

for flavour in range(len(distorted_shape[0])):
    plt.plot(B8_shape['Energy MeV'], [prob[flavour] for prob in distorted_shape], label=labels[flavour])

#plt.yscale('log')
#plt.xscale('log')
    
plt.xlabel('Energy [MeV]')
plt.ylabel('Fraction')

plt.legend()

plt.show()