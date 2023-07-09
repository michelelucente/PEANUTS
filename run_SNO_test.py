#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2022

@author: Michele Lucente <lucente@physik.rwth-aachen.de>
@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import os
import numpy as np
from math import sqrt
from numpy import arctan, arcsin
from optparse import OptionParser
from scipy.interpolate import interp1d

import peanuts.files as f
from peanuts.pmns import PMNS
from peanuts.solar import SolarModel, Psolar

mainfilename = 'run_SNO_test'

parser = OptionParser()
parser.add_option("-v", "--verbose", help = "Print debug output", action='store_true', dest='verbose', default=False)
parser.add_option("-s", "--solar", help ="Add custom solar model", action='store', dest="solar", default="")
parser.add_option("-d", "--density", help ="Add custom earth density profile", action='store', dest="density", default="")
parser.add_option("-f", "--file", help="Input slha file", action='store', dest="slha_file", default="")
(options, args) = parser.parse_args()
if len(args) > 0 :
  print('Wrong number of arguments \n\
        \n\
Usage: python '+mainfilename+'.py [options]\n\
\n\
Options:\n\
       -h, --help                    Show this help message and exit\n\
       -v, --verbose                 Print debug output\n\
       -s, --solar                   Add custom solar model\n\
       -d, --density                 Add custom earth density profile\n\
       -f, --file                    Input slha file')

  exit()

# Read the input files
path = os.path.dirname(os.path.realpath( __file__ ))
solar_file = path + '/Data/bs2005agsopflux.csv' if options.solar == "" else options.solar
density_file = path + '/Data/Earth_Density.csv' if options.density == '' else options.density


# Define paths to save plots
import matplotlib.pyplot as plt
project_folder = os.path.dirname(os.path.realpath( __file__ ))
plots_folder = project_folder + "/figs/"
if not os.path.exists(plots_folder):
  os.makedirs(plots_folder)


# Test Sun survival probabilities
#################################

# Import data from solar model
solar_model = SolarModel(solar_file)

# Plot solar model

plt.plot(solar_model.radius(), solar_model.density().table())
plt.yscale("log")
plt.xlabel("r")
plt.ylabel("$n_e(r)$ [mol/cm${}^3$]")
plt.legend(labels=['Solar electron density'])

plt.savefig(plots_folder + "sun_density.pdf")

plt.show()



plt.plot(solar_model.radius(), solar_model.fraction('8B'))
plt.plot(solar_model.radius(), solar_model.fraction('hep'))
plt.xlabel('r')
plt.ylabel('Neutrino fraction')
plt.legend(labels=['${}^8$B', 'hep'])

plt.savefig(plots_folder + "reaction_fraction.pdf")

plt.show()


# Read example slha file if providedand fill PMNS matrix
if options.slha_file != "":
  nu_params = f.read_slha(slha_file)
  th12 = nu_params['theta12']
  th13 = nu_params['theta13']
  th23 = nu_params['theta23']
  d = nu_params['delta']
  pmns = PMNS(th12, th13, th23, d)

  DeltamSq21 = nu_params['dm21']
  DeltamSq3l = nu_params['dm3l']

else:
  # Values to compare with SNO, if no slha is provided
  th12 = arctan(sqrt(0.469))
  th13 = arcsin(sqrt(0.01))
  th23 = 0.85521
  d = 3.4034
  DeltamSq21 = 7.9e-5
  DeltamSq3l = 2.46e-3
  pmns = PMNS(th12, th13, th23, d)

# Energy
E = 10

# Compute probability for the sample fractions '8B' and 'hep' in the energy ragnge E=[1,20]
xrange = np.arange(1,20,0.1)
ProbB8 = [Psolar(pmns, DeltamSq21, DeltamSq3l, E, solar_model.radius(), solar_model.density(), solar_model.fraction('8B')) for E in xrange]
Probhep = [Psolar(pmns, DeltamSq21, DeltamSq3l, E, solar_model.radius(), solar_model.density(), solar_model.fraction('hep')) for E in xrange]

# Use SNO's example data files for comparison
SNO_B8 = f.read_csv(path + "/Data/SNO_8B.csv", skiprows=4, names=['energy', 'Pnuenue'])
SNO_hep = f.read_csv(path + "/Data/SNO_hep.csv", skiprows=4, names=['energy', 'Pnuenue'])

labels = ["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"]

for flavour in range(len(ProbB8[0])):
    plt.plot(xrange, [prob[flavour] for prob in ProbB8], label=labels[flavour])

plt.plot(SNO_B8.energy, SNO_B8.Pnuenue, label="SNO $\\nu_e \\rightarrow \\nu_e$", linestyle="dashed")
plt.title("${}^8$B neutrinos")
plt.xlabel('Energy [MeV]')
plt.ylabel(r"P(${\nu_e \rightarrow \nu_\alpha})$")
plt.legend()
plt.savefig(plots_folder + "8B_SNO_comparison.pdf")

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

# Compare 8B energy spectrum with distorted flux
B8_spectrum = solar_model.spectrum("8B")

survival_prob = np.array([Psolar(pmns, DeltamSq21, DeltamSq3l, E, solar_model.radius(), solar_model.density(), solar_model.fraction('8B')) for E in B8_spectrum.Energy])
SNO_interp = interp1d(SNO_B8.energy, SNO_B8.Pnuenue, kind='cubic', fill_value='extrapolate')
SNO_survival_prob = np.array([SNO_interp(E) for E in B8_spectrum.Energy])
distorted_shape = np.array([B8_spectrum.Spectrum]).T * survival_prob
SNO_distorted_shape = np.array([B8_spectrum.Spectrum]).T * np.array([SNO_survival_prob]).T

labels = ["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"]

for flavour in range(len(distorted_shape[0])):
    plt.plot(B8_spectrum.Energy, [prob[flavour] for prob in distorted_shape], label=labels[flavour])
plt.plot(B8_spectrum.Energy, SNO_distorted_shape, label="SNO $\\nu_e$", linestyle='dashed')
plt.plot(B8_spectrum.Energy, B8_spectrum.Spectrum, label='$\\nu_e$ undistorted', linestyle='dotted')
plt.xlabel('Energy [MeV]')
plt.ylabel('8B Spectrum')
plt.legend()
plt.savefig(plots_folder + "8B_SNO_comparison_spectrum.pdf")

plt.show()

# Compare hep energy spectrum with distorted flux
hep_spectrum = solar_model.spectrum("hep")

survival_prob = np.array([Psolar(pmns, DeltamSq21, DeltamSq3l, E, solar_model.radius(), solar_model.density(), solar_model.fraction('hep')) for E in hep_spectrum.Energy])
SNO_interp = interp1d(SNO_hep.energy, SNO_hep.Pnuenue, kind='cubic', fill_value='extrapolate')
SNO_survival_prob = np.array([SNO_interp(E) for E in hep_spectrum.Energy])
distorted_shape = np.array([hep_spectrum.Spectrum]).T * survival_prob
SNO_distorted_shape = np.array([hep_spectrum.Spectrum]).T * np.array([SNO_survival_prob]).T


labels = ["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"]


for flavour in range(len(distorted_shape[0])):
    plt.plot(hep_spectrum.Energy, [prob[flavour] for prob in distorted_shape], label=labels[flavour])
plt.plot(hep_spectrum.Energy, SNO_distorted_shape, label="SNO $\\nu_e$", linestyle='dashed')
plt.plot(hep_spectrum.Energy, hep_spectrum.Spectrum, label='$\\nu_e$ undistorted', linestyle='dashed')
plt.xlabel('Energy [MeV]')
plt.ylabel('hep Spectrum')
plt.legend()
plt.savefig(plots_folder + "hep_SNO_comparison_spectrum.pdf")

plt.show()



# Test Earth density profiles
#############################
from math import pi
from peanuts.earth import EarthDensity

x = np.arange(0,1,0.001)
eta = [0, pi/6, pi/4, pi/3]
labels = ["0", "$\pi$/6", "$\pi$/4", "$\pi$/3"]

earth_density = EarthDensity(density_file)
density = [ [earth_density.call(r, n) for r in x] for n in eta]

plt.xlabel("x")
plt.ylabel("Density [mol/cm${}^3$]")
for i in range(len(density)):
    plt.plot(x,density[i], label = "$\eta$ = %s" %labels[i])
plt.legend()
plt.savefig(plots_folder + "earth_density.pdf")

plt.show()


# Test analytical vs numerical solutions
########################################
from peanuts.earth import Pearth

# Use SNO location for comparison
H = 2e3 # meters

# Sample neutrino state, pure mass eigenstate
state = pmns.pmns[:,1]

# Case 1: 0 <= eta <= pi/2
eta = 0
E = 10 # MeV

# Uncomment to randomise values of eta and E
#eta = np.random.uniform(0, pi/2)
#E = np.random.uniform(1,20)

sol, x = Pearth(state, earth_density, pmns, DeltamSq21, DeltamSq3l, E, eta, H, massbasis=False, mode="numerical", full_oscillation=True)

One_num = sol[-1]
One_num = np.array([One_num])

# Check analytical solution
One_an = Pearth(state, earth_density, pmns, DeltamSq21, DeltamSq3l, E, eta, H, massbasis=False)

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
eta = pi/2
E = 10 # MeV

# Uncomment to randomise values of eta and E
#eta = np.random.uniform(pi/2, pi)
#E = np.random.uniform(1,20)

sol, x = Pearth(state, earth_density, pmns, DeltamSq21, DeltamSq3l, E, eta, H, massbasis=False, mode="numerical", full_oscillation=True)

One_num = sol[-1]
One_num = np.array([One_num])

# Check analytical solution
One_an = Pearth(state, earth_density, pmns, DeltamSq21, DeltamSq3l, E, eta, H, massbasis=False)

err = np.linalg.norm(One_num - One_an)/np.linalg.norm(One_num + One_an)

print("For E = %.2f and eta = %.2f pi the relative error between analytic and numerical solutions is %f" % (E, eta/pi, err))

probs = sol

plt.xlabel("Trajectory coordinate")
plt.ylabel("Probability")
plt.title("Energy = %.2f MeV, nadir $\eta$ = %.2f $\pi$ | error = %f" % (E, eta/pi, err))
plt.plot(x, probs, label=["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"])
plt.legend()
plt.savefig(plots_folder + "earth_regeneration90180.pdf")

plt.show()



# Test Sun-Earth survival probability
#####################################
from peanuts.solar import solar_flux_mass
from peanuts.evolutor import FullEvolutor

E = np.random.uniform(1, 20)
eta = np.random.uniform(0, pi)

H = 2e3

radius_samples = solar_model.radius()
density = solar_model.density()
fraction = solar_model.fraction('8B')


mass_weights = solar_flux_mass(pmns, DeltamSq21, DeltamSq3l, E, radius_samples, density, fraction)

flavour_probabilities = Pearth(mass_weights, earth_density, pmns, DeltamSq21, DeltamSq3l, E, eta, H, massbasis=True)

print("For E = %.2f and eta = %.2f pi the flavour probabilitites are %s" % (E, eta/pi, str(flavour_probabilities)) )


# Test exposure time average
#############################
from math import radians
from peanuts.exposure import NadirExposure

# Get the values of eta for all the shells
shells_eta = np.insert(earth_density.shells_eta(), 0, 0)
colors = ['b', 'g', 'r', 'c', 'm']

# Compute nadir exposure for various latitude values
lat = [0, 46, 89]
exposure = [NadirExposure(lam=radians(lam), normalized=True, ns=480) for lam in lat]

plt.xlabel("Nadir angle $\eta$ / $\pi$")
plt.ylabel("Weight")
plt.title("Annual nadir exposure for an experiment at various latitudes")
ax = plt.gca()
ax.set_xlim([0,1])
for i in range(len(shells_eta) - 1):
    plt.axvspan(shells_eta[i], shells_eta[i+1], alpha=0.1, color=colors[i])

for lam in range(len(lat)):
    plt.plot(exposure[lam][:,0]/pi, exposure[lam][:,1], label="$\lambda$ = %.f°" % lat[lam])

# SNO actual exposure
SNO_exposure = NadirExposure(file=path+'/Data/SnoCosZenith.dat', angle="CosZenith", normalized=True, ns=480)
plt.plot(exposure[lam][:,0]/pi, SNO_exposure[:,1], label="SNO exposure")

plt.legend()

plt.savefig(plots_folder + "eta_weights.pdf")
plt.show()

