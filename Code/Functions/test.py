#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 17:47:01 2022

@author: michele
"""
import pandas as pd


# Import data from solar model
solar_model = pd.read_csv("./Data/bs2005agsopflux.csv", 
                          usecols=[1, 3, *range(6,14)],
                          names = ['radius', 'density_log_10', 'pp fraction', '8B fraction', '13N fraction',
                                   '15O fraction', '17F fraction', '7Be fraction', 'pep fraction',
                                   'hep fraction'],
                          sep=" ", skiprows=27, header=None)

solar_model['density'] = 10**solar_model['density_log_10']


# Define paths to save plots
from pathlib import Path

project_folder = str(Path(Path.cwd()).parents[1])
plots_folder = project_folder + "/TeX/figs/"


# Plot solar model
import matplotlib.pyplot as plt

plot = solar_model.plot(x='radius', y='density', logy=True)
plot.set_xlabel("r")
plot.set_ylabel("$n_e(r)$ [mol/cm${}^3$]")
plot.legend(labels=['Solar electron density'])

plt.savefig(plots_folder + "sun_density.pdf")

plt.show()



plot = solar_model.plot(x='radius', y=['pp fraction', '8B fraction', '13N fraction',
                                   '15O fraction', '17F fraction', '7Be fraction', 'pep fraction',
                                   'hep fraction'])
plot.set_xlabel('r')
plot.set_ylabel('Neutrino fraction')
plot.legend(labels=['pp', '${}^8$B', '${}^{13}$N','${}^{15}$O', '${}^{17}$F', '${}^7$Be', 'pep', 'hep'])

plt.savefig(plots_folder + "reaction_fraction.pdf")

plt.show()


# Test Sun survival probabilities
import numpy as np 

from math import sqrt, atan, asin
from solar import Psolar

th12 = atan(sqrt(0.469))
th13 = asin(sqrt(0.01))
[th23, d] = [0.855211, 3.40339]
DeltamSq21 = 7.9e-5
DeltamSq31 = 2.46e-3
E = 10
ne = 100
radius_samples = solar_model.radius
density = solar_model.density
fraction = solar_model['8B fraction']

xrange = np.arange(1,20,0.1)
ProbB8 = [Psolar(th12, th13, th23, d, DeltamSq21, DeltamSq31, E, radius_samples, density, solar_model['8B fraction']) for E in xrange]
Probhep = [Psolar(th12, th13, th23, d, DeltamSq21, DeltamSq31, E, radius_samples, density, solar_model['hep fraction']) for E in xrange]


SNO_B8 = pd.read_csv("./Data/B8.csv", names=['energy', 'Pnuenue'])
SNO_hep = pd.read_csv("./Data/hep.csv", names=['energy', 'Pnuenue'])


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




labels = ["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"]

for flavour in range(len(ProbB8[0])):
    plt.plot(xrange, [prob[flavour] for prob in Probhep], label=labels[flavour])

plt.plot(SNO_hep.energy, SNO_hep.Pnuenue, label="SNO $\\nu_e \\rightarrow \\nu_e$", linestyle="dashed")
plt.title("hep neutrinos")
plt.xlabel('Energy [MeV]')
plt.ylabel(r"P(${\nu_e \rightarrow \nu_\alpha})$")
plt.legend()
plt.savefig(plots_folder + "hep_SNO_comparison.pdf")

plt.show()




# Test Earth density profiles
import matplotlib.pyplot as plt
from math import pi

from earth import EarthDensity

x = np.arange(0,1,0.001)
eta = [0, pi/6, pi/4, pi/3]
labels = ["0", "pi/6", "pi/4", "pi/3"]

density = [ [EarthDensity(r, n) for r in x] for n in eta]

plt.xlabel("x")
plt.ylabel("Density [mol/cm${}^3$]")
for i in range(len(density)):
    plt.plot(x,density[i], label = "$\eta$ = %s" %labels[i])
plt.legend()
plt.savefig(plots_folder + "earth_density.pdf")
    
plt.show()



# Test Earth regeneration
from pmns import R12, R13, R23, Delta
from numpy.linalg import multi_dot
from math import sin, cos
from scipy.integrate import complex_ode

from potentials import k, MatterPotential, R_E
from evolutor import FullEvolutor

r13 = R13(th13)
r12 = R12(th12)
r23 = R23(th23)
delta = Delta(d)

pmns = multi_dot([r23, delta, r13, delta.conjugate(), r12])
U = np.dot(r13, r12)


# Case 1: 0 <= eta <= pi/2
eta = np.random.uniform(0, pi/2)
E = np.random.uniform(1,20)

Hk = multi_dot([U, np.diag(k(np.array([0, DeltamSq21, DeltamSq31]), E)), U.transpose()])

H = 2e3 # meters
h = H/R_E
r_d = 1 - h
x_d = r_d * cos(eta) #sqrt(r_d**2 - sin(eta)**2)
Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)
n_1 = EarthDensity(x=1 - h/2)
eta_prime = asin(r_d * sin(eta))

params = EarthDensity(eta=eta_prime, parameters=True)
x1, x2 = (-params[-1][1], x_d) if 0 <= eta < pi/2 else (0, Deltax)

def model(t, y):
    nue, numu, nutau = y
    dnudt = - 1j * np.dot(multi_dot([r23, delta.conjugate(), Hk + np.diag([
        MatterPotential(EarthDensity(t, eta=eta_prime)) if 0 <= eta < pi/2 else n_1
        ,0,0]), delta, r23.transpose()]), [nue, numu, nutau])
    return dnudt

nu0 = (pmns.transpose()[1, :]).conjugate()

nu = complex_ode(model)

nu.set_integrator("Isoda")
nu.set_initial_value(nu0, x1)


x = np.linspace(x1, x2, 10**3)
sol = [nu.integrate(xi) for xi in x[1::]]
sol.insert(0, np.array(nu0)[0])

One_num = sol[-1]
One_num = np.array([One_num])

One_an = np.dot(FullEvolutor(0, DeltamSq21, DeltamSq31, E, th12, th13, th23, d, eta, H), nu0.transpose()).transpose()
One_an = np.array(One_an)

err = np.linalg.norm(One_num - One_an)/np.linalg.norm(One_num + One_an)

print("For E = %.2f MeV and eta = %.2f pi the relative error between analytic and numerical solutions is %f" % (E, eta/pi, err))


probs = np.square(np.abs(sol))

plt.xlabel("Trajectory coordinate")
plt.ylabel("Probability")
plt.title("Energy = %.2f MeV, nadir $\eta$ = %.2f $\pi$ | error = %f" % (E, eta/pi, err))
plt.plot(x, probs, label=["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"])
plt.legend()
plt.savefig(plots_folder + "eart_regeneration090.pdf")

plt.show()


# Case 2: pi/2 <= eta <= pi
eta = np.random.uniform(pi/2, pi)
E = np.random.uniform(1,20)

Hk = multi_dot([U, np.diag(k(np.array([0, DeltamSq21, DeltamSq31]), E)), U.transpose()])

H = 2e3 # meters
h = H/R_E
r_d = 1 - h
x_d = r_d * cos(eta) #sqrt(r_d**2 - sin(eta)**2)
Deltax = r_d * cos(eta) + sqrt(1 - r_d**2 * sin(eta)**2)
n_1 = EarthDensity(x=1 - h/2)
eta_prime = asin(r_d * sin(eta))

params = EarthDensity(eta=eta_prime, parameters=True)
x1, x2 = (-params[-1][1], x_d) if 0 <= eta < pi/2 else (0, Deltax)

def model(t, y):
    nue, numu, nutau = y
    dnudt = - 1j * np.dot(multi_dot([r23, delta.conjugate(), Hk + np.diag([
        MatterPotential(EarthDensity(t, eta=eta_prime)) if 0 <= eta < pi/2 else n_1
        ,0,0]), delta, r23.transpose()]), [nue, numu, nutau])
    return dnudt

nu0 = (pmns.transpose()[1, :]).conjugate()

nu = complex_ode(model)

nu.set_integrator("Isoda")
nu.set_initial_value(nu0, x1)


x = np.linspace(x1, x2, 10**3)
sol = [nu.integrate(xi) for xi in x[1::]]
sol.insert(0, np.array(nu0)[0])

One_num = sol[-1]
One_num = np.array([One_num])

One_an = np.dot(FullEvolutor(0, DeltamSq21, DeltamSq31, E, th12, th13, th23, d, eta, H), nu0.transpose()).transpose()
One_an = np.array(One_an)

err = np.linalg.norm(One_num - One_an)/np.linalg.norm(One_num + One_an)

print("For E = %.2f MeV and eta = %.2f pi the relative error between analytic and numerical solutions is %f" % (E, eta/pi, err))


probs = np.square(np.abs(sol))

plt.xlabel("Trajectory coordinate")
plt.ylabel("Probability")
plt.title("Energy = %.2f MeV, nadir $\eta$ = %.2f $\pi$ | error = %f" % (E, eta/pi, err))
plt.plot(x, probs, label=["$\\nu_e$", "$\\nu_\mu$", "$\\nu_\\tau$"])
plt.legend()
plt.savefig(plots_folder + "eart_regeneration90180.pdf")

plt.show()


# Test Sun-Earth xurvival probability
from solar import solar_flux_mass

th12 = atan(sqrt(0.469))
th13 = asin(sqrt(0.01))
DeltamSq21 = 7.9e-5
DeltamSq31 = 2.46e-3

pmns = multi_dot([r23, delta, r13, delta.conjugate(), r12])

E = np.random.uniform(1, 20)
eta = np.random.uniform(0, pi)

H = 2e3

radius_samples = solar_model.radius
density = solar_model.density
fraction = solar_model['8B fraction']


mass_weights = solar_flux_mass(th12, th13, DeltamSq21, DeltamSq31, E, radius_samples, density, fraction)

mass_to_flavour_probabilitites = np.square(np.abs(np.dot(FullEvolutor(0, DeltamSq21, DeltamSq31, E, th12, th13, th23, d, eta, H), pmns.conjugate())))

flavour_probabilities = np.array(np.dot(mass_to_flavour_probabilitites, mass_weights))

print("For E = %.2f and eta = %.2f pi the flavour probabilitites are %s" % (E, eta/pi, str(flavour_probabilities[0])) )



# Test time average
from math import radians
from time_average import IntegralDay
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
B8_shape = pd.read_csv('./Data/8B_shape.csv', header=None, names=['Energy MeV', 'Fraction'])
B8_shape.Fraction[B8_shape.Fraction < 0] = 0
print(integrate.trapezoid(x=B8_shape['Energy MeV'], y=B8_shape['Fraction']))


B8_shape.plot(x='Energy MeV', y='Fraction', title='${}^8$B energy spectrum')
plt.show()



# Compute distorted flux
th12 = atan(sqrt(0.469))
th13 = asin(sqrt(0.01))
[th23, d] = [0.855211, 3.40339]
DeltamSq21 = 7.9e-5
DeltamSq31 = 2.46e-3
radius_samples = solar_model.radius
density = solar_model.density
fraction = solar_model['8B fraction']

survival_prob = np.array([Psolar(th12, th13, th23, d, DeltamSq21, DeltamSq31, E, radius_samples, density, fraction) for E in B8_shape['Energy MeV']])
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
