#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on My 11 2022

@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import numpy as np
import copy

from src.pmns import PMNS

class Param:

  def __init__(self, label, value):

    self.nparams = 1
    setattr(self,label, value)

  def add(self, label, value):
    self.nparams += 1
    setattr(self,label, value)

  def __repr__(self):
    ret = "["
    for attr in dir(self):
      if not attr.startswith("__") and not attr == "add" and not attr == "nparams":
        ret  += attr + " : " + str(getattr(self, attr)) + ", "
    ret = ret[:-2]+"]"
    return ret

class Scan:

  def __init__(self):

    self.params = list()
    self.labels = list()
    self._index = 0

  def __iter__(self):
    return self

  def __next__(self):
    if self._index < len(self.params):
      self._index += 1
      return self.params[self._index-1]
    else:
      raise StopIteration

  def __len__(self):
    return len(self.params)

  def enumerate(self):
    return enumerate(self.params)

  def add(self, label, param):

    if isinstance(param, list):
      # Assume it is given as [min, max] or [min, max, step]
      if len(param) < 2 or len(param) > 3:
        print("Error: Parameter", label, "should be given as single number or as range [min, max, (step)].")
        exit()

      self.labels.append(label)

      parammin = float(param[0])
      parammax = float(param[1])
      if len(param) == 3:
        step = float(param[2])
        N = int( (parammax-parammin)/step)+1
      else:
        # If step is not given, assume 10 iterations
        N = 10
        step = (parammax-parammin)/(N-1)
      values = [parammin + i*step for i in range(0,N)]

      if len(self.params):
        newparams = list()
        for par in self.params:
          for val in values:
            newparam = copy.copy(par)
            newparam.add(label,val)
            newparams.append(newparam)
        self.params = newparams
      else:
        for val in values:
          self.params.append(Param(label,val))

    else:
      if len(self.params):
        for par in self.params:
          par.add(label, param)
      else:
        self.params.append(Param(label,param))


class Settings:

  def __init__(self, *args):

    self.solar = False
    self.earth = False
    self.scan = Scan()

    # If there is only one argument, it is a settings dictionary
    if len(args) == 1:
      settings = args[0]

      # Select mode first
      if "Solar" in settings:
        self.solar = True
      if "Earth" in settings:
        self.earth = True
      if not self.solar and not self.earth:
        print("Error: unkown mode, please provide a running mode, \"Solar\", \"Earth\" or both")
        exit()

      # Extract neutrino parameters
      if "Neutrinos" not in settings:
        print("Error: missing neutrino information, please provide dm21, dm31, theta12, theta13, theta23 and delta")
        exit()
      elif isinstance(settings["Neutrinos"], dict):
        if "dm21" not in settings["Neutrinos"] or\
           "dm31" not in settings["Neutrinos"] or\
           "theta12" not in settings["Neutrinos"] or\
           "theta13" not in settings["Neutrinos"] or\
           "theta23" not in settings["Neutrinos"] or\
           "delta" not in settings["Neutrinos"] :
          print("Error: missing neutrino information, please provide dm21, dm31, theta12, theta13, theta23 and delta")
          exit()
        else:
          self.dm21 = settings["Neutrinos"]["dm21"]
          self.dm31 = settings["Neutrinos"]["dm31"]
          self.theta12 = settings["Neutrinos"]["theta12"]
          self.theta13 = settings["Neutrinos"]["theta13"]
          self.theta23 = settings["Neutrinos"]["theta23"]
          self.delta = settings["Neutrinos"]["delta"]

          self.scan.add("dm21", self.dm21)
          self.scan.add("dm31", self.dm31)
          self.scan.add("theta12", self.theta12)
          self.scan.add("theta13", self.theta13)
          self.scan.add("theta23", self.theta23)
          self.scan.add("delta", self.delta)
      else:

        import src.files as f

        slha_file = settings["Neutrinos"]
        try:
          nu_params = f.read_slha(slha_file)
          th12 = nu_params['theta12']
          th13 = nu_params['theta13']
          th23 = nu_params['theta23']
          d = nu_params['delta']
          self.pmns = PMNS(th12, th13, th23, d)
          self.dm21 = nu_params['dm21']
          self.dm31 = nu_params['dm31']
        except FileNotFoundError:
          print("Error: slha file " + slha_file + " not found.")
          exit()



      # Extract solar parameters
      if "Solar" in settings:

        if "fraction" not in settings["Solar"]:
          print("Error: missing solar neutrino fraction.")
          exit()
        else:
          self.fraction = settings["Solar"]["fraction"]

        self.solar_file = settings["Solar"]["solar_model"] if "solar_model" in settings["Solar"] else None

        self.spectra = settings["Solar"]["spectra"] if "spectra" in settings["Solar"] else None

        self.probabilities = settings["Solar"]["probabilities"] if "probabilities" in settings["Solar"] else True

        self.flux = settings["Solar"]["flux"] if "flux" in settings["Solar"] else False

        self.undistorted_spectrum = False
        self.distorted_spectrum =  False
        if "spectrum" in settings["Solar"]:
          if settings["Solar"]["spectrum"] == "undistorted":
            self.undistorted_spectrum = True
          elif settings["Solar"]["spectrum"] == "distorted":
            self.distorted_spectrum = True
          else:
            print("Error: unknown option for spectrum, select undistorted or distorted")
            exit()

      # Extract earth parameters
      if "Earth" in settings:

        if "Solar" not in settings and\
           ("state" not in settings["Earth"] or "basis" not in settings["Earth"]):
          print("Error: missing input neutrino state or basis, please provide both.")
          exit()
        elif "Solar" not in settings:
          self.nustate = np.array(settings["Earth"]["state"])
          self.basis = settings["Earth"]["basis"]
          self.probabilities = True

        if "depth" not in settings["Earth"]:
          print("Error: missing depth of experiment, please provide it.")
          exit()
        else:
          self.depth = settings["Earth"]["depth"]

        # Either a specific nadir angle, eta, or a latitude or exposure file must be provided
        if "eta" not in settings["Earth"] and "latitude" not in settings["Earth"] and not "exposure_file" in settings["Earth"]:
          print("Error: please provide a nadir angle (eta), a latitude or exposure file path.")
          exit()
        elif "eta" in settings["Earth"]:
          self.eta = settings["Earth"]["eta"]
          self.exposure = False
          self.scan.add("eta", self.eta)
        elif "latitude" in settings["Earth"] or "exposure_file" in settings["Earth"]:
          if("latitude" in settings["Earth"] and "exposure_file" in settings["Earth"]):
            print("Warning: both latitude and exposure file provided, latitude value will be ignored")
          self.exposure = True
          self.latitude = settings["Earth"]["latitude"] if "latitude" in settings["Earth"] and "exposure_file" not in settings["Earth"] else -1
          self.exposure_normalized = settings["Earth"]["exposure_normalized"] if "exposure_normalized" in settings["Earth"] else False
          self.exposure_time = settings["Earth"]["exposure_time"] if "exposure_time" in settings["Earth"] else [0,365]
          self.exposure_samples = settings["Earth"]["exposure_samples"] if "exposure_samples" in settings["Earth"] else 1000
          self.exposure_file = settings["Earth"]["exposure_file"] if "exposure_file" in settings["Earth"] else None
          self.exposure_angle = settings["Earth"]["exposure_angle"] if "exposure_angle" in settings["Earth"] else "Nadir"
        else:
          print("Error: a nadir angle (eta) and exposure option (latitude or file) were found, please provide only one of them.")
          exit()

        self.density_file = settings["Earth"]["density"] if "density" in settings["Earth"] else None
        self.evolution = settings["Earth"]["evolution"] if "evolution" in settings["Earth"] else "analytical"
        self.evolved_state = settings["Earth"]["evolved_state"] if "evolved_state" in settings["Earth"] else False

      # Extract energy
      if "Energy" not in settings:
        print("Error: missing energy, please provide value or range.")
        exit()
      else:
        self.energy = settings["Energy"]
        self.scan.add("energy", self.energy)

      # Select printing mode, default is stdout
      if "Output" in settings:
        self.output = settings["Output"]
      else:
        self.output = "stdout"

    # If there are exactly 6 arguments, we are in solar mode
    # args = (pmns, dm21, dm31, E, fraction, options)
    elif len(args) == 6:

      self.solar = True
      self.pmns = args[0]
      self.theta12 = self.pmns.theta12
      self.theta13 = self.pmns.theta13
      self.theta23 = self.pmns.theta23
      self.delta = self.pmns.delta
      self.dm21 = args[1]
      self.dm31 = args[2]
      self.energy = args[3]
      self.fraction = args[4]
      self.solar_file = args[5].solar if args[5].solar != "" else None
      self.probabilities = True
      self.undistorted_spectrum = False
      self.distorted_spectrum =  False

    # If there are exactly 7 arguments, we are in earth mode
    # args = (pmns, dm21, dm31, E, eta, H, options)
    elif len(args) == 7:

      self.earth = True
      self.pmns = args[0]
      self.theta12 = self.pmns.theta12
      self.theta13 = self.pmns.theta13
      self.theta23 = self.pmns.theta23
      self.delta = self.pmns.delta
      self.dm21 = args[1]
      self.dm31 = args[2]
      self.energy = args[3]
      self.eta = args[4]
      self.depth = args[5]

      if args[6].flavour is not None:
        self.nustate = args[6].flavour
        self.basis = "flavour"
      elif args[6].mass is not None:
        self.nustate = args[6].mass
        self.basis = "mass"
      else:
        print("Error: unknown basis, please choose either flavour or mass basis.")
        exit()

      self.density_file = args[6].density if args[6].density != "" else None
      self.evolution = "analytical" if args[6].analytical else "numerical"

      self.exposure = False

