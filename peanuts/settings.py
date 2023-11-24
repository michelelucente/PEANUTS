#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on My 11 2022

@author Tomas Gonzalo <tomas.gonzalo@kit.edu>
"""

import numpy as np
import copy

from peanuts.pmns import PMNS

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
      # Assume it is given as [min, max], [min, max, step] or [min, max, step, mode]
      if len(param) < 2 or len(param) > 4:
        print("Error: Parameter", label, "should be given as single number or as range [min, max, (step), (mode)].")
        exit()

      self.labels.append(label)

      parammin = float(param[0])
      parammax = float(param[1])

      if len(param) > 2:
        step = float(param[2])
        N = int( (parammax-parammin)/step)+1
      else:
        # If step is not given, assume 10 iterations
        N = 10
        step = (parammax-parammin)/(N-1)

      # Set scan mode
      mode = "linear"
      if len(param) == 4:
        if param[3] == "log":
          mode = "log"
        elif not param[3] == "linear":
          print("Error: Unknown scan mode `"+param[3]+"`. It should be \"linear\" or \"log\".")

      # Set values, depdending on whether we are in linear or log mode
      if mode == "linear":
        values = [parammin + i*step for i in range(0,N)]
      elif mode == "log":
        values = [10**(parammin + i*step) for i in range(0,N)]

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

    self.vacuum = False
    self.solar = False
    self.earth = False
    self.scan = Scan()

    # If there is only one argument, it is a settings dictionary
    if len(args) == 1:
      settings = args[0]

      # Select mode first
      if "Vacuum" in settings:
        self.vacuum = True
      if "Solar" in settings:
        self.solar = True
      if "Earth" in settings:
        self.earth = True
      if not self.vacuum and not self.solar and not self.earth:
        print("Error: unkown mode, please provide a running mode, \"Vacuum\", \"Solar\", \"Earth\" or a combination of them")
        exit()

      # Extract neutrino parameters
      if "Neutrinos" not in settings:
        print("Error: missing neutrino information, please provide dm21, dm3l, theta12, theta13, theta23 and delta")
        exit()
      elif isinstance(settings["Neutrinos"], dict):
        if "dm21" not in settings["Neutrinos"] or\
           "dm3l" not in settings["Neutrinos"] or\
           "theta12" not in settings["Neutrinos"] or\
           "theta13" not in settings["Neutrinos"] or\
           "theta23" not in settings["Neutrinos"] or\
           "delta" not in settings["Neutrinos"] :
          print("Error: missing neutrino information, please provide dm21, dm3l, theta12, theta13, theta23 and delta")
          exit()
        else:
          self.dm21 = settings["Neutrinos"]["dm21"]
          self.dm3l = settings["Neutrinos"]["dm3l"]
          self.theta12 = settings["Neutrinos"]["theta12"]
          self.theta13 = settings["Neutrinos"]["theta13"]
          self.theta23 = settings["Neutrinos"]["theta23"]
          self.delta = settings["Neutrinos"]["delta"]

          self.scan.add("dm21", self.dm21)
          self.scan.add("dm3l", self.dm3l)
          self.scan.add("theta12", self.theta12)
          self.scan.add("theta13", self.theta13)
          self.scan.add("theta23", self.theta23)
          self.scan.add("delta", self.delta)
      else:

        import peanuts.files as f

        slha_file = settings["Neutrinos"]
        try:
          nu_params = f.read_slha(slha_file)
          th12 = nu_params['theta12']
          th13 = nu_params['theta13']
          th23 = nu_params['theta23']
          d = nu_params['delta']
          self.pmns = PMNS(th12, th13, th23, d)
          self.dm21 = nu_params['dm21']
          self.dm3l = nu_params['dm3l']
        except FileNotFoundError:
          print("Error: slha file " + slha_file + " not found.")
          exit()

      # Extract vacuum parameters
      if "Vacuum" in settings:
        if "Solar" in settings or "Earth" in settings:
          print("Error: Vacuum mode can only be used on its own")
          exit()
        elif "state" not in settings["Vacuum"] or "basis" not in settings["Vacuum"] or "baseline" not in settings["Vacuum"]:
          print("Error: Vacuum oscillations require an input state, its basis and the baseline")
          exit()
        else:
          self.nustate = np.array(settings["Vacuum"]["state"],dtype=complex)
          self.antinu = settings["Vacuum"]["antinu"] if "antinu" in settings["Vacuum"] else False
          self.basis = settings["Vacuum"]["basis"]
          self.baseline = settings["Vacuum"]["baseline"]
          self.scan.add("baseline", self.baseline)
          self.probabilities = settings["Vacuum"]["probabilities"] if "probabilities" in settings["Vacuum"] else True
          self.evolved_state = settings["Vacuum"]["evolved_state"] if "evolved_state" in settings["Vacuum"] else False

      # Extract solar parameters
      if "Solar" in settings:

        if "fraction" not in settings["Solar"]:
          print("Error: missing solar neutrino fraction.")
          exit()
        else:
          self.fraction = settings["Solar"]["fraction"]

        self.antinu = False
        self.solar_file = settings["Solar"]["solar_model"] if "solar_model" in settings["Solar"] else None
        self.flux_file = settings["Solar"]["flux_file"] if "flux_file" in settings["Solar"] else None
        self.fluxrows = settings["Solar"]["fluxrows"] if "fluxrows" in settings["Solar"] else None
        self.fluxcols = settings["Solar"]["fluxcols"] if "fluxcols" in settings["Solar"] else None
        self.fluxscale = settings["Solar"]["fluxscale"] if "fluxscale" in settings["Solar"] else None
        self.distrow = settings["Solar"]["distrow"] if "distrow" in settings["Solar"] else None
        self.radiuscol = settings["Solar"]["radiuscol"] if "radiuscol" in settings["Solar"] else None
        self.densitycol = settings["Solar"]["densitycol"] if "densitycol" in settings["Solar"] else None
        self.fractioncols = settings["Solar"]["fractioncols"] if "fractioncols" in settings["Solar"] else None

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

        self.antinu = False
        if "Solar" not in settings and\
           ("state" not in settings["Earth"] or "basis" not in settings["Earth"]):
          print("Error: missing input neutrino state or basis, please provide both.")
          exit()
        elif "Solar" not in settings:
          self.nustate = np.array(settings["Earth"]["state"],dtype=complex)
          self.antinu = settings["Earth"]["antinu"] if "antinu" in settings["Earth"] else False
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
        self.custom_density = settings["Earth"]["custom_density"] if "custom_density" in settings["Earth"] else False
        self.tabulated_density = settings["Earth"]["tabulated_density"] if "tabulated_density" in settings["Earth"] else False
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
    # args = (pmns, dm21, dm3l, E, fraction, options)
    elif len(args) == 6:

      self.solar = True
      self.pmns = args[0]
      self.theta12 = self.pmns.theta12
      self.theta13 = self.pmns.theta13
      self.theta23 = self.pmns.theta23
      self.delta = self.pmns.delta
      self.dm21 = args[1]
      self.dm3l = args[2]
      self.energy = args[3]
      self.fraction = args[4]
      self.solar_file = args[5].solar if args[5].solar != "" else None
      self.probabilities = True
      self.undistorted_spectrum = False
      self.distorted_spectrum =  False

    # If there are exactly 7 arguments, we are in earth mode
    # args = (pmns, dm21, dm3l, E, eta, depth, options)
    elif len(args) == 7:

      self.earth = True
      self.antinu = args[6].antinu
      self.pmns = args[0]
      self.theta12 = self.pmns.theta12
      self.theta13 = self.pmns.theta13
      self.theta23 = self.pmns.theta23
      self.delta = self.pmns.delta
      self.dm21 = args[1]
      self.dm3l = args[2]
      self.energy = args[3]
      self.eta = args[4]
      self.depth = args[5]

      if args[6].flavour is not None:
        self.basis = "flavour"
        self.nustate = args[6].flavour
      elif args[6].mass is not None:
        self.basis = "mass"
        self.nustate = args[6].mass
      else:
        print("Error: unknown basis, please choose either flavour or mass basis.")
        exit()

      self.density_file = args[6].density if args[6].density != "" else None
      self.evolution = "analytical" if args[6].analytical else "numerical"

      self.exposure = False

