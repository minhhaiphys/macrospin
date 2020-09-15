#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Physical constants
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from __future__ import division
import numpy as np

# Constants in CGS

ech   = 1.6022e-20               # Electron Charge
hbar  = 6.6261e-27 / (2.0*np.pi) # Reduced Planck's
muB   = 9.2740e-21               # Bohr Magneton
mu0   = np.pi*4e-7               # Permeability
kB    = 1.3807e-16               # Boltzmann
g     = 2.1                      # Lande Factor
gyromagnetic_ratio = g*muB / hbar             # gyromagnetic ratio
