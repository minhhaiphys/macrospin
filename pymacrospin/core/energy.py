#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Functions for the free energy of a macrospin magnet
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from __future__ import division
import numpy as np

def zeeman(m, Ms, H):
    """ Returns the Zeeman energy in erg/cc
    m: Reduced magnetization M/Ms (unit vector)
    Ms: Satuation magnetization (emu/cc)
    H: Applied magnetic field (Oe)
    """
    return Ms*m.dot(H)

def uniaxial_anisotropy(m, u=np.array([0, 0, 1]), Ku1=0, Ku2=0):
    """ Returns the uniaxial anisotropy part of the free energy in erg/cc
    m: Reduced magnetization M/Ms (unit vector)
    u: Uniaxial direction (unit vector)
    Ku1: Uniaxial anisotropy energy 1 (erg/cc)
    Ku2: Uniaxial anisotropy energy 2 (erg/cc)
    """
    return -Ku1*(u.dot(m)**2) - Ku2*(u.dot(m)**4)

def cubic_anisotropy(m, c1, c2, Kc1=0, Kc2=0, Kc3=0):
    """ Returns the cubic anisotropy part of the free energy in erg/cc
    m: Reduced magnetization M/Ms (unit vector)
    c1: In-plane orthogonal crystal direction 1 (unit vector)
    c2: In-plane orthogonal crystal direction 2 (unit vector)
    Kc1: Cubic anisotropy energy 1 (erg/cc)
    Kc2: Cubic anisotropy energy 2 (erg/cc)
    Kc2: Cubic anisotropy energy 3 (erg/cc)
    """
    if c1.dot(c2) != 0:
        raise Exception("Crystal directions c1 and c2 must be orthogonal")
    c3 = np.cross(c1, c2)

    mc1 = c1.dot(m)
    mc2 = c2.dot(m)
    mc3 = c3.dot(m)

    # First order term
    Fc = Kc1*((mc1**2)*(mc2**2))
    Fc += Kc1*((mc2**2)*(mc3**2))
    Fc += Kc1*((mc1**2)*(mc3**2))

    # Second order term
    Fc += Kc2*((mc1**2)*(mc2**2)*(mc3**2))

    # Third order term
    Fc += Kc3*((mc1**4)*(mc2**4))
    Fc += Kc3*((mc2**4)*(mc3**4))
    Fc += Kc3*((mc1**4)*(mc3**4))

    return Fc


def shape_anisotropy(m, Ms, Nxx, Nyy, Nzz):
    """ Returns the shape anisotropy part of the free energy in erg/cc.
    Assumes the demagnetization only requires diagonal elements of the
    demagnetziation
    m: Reduced magnetization M/Ms (unit vector)
    Ms: Saturation magnetization (emu/cc)
    Nxx: Demagnetization component along xx
    Nyy: Demagnetization component along yy
    Nzz: Demagnetization component along zz

    Nxx + Nyy + Nzz = 4 pi
    """
    return np.dot([Nxx, Nyy, Nzz],m)*(Ms**2)
