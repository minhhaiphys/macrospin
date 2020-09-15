#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Functions for calculating field-induced and spin-transfer torque
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy as np
import numba as nb
from pymacrospin.constants import hbar, ech, mu0
from pymacrospin.numba.__init__ import dot

@nb.njit(cache=True)
def landau_lifshitz(m, heff, alpha):
    """ Return the Landau-Lifshitz torque

    m: moment unit vector
    heff: effective magnetic field
    alpha: Gilbert damping parameter
    """
    hxm = np.cross(heff,m)
    return hxm + alpha*np.cross(m,hxm)


@nb.njit(cache=True)
def slonczewski(m, mp, Jc, P, Lambda):
    """ Return the Slonczewski damping-like spin-transfer torque

    m: moment unit vector
    mp: spin polarization unit vector
    Jc: current (volume) density (A/m^3)
    P: polarization
    Lambda: spin transfer efficiency
    """
    Js = (hbar/ech)*Jc # Spin density
    L2 = Lambda*Lambda
    epsilon = P*L2 / ( (L2+1) + (L2-1)*dot(m,mp) )
    return Js*epsilon*np.cross(m,np.cross(mp,m))


@nb.njit(cache=True)
def slonczewski_fieldlike(m, mp, Jc, epsilon):
    """ Return the Slonczewski field-like spin-transfer torque

    m: moment unit vector
    mp: spin polarization unit vector
    Jc: current (volume) density (A/m^3)
    epsilon: secondary spin transfer term
    """
    Js = (hbar/ech)*Jc # Spin density
    return -Js*epsilon*np.cross(m,mp)
