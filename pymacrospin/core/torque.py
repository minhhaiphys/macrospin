#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Functions for calculating field-induced and spin-transfer torque
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy as np


def landau_lifshitz(m, heff, alpha):
    """ Return the Landau-Lifshitz torque

    m - moment unit vector
    heff - effective magnetic field
    alpha - Gilbert damping parameter
    """
    hxm = np.cross(heff,m)
    return hxm + alpha*np.cross(m,hxm)


def slonczewski(m, Jc, stt):
    """ Return the Slonczewski spin-transfer torque

    m - moment unit vector
    Jc - current density vector
    stt - torque prefactor (pre-calculated)
    """
    p = -Jc*stt
    return np.cross(m,np.cross(p,m))
