#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Includes helper functions for pymacrospin
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy as np
import numba as nb

@nb.njit(cache=True)
def normalize(array):
    """ Normalize """
    norm = np.sqrt(np.sum(array*array))
    return array/norm


@nb.njit(cache=True)
def dot(u1,u2):
    """ Dot product of two vectors """
    return u1[0]*u2[0] + u1[1]*u2[1] + u1[2]*u2[2]


def normalize_field(value, Ms):
    """ Normalize field

    value: Field vector
    Ms: saturation magnetization
    """
    return np.array(value, dtype=np.float32)/Ms


def normalize_energy(value, Ms):
    """ Normalize energy

    value: Field vector
    Ms: saturation magnetization
    """
    return 2*value/(Ms**2)
