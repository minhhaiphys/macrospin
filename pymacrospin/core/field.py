#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Functions for calculating effective fields
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def demagnetization(m, Nd):
    """ Returns the demagnetization field based on the diagonal
    elements of the demagnetization tensor

    m: moment unit vector
    Nd: diagonal elements of demagnetization tensor
    """
    return -m*Nd


def field_basic(m, hext, Nd):
    """ Return the effective field for BasicKernel

    m: moment unit vector
    hext: external field
    Nd: diagonal elements of demagnetization tensor
    """
    return hext + demagnetization(m, Nd)


def uniaxial_anisotropy(m, u, hu1, hu2):
    """ Returns the uniaxial anisotropy field

    m: moment unit vector
    u: uniaxial anisotropy unit vector
    hu1: normalized uniaxial anisotropy field (1st order)
    hu2: normalized uniaxial anisotropy field (2nd order)
    """
    m_u = m.dot(u)
    return u*(m_u*hu1) + u*(m_u*m_u*m_u*hu2)


def cubic_anisotropy(m, c1, c2, c3, hc1, hc2):
    """ Returns the cubic anisotropy field

    m: moment unit vector
    c1, c2, c3: orthogonal cubic axis unit vectors
    hc1: normalized cubic anisotropy field (1st order)
    hc2: normalized cubic anisotropy field (2nd order)
    """
    m_c1 = m.dot(c1)
    m_c2 = m.dot(c2)
    m_c3 = m.dot(c3)
    h =  hc1*(m_c2*m_c2 + m_c3*m_c3)*(m_c1*c1)
    h += hc1*(m_c1*m_c1 + m_c3*m_c3)*(m_c2*c2)
    h += hc1*(m_c1*m_c1 + m_c2*m_c2)*(m_c3*c3)

    h += hc2*(m_c2*m_c2 * m_c3*m_c3)*(m_c1*c1)
    h += hc2*(m_c1*m_c1 * m_c3*m_c3)*(m_c2*c2)
    h += hc2*(m_c1*m_c1 * m_c2*m_c2)*(m_c3*c3)
    return h
