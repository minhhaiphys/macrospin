#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Functions for solving ODE
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import numpy as np
import numba as nb
from pymacrospin.numba.__init__ import normalize

@nb.njit(cache=True)
def euler_step(torque, dt, m, hext, Jc):
    """Takes one step using the Euler method

    torque: function to calculate torque from m, hext and Jc
    dt: time step
    m: moment unit vector
    hext: external field
    Jc: current density
    """
    t = torque(m, hext, Jc)
    return normalize(m + dt*t)

@nb.njit(cache=True)
def huen_step(torque, dt, m, hext, Jc):
    """ Takes one step using Huen's method

    torque: function to calculate torque from m, hext and Jc
    dt: time step
    m: moment unit vector
    hext: external field
    Jc: current density
    """
    k1 = torque(m)
    m1 = m + dt*k1
    k2 = torque(m1)
    m = m + dt*(k1 + k2)/2.0
    return normalize(m)

@nb.njit(cache=True)
def rk23_step(torque, dt, m, hext, Jc):
    """ Takes one step using the Bogacki-Shampine method (Runga-Kutta RK23)

    torque: function to calculate torque from m, hext and Jc
    dt: time step
    m: moment unit vector
    hext: external field
    Jc: current density
    """
    k1 = torque(m, hext, Jc)
    k2 = torque(m + dt*k1/2.0, hext, Jc)
    k3 = torque(m + 3.0*dt*k2/2.0, hext, Jc)
    m = m + 2.0*dt*k1/9.0 + dt*k2/3.0 + 4*dt*k3/9.0
    return normalize(m)


@nb.njit(cache=True)
def rk4_step(torque, dt, m, hext, Jc):
    """ Takes one step using the Classic 4th order Runga-Kutta method

    torque: function to calculate torque from m, hext and Jc
    dt: time step
    m: moment unit vector
    hext: external field
    Jc: current density
    """
    k1 = torque(m, hext, Jc)
    k2 = torque(m + dt*k1/2.0, hext, Jc)
    k3 = torque(m + dt*k2/2.0, hext, Jc)
    k4 = torque(m + dt*k3, hext, Jc)
    m = m + dt*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
    return normalize(m)

@nb.njit
def run(step_func, steps, m, hext, Jc):
    """ Run multiple steps over time period t

    step_func: step run function
    steps: number of steps
    m: moment unit vector
    hext: external field
    Jc: current density
    """
    ms = np.zeros((steps+1,3),dtype=np.float32)
    ms[0] = m
    for i in range(steps):
        ms[i+1] = step_func(ms[i], hext, Jc)
    return ms[1:]

@nb.njit
def relax(step_func, energy_func, precision, steps, max_iters, m, hext, Jc):
    """ Run the simulation until energy variation falls within a threshold

    precision: energy's relative error for halting condition
    steps: number of steps per iteration
    max_iters: maximum number of iterations
    m: moment unit vector
    hext: external field
    Jc: current density
    """
    ms = np.zeros((steps,3),dtype=np.float32)
    ms[-1] = m
    g1 = energy_func(m, hext)
    for i in range(max_iters):
        g0 = g1
        ms = run(step_func, steps,ms[-1], hext, Jc)
        g1 = energy_func(ms[-1], hext)
        if g0-g1 < abs(g0*precision):
            # Reach local minimum within precision
            return ms[-1], i*steps
    return ms[-1], i*steps

@nb.njit
def stabilize(step_func, torque_func, dm_thres, steps, max_iters, dt, m, hext, Jc):
    """ Run until torque is below a threshold within a defined errorbar

    step_func: step run function
    torque_func: function to calculate torque
    dm_thres: halting threshold for dm
    steps: number of steps per iteration
    max_iters: maximum number of iterations
    dt: time step
    m: moment unit vector
    hext: external field
    Jc: current density
    """
    ms = np.zeros((steps,3),dtype=np.float32)
    ms[-1] = m
    for i in range(max_iters):
        if np.linalg.norm(torque_func(ms[-1], hext, Jc))*dt < dm_thres:
            return ms[-1], i*steps
        else:
            ms = run(step_func, steps, ms[-1], hext, Jc)
    return ms[-1], i*steps
