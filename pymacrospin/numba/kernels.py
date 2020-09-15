#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Kernel classes for efficiently evolving the macrospin
# Numba kernels: Use Numba
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import numpy as np
import numba as nb

from pymacrospin.constants import hbar, ech, gyromagnetic_ratio
from pymacrospin.numba.__init__ import normalize, normalize_field, normalize_energy
from pymacrospin.numba import field, torque, solvers, energy


class Kernel:
    """ Encapsulates the time evolution algorithm for solving the
    Landau-Liftshitz-Gilbert and Slonczewski equation

    The following parameters can be provided:
    dt: time step
    Ms: saturation magnetization
    m0: initial moment
    damping: Gilbert damping constant

    method: ODE solver, either 1-Euler, 2-Huen, 3-RK23 (default) or 4-RK4
    unit: 'CGS' (default) or 'MKS' (not yet supported)
    gyromagnetic_ratio: gyromagnetic ratio
    """

    def __init__(self, **parameters):
        self.parameters = {'unit': "CGS",
                                'method': "RK23",
                                'm0': [1,0,0],
                                'dt': 1e-12
                                }
        # Basic properties
        self.method = "RK23"
        self.Ms = 1
        self.damping = 0
        self._Hext = np.array([0,0,0],dtype=np.float32)
        self.hext = np.array([0,0,0],dtype=np.float32)
        self.Jc = 0
        self.gyromagnetic_ratio = gyromagnetic_ratio
        # Update parameters
        self.update_params(**parameters)
        self.reset_environment()
        self.test_run()  # Move compiling overhead here

    def Hext():
        doc = "The Hext property: Externally applied field"
        def fget(self):
            return self._Hext
        def fset(self, value):
            self._Hext = np.array(value, dtype=np.float32)
            self.hext = normalize_field(value, self.Ms)
        def fdel(self):
            del self._Hext
        return locals()
    Hext = property(**Hext())


    def update_params(self,**params):
        """ Update raw parameters

        params: keyword parameters
        """
        self.parameters.update(params)
        if self.parameters['unit'].lower()=="mks":
            raise ValueError("MKS is not yet supported.")

        # Load the parameters by translate the dict to kernel attributes
        for k,v in self.parameters.items():
            if isinstance(v,list):
                setattr(self, k, np.array(v, dtype=np.float32))
            else:
                setattr(self, k, v)
        # Rescale gyromagnetic ratio to Landau-Lifshitz version
        self.gyromagnetic_ratio = self.gyromagnetic_ratio/(1.0 + self.damping**2)
        # Rescale time in terms of (gamma Ms)^-1
        self.time_conversion = self.gyromagnetic_ratio * self.Ms
        self.dt = self.dt * self.time_conversion


    def reset_environment(self):
        """ Set up basic exec functions
        Only include external field
        Without anisotropy or spin-torque
        """
        self.make_field()
        self.make_torque()
        self.make_energy()
        self.make_step()


    def test_run(self):
        """ Test run 2 steps """
        self.reset()
        self.run(2*self.dt/self.time_conversion)
        self.reset()


    def reset(self):
        """ Resets the kernel to the initial conditions
        """
        self.m = normalize(self.m0)
        self.t = 0.0


    def make_field(self):
        """ Define function to calculate effective field
        """
        @nb.njit
        def field_func(m, hext):
            return hext
        self.field = field_func


    def make_torque(self):
        """ Define function to calculate torque
        = Landau_Lifshitz torque
        """
        field_func = self.field
        alpha = self.damping
        @nb.njit
        def torque_func(m, hext, Jc):
            heff = field_func(m, hext)
            total_torque = torque.landau_lifshitz(m, heff, alpha)
            return total_torque
        self.torque = torque_func


    def make_energy(self):
        """ Define function to calculate energy
        = Zeeman energy + demagnetization energy
        """
        field_func = self.field
        Ms = self.Ms
        @nb.njit
        def energy_func(m, hext):
            heff = field_func(m, hext)
            E = -energy.zeeman(m, Ms, heff)
            return E
        self.energy = energy_func


    def add_demag(self, Nd):
        """ Add demagnetization field

        Nd: Demagnetization diagonal tensor elements
        """
        # Change the field function to include demag field
        current_field_func = self.field
        @nb.njit
        def field_func(m, hext):
            return current_field_func(m, hext) + field.demagnetization(m, Nd)
        self.field = field_func

        # Change the torque function to include demag field
        current_torque_func = self.torque
        alpha = self.damping
        @nb.njit
        def torque_func(m, hext, Jc):
            total_torque = current_torque_func(m, hext, Jc)
            heff = field.demagnetization(m, Nd)
            total_torque += torque.landau_lifshitz(m, heff, alpha)
            return total_torque
        self.torque = torque_func

        # Change the energy function to inlcude uniaxial anisotropy
        Ms = self.Ms
        current_energy_func = self.energy
        @nb.njit
        def energy_func(m, hext):
            return current_energy_func(m, hext) + energy.shape_anisotropy(m, Ms, Nd[0], Nd[1], Nd[2])
        self.energy = energy_func
        # Initialize
        self.make_step()
        self.test_run()


    def add_uniaxial_anisotropy(self, u, Ku1, Ku2):
        """ Add Uniaxial Anisotropy to the sample

        u: uniaxial anisotropy unit vector
        Ku1: Uniaxial anisotropy energy 1 (erg/cc)
        Ku2: Uniaxial anisotropy energy 2 (erg/cc)
        """
        # Normalize parameters
        u = normalize(np.array(u,dtype=np.float32))
        Ms = self.Ms
        hu1 = normalize_energy(Ku1, Ms)
        hu2 = normalize_energy(Ku2, Ms)

        # Change the field function to inlcude uniaxial anisotropy
        current_field_func = self.field
        @nb.njit
        def field_func(m, hext):
            return current_field_func(m, hext) + field.uniaxial_anisotropy(m, u, hu1, hu2)
        self.field = field_func

        # Change the torque function to include demag field
        current_torque_func = self.torque
        alpha = self.damping
        @nb.njit
        def torque_func(m, hext, Jc):
            total_torque = current_torque_func(m, hext, Jc)
            heff = field.uniaxial_anisotropy(m, u, hu1, hu2)
            total_torque += torque.landau_lifshitz(m, heff, alpha)
            return total_torque
        self.torque = torque_func

        # Change the energy function to inlcude uniaxial anisotropy
        current_energy_func = self.energy
        @nb.njit
        def energy_func(m, hext):
            return current_energy_func(m, hext) + energy.uniaxial_anisotropy(m, u, Ku1, Ku2)
        self.energy = energy_func
        # Initialize
        self.make_step()
        self.test_run()


    def add_cubic_anisotropy(self, c1, c2, Kc1, Kc2, Kc3):
        """ Add Cubic Anisotropy to the sample

        c1: In-plane orthogonal crystal direction 1 (unit vector)
        c2: In-plane orthogonal crystal direction 2 (unit vector)
        Kc1: Cubic anisotropy energy 1 (erg/cc)
        Kc2: Cubic anisotropy energy 2 (erg/cc)
        Kc3: Cubic anisotropy energy 3 (erg/cc)
        """
        # Normalize parameters
        c1 = normalize(np.array(c1,dtype=np.float32))
        c2 = normalize(np.array(c2,dtype=np.float32))
        Ms = self.Ms
        hc1 = normalize_energy(Kc1, Ms)
        hc2 = normalize_energy(Kc2, Ms)
        c3 = np.cross(c1,c2)

        # Change the field function to inlcude uniaxial anisotropy
        current_field_func = self.field
        @nb.njit
        def field_func(m, hext):
            return current_field_func(m, hext) + field.cubic_anisotropy(m, c1, c2, c3, hc1, hc2)
        self.field = field_func

        # Change the torque function to include demag field
        current_torque_func = self.torque
        alpha = self.damping
        @nb.njit
        def torque_func(m, hext, Jc):
            total_torque = current_torque_func(m, hext, Jc)
            heff = field.cubic_anisotropy(m, c1, c2, c3, hc1, hc2)
            total_torque += torque.landau_lifshitz(m, heff, alpha)
            return total_torque
        self.torque = torque_func

        # Change the energy function to inlcude uniaxial anisotropy
        current_energy_func = self.energy
        @nb.njit
        def energy_func(m, hext):
            return current_energy_func(m, hext) + energy.cubic_anisotropy(m, c1, c2, c3, Kc1, Kc2, Kc3)
        self.energy = energy_func
        # Initialize
        self.make_step()
        self.test_run()


    def add_spintorque(self, mp, P, Lambda=2, epsilon=0):
        """ Add spin torque

        mp: spin polarization unit vector
        P: polarization
        Lambda: spin transfer efficiency, default = 2
        epsilon: secondary spin transfer term (field-like torque), default = 0
        """
        alpha = self.damping
        mp = np.array(mp,dtype=np.float32)
        mp = normalize(mp)
        is_FL = (epsilon != 0)
        current_torque_func = self.torque
        @nb.njit
        def torque_func(m, hext, Jc):
            total_torque = current_torque_func(m, hext, Jc)
            total_torque += torque.slonczewski(m, mp, Jc, P, Lambda)
            if is_FL:
                total_torque += torque.slonczewski_fieldlike(m, mp, Jc, epsilon)
            return total_torque
        self.torque = torque_func
        # Initialize
        self.make_step()
        self.test_run()


    def make_step(self):
        """ Select run step function
        """
        name = self.method.lower()
        if   name=="euler": run_step = solvers.euler_step
        elif name=="huen":  run_step = solvers.huen_step
        elif name=="rk4":   run_step = solvers.rk4_step
        else:               run_step = solvers.rk23_step

        torque_func = self.torque
        dt = self.dt
        @nb.njit
        def step_func(m,hext,Jc):
            return run_step(torque_func, dt, m, hext, Jc)
        self.step_func = step_func


    def run(self, time_total, num_points=None):
        """ Run the simulation for a given time

        time_total: total time to simulate
        num_points: number of data points to be taken out
                    if None, return every point (may consume memory!)
        """
        time_total *= self.time_conversion
        steps = int(time_total/self.dt)
        if num_points is None:
            interval = 1
        else:
            interval = int(steps/num_points)
        # Execute the simulation
        moments = solvers.run(self.step_func, steps, self.m, self.hext, self.Jc)
        self.m = moments[-1]
        times = self.t + np.arange(1,steps+1)*self.dt
        self.t = times[-1]
        return times[::interval]/self.time_conversion, moments[::interval]


    def relax(self, precision=1e-3, iter_time=1e-9, max_time=1e-7):
        """ Run the simulation until energy variation falls within a threshold

        precision: energy's relative error for halting condition (default: 1e-3)
        iter_time: time per iteration (default: 1e-9)
        max_time: maximum simulation time (stopping condition) (default: 1e-7)
        """
        max_iters = int(max_time/iter_time) # Max number of iterations
        iter_time *= self.time_conversion
        steps = int(iter_time/self.dt)
        # Start stabilizing
        m_final, num_steps = solvers.relax(self.step_func, self.energy,
                        precision, steps, max_iters, self.m, self.hext, self.Jc)
        # Update status
        self.m = m_final
        self.t += num_steps*self.dt


    def stabilize(self, threshold=1e-3, iter_time=1e-9, max_time=1e-7):
        """ Run the simulation until dm falls within the threshold

        threshold: simulation stops when dm falls under this value (default: 1e-3)
        iter_time: time per iteration (default: 1e-9)
        max_time: maximum simulation time (stopping condition) (default: 1e-7)
        """
        max_iters = int(max_time/iter_time) # Max number of iterations
        iter_time *= self.time_conversion
        steps = int(iter_time/self.dt)
        # Start stabilizing
        m_final, num_steps = solvers.stabilize(self.step_func, self.torque,
                threshold, steps, max_iters, self.dt, self.m, self.hext, self.Jc)
        # Update status
        self.m = m_final
        self.t += num_steps*self.dt


    def energy_surface(self, points=100):
        """ Returns a numpy array of energy on unit sphere

        points: number of points (default: 100)
        """
        points = int(points)
        theta = np.linspace(0, np.pi, num=points, dtype=np.float32)
        phi = np.linspace(-np.pi, np.pi, num=points, dtype=np.float32)
        ms = np.zeros((points**2, 3), dtype=np.float32)
        energies = np.zeros(points**2, dtype=np.float32)
        m = np.zeros(3)

        for i in range(points):
            for j in range(points):
                m[0] = np.sin(theta[i])*np.cos(phi[j])
                m[1] = np.sin(theta[i])*np.sin(phi[j])
                m[2] = np.cos(theta[i])
                g = self.energy(m, self.hext)
                idx = points*i + j
                ms[idx] = m
                energies[idx] = g
        return ms, energies


    def torque_surface(self, points=100):
        """ Returns a numpy array of torque vectors on unit sphere

        points: number of points (default: 100)
        """
        points = int(points)
        theta = np.linspace(0, np.pi, num=points, dtype=np.float32)
        phi = np.linspace(-np.pi, np.pi, num=points, dtype=np.float32)
        ms = np.zeros((points**2, 3), dtype=np.float32)
        torques = np.zeros((points**2, 3), dtype=np.float32)
        m = np.zeros(3)

        for i in range(points):
            for j in range(points):
                m[0] = np.sin(theta[i])*np.cos(phi[j])
                m[1] = np.sin(theta[i])*np.sin(phi[j])
                m[2] = np.cos(theta[i])
                t = self.torque(m, self.hext,self.Jc)
                idx = points*i + j
                ms[idx] = m
                torques[idx] = t
        return ms, torques


    @property
    def t_sec(self):
        """ Returns the simulation time in seconds
        """
        return self.t/self.time_conversion
