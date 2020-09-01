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

from pymacrospin.numba.__init__ import normalize
from pymacrospin.numba import field, torque, solvers, energy
from pymacrospin.parameters import CgsParameters, MksParameters, NormalizedParameters


class Kernel:
    """ Encapsulates the time evolution algorithm for solving the
    Landau-Liftshitz equation

    The following parameters can be provided:
    unit: 'CGS' (default) or 'MKS'
    method: ODE solver, either 1-Euler, 2-Huen, 3-RK23 (default) or 4-RK4
    dt: time step
    m0: initial moment
    Ms: saturation magnetization
    damping: Gilbert damping constant
    Hext: external field
    Jc: current density
    stt: torque prefactor (pre-calculated)
    """

    def __init__(self, **parameters):
        self.raw_parameters = {'unit': "CGS",
                                'method': "RK23",
                                'm0': [1,0,0],
                                'dt': 1e-12
                                }
        self.method = "RK23"
        self.update_params(**parameters)
        self.reset()


    def _load(self):
        """ Load the parameters by translate the dict to kernel attributes
        """
        for k,v in self.parameters.items():
            if isinstance(v,list):
                setattr(self,k,np.array(v,dtype=np.float32))
            else:
                setattr(self,k,v)


    def update_params(self,**params):
        """ Update raw parameters

        params: keyword parameters
        """
        self.raw_parameters.update(params)
        if self.raw_parameters['unit'].lower()=="mks":
            raise ValueError("MKS is not yet supported.")
            new_params = MksParameters()
        else: # defaul is CGS
            new_params = CgsParameters()
        new_params.update(self.raw_parameters)
        self.raw_parameters = new_params
        self.parameters = NormalizedParameters(self.raw_parameters)
        self._load()
        # Set up exec functions
        self.make_field()
        self.make_torque()
        self.make_energy()
        self.make_step()


    def set_field(self,Hext):
        """ Set the external field without changing other parameters

        Hext: externally applied field
        """
        self.raw_parameters["Hext"] = Hext
        self.parameters = NormalizedParameters(self.raw_parameters)
        self._load()


    def set_current(self,Jc):
        """ Set the current density without changing other parameters

        Jc: current density
        """
        self.raw_parameters["Jc"] = Jc
        self.parameters = NormalizedParameters(self.raw_parameters)
        self._load()


    def reset(self):
        """ Resets the kernel to the initial conditions
        """
        self.m = normalize(self.m0)
        self.t = 0.0


    def make_field(self):
        """ Define field function
        """
        self.field = lambda x: x


    def make_torque(self):
        """ Define torque function
        """
        self.torque = lambda x: x


    def make_energy(self):
        """ Define energy function
        """
        self.energy = lambda x: x


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
        """ Returns a numpy array of vectors that have a length of the
        energy at their orientation

        points: number of points (default: 100)
        """
        points = int(points)
        theta = np.linspace(0, np.pi, num=points, dtype=np.float32)
        phi = np.linspace(-np.pi, np.pi, num=points, dtype=np.float32)
        energies = np.zeros((points**2, 3), dtype=np.float32)
        m = np.zeros(3)

        for i in range(points):
            for j in range(points):
                m[0] = np.sin(theta[i])*np.cos(phi[j])
                m[1] = np.sin(theta[i])*np.sin(phi[j])
                m[2] = np.cos(theta[i])
                g = self.energy(m, self.hext)
                idx = n*i + j
                energies[idx][0] = g*m[0]
                energies[idx][1] = g*m[1]
                energies[idx][2] = g*m[2]
        return energies


    @property
    def t_sec(self):
        """ Returns the simulation time in seconds
        """
        return self.t/self.parameters['time_conversion']



#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Basic Kernel
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


class BasicKernel(Kernel):
    """ Basic Kernel for solving the Landau-Liftshitz equation

    unit: 'CGS' (default) or 'MKS'
    method: ODE solver, either 1-Euler, 2-Huen, 3-RK23 (default) or 4-RK4
    dt: time step
    Ms: saturation magnetization
    Hext: external field
    Nd: Demagnetization diagonal tensor elements
    m0: initial moment
    damping: Gilbert damping constant
    """
    def __init__(self, **parameters):
        self.Nd = np.array([0,0,0],dtype=np.float32)
        self.Ms = 1
        self.damping = 0
        self.hext = np.array([0,0,0],dtype=np.float32)
        self.Jc = np.array([0,0,0],dtype=np.float32)
        self.stt = 0
        super(BasicKernel,self).__init__(**parameters)
        self.run(2*self.dt/self.time_conversion) # Move compiling overhead here
        self.reset()

    def make_field(self):
        """ Define function to calculate effective field
        = sum of applied field and demag field
        """
        Nd = self.Nd
        @nb.njit
        def field_func(m, hext):
            return hext + field.demagnetization(m, Nd)
        self.field = field_func


    def make_torque(self):
        """ Define function to calculate torque
        = Landau_Lifshitz torque
        """
        field_func = self.field
        alpha = self.damping
        is_stt = (self.stt != 0)
        Jc = self.Jc
        stt = self.stt
        @nb.njit
        def torque_func(m, hext, Jc):
            heff = field_func(m, hext)
            total_torque = torque.landau_lifshitz(m, heff, alpha)
            if is_stt:
                total_torque += torque.slonczewski(m, Jc, stt)
            return total_torque
        self.torque = torque_func

    def make_energy(self):
        """ Define function to calculate energy
        = Zeeman energy + demagnetization energy
        """
        field_func = self.field
        Ms = self.Ms
        Nxx = self.Nd[0]
        Nyy = self.Nd[1]
        Nzz = self.Nd[2]
        @nb.njit
        def energy_func(m, hext):
            heff = field_func(m, hext)
            return -energy.zeeman(m, Ms, heff) \
                    + energy.shape_anisotropy(m, Ms, Nxx, Nyy, Nzz)
        self.energy = energy_func


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Anisotropy Kernel
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class AnisotropyKernel(BasicKernel):
    """ Kernel for sample having uniaxial or cubic anisotropy

    unit: 'CGS' (default) or 'MKS'
    method: ODE solver, either 1-Euler, 2-Huen, 3-RK23 (default) or 4-RK4
    dt: time step
    Ms: saturation magnetization
    Hext: external field
    Nd: Demagnetization diagonal tensor elements
    m0: initial moment
    damping: Gilbert damping constant
    """
    def __init__(self, **parameters):
        init_params = {
            'Kc1': 0.0,
            'Kc2': 0.0,
            'Kc3': 0.0,
            'Ku1': 0.0,
            'Ku2': 0.0,
        }
        init_params.update(parameters)
        self.u = np.array([0,0,0],dtype=np.float32)
        self.hu1 = 0
        self.hu2 = 0
        self.c1 = np.array([0,0,0],dtype=np.float32)
        self.c2 = np.array([0,0,0],dtype=np.float32)
        self.hc1 = 0
        self.hc2 = 0
        super(AnisotropyKernel,self).__init__(**init_params)


    def make_field(self):
        """ Define function to calculate effective field
        = sum of applied field and demag field
        """
        Nd = self.Nd
        u = self.u
        hu1 = self.hu1
        hu2 = self.hu2
        c1 = self.c1
        c2 = self.c2
        c3 = np.cross(c1,c2)
        hc1 = self.hc1
        hc2 = self.hc2
        uniaxial = u[0]*u[1]*u[2] != 0
        cubic = c1[0]*c1[1]*c1[2]*c2[0]*c2[1]*c2[2] != 0
        @nb.njit
        def field_func(m, hext):
            heff = hext + field.demagnetization(m, Nd)
            if uniaxial:
                heff += field.uniaxial_anisotropy(m, u, hu1, hu2)
            if cubic:
                heff += field.cubic_anisotropy(m, c1, c2, c3, hc1, hc2)
            return heff
        self.field = field_func


    def make_energy(self):
        """ Define function to calculate energy
        = Zeeman energy + demagnetization energy + anisotropy energies
        """
        field_func = self.field
        Ms = self.Ms
        Nxx = self.Nd[0]
        Nyy = self.Nd[1]
        Nzz = self.Nd[2]
        u = self.u
        Ku1 = self.raw_parameters["Ku1"]
        Ku2 = self.raw_parameters["Ku2"]
        Kc1 = self.raw_parameters["Kc1"]
        Kc2 = self.raw_parameters["Kc2"]
        Kc3 = self.raw_parameters["Kc3"]
        c1 = self.c1
        c2 = self.c2
        c3 = np.cross(c1,c2)
        @nb.njit
        def energy_func(m, hext):
            heff = field_func(m, hext)
            return -energy.zeeman(m, Ms, heff) \
                    + energy.shape_anisotropy(m, Ms, Nxx, Nyy, Nzz) \
                    + energy.uniaxial_anisotropy(m, u, Ku1, Ku2) \
                    + energy.cubic_anisotropy(m, c1, c2, c3, Kc1, Kc2, Kc3)
        self.energy = energy_func
