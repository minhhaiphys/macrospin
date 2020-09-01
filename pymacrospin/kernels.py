#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Kernel classes for efficiently evolving the macrospin
# Core kernels: Use Python and Scipy/Numpy packages
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
import numpy as np

from pymacrospin.__init__ import normalize
from pymacrospin.core import field, torque, solvers, energy
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
        # Set up exec functions
        self.make_field()
        self.make_torque()
        self.make_energy()
        self.make_step()


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

        def step_func(m):
            return run_step(self.dt,m,self.torque)
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
        moments = solvers.run(self.step_func, steps, self.m)
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
                            precision, steps, max_iters, self.m)
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
                            threshold, steps, max_iters, self.m, self.dt)
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
                g = self.energy(m)
                idx = points*i + j
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

    def make_field(self):
        """ Define function to calculate effective field
        = sum of applied field and demag field
        """
        def field_func(m):
            return self.hext + field.demagnetization(m, self.Nd)
        self.field = field_func


    def make_torque(self):
        """ Define function to calculate torque
        = Landau_Lifshitz torque
        """
        def torque_func(m):
            heff = self.field(m)
            total_torque = torque.landau_lifshitz(m, heff, self.damping)
            if self.stt != 0:
                total_torque += torque.slonczewski(m, self.Jc, self.stt)
            return total_torque
        self.torque = torque_func

    def make_energy(self):
        """ Define function to calculate energy
        = Zeeman energy
        """
        def energy_func(m):
            heff = self.field(m)
            return -energy.zeeman(m, self.Ms, heff) \
                    + energy.shape_anisotropy(m, self.Ms,
                                        self.Nd[0], self.Nd[1], self.Nd[2])
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
        self.c3 = np.cross(self.c1,self.c2)


    def make_field(self):
        """ Define function to calculate effective field
        = sum of applied field and demag field
        """
        uniaxial = self.u[0]*self.u[1]*self.u[2] != 0
        cubic = self.c1[0]*self.c1[1]*self.c1[2]*self.c2[0]*self.c2[1]*self.c2[2] != 0
        @nb.njit
        def field_func(m):
            heff = self.hext + field.demagnetization(m, self.Nd)
            if uniaxial:
                heff += field.uniaxial_anisotropy(m, self.u, self.hu1, self.hu2)
            if cubic:
                heff += field.cubic_anisotropy(m, self.c1, self.c2, self.c3, self.hc1, self.hc2)
            return heff
        self.field = field_func


    def make_energy(self):
        """ Define function to calculate energy
        = Zeeman energy + demagnetization energy + anisotropy energies
        """
        @nb.njit
        def energy_func(m):
            heff = self.field(m)
            return -energy.zeeman(m, self.Ms, heff) \
                    + energy.shape_anisotropy(m, self.Ms, self.Nd[0], self.Nd[1], self.Nd[2]) \
                    + energy.uniaxial_anisotropy(m, self.u, self.Ku1, self.Ku2) \
                    + energy.cubic_anisotropy(m, self.c1, self.c2, self.c3,
                                                self.Kc1, self.Kc2, self.Kc3)
        self.energy = energy_func
