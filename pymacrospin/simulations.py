#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Simulation class for threading Kernel objects
#
# pymacrospin Python package
# Authors: Colin Jermain, Minh-Hai Nguyen
# Copyright: 2014-2020
#
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

from pymacrospin.__init__ import sphere2cartesian
import numpy as np
from threading import Thread

class Simulation(object):
    """ Simulation object coordinates the Thread that runs the kernel
    """
    FAILED, STOPPED, QUEUED, RUNNING = 0, 1, 2, 3

    def __init__(self, kernel):
        self.kernel = kernel
        self.status = Simulation.QUEUED
        self.thread = None

    def run(self, time, timeout=None, thread_timeout=1e5):
        """ Runs the simulation for a given simulation time in seconds
        """
        if self.status == Simulation.RUNNING:
            raise Exception("Simulation is already running")
        elif self.status == Simulation.FAILED:
            raise Exception("Can not run a simulation that has failed")
        self.status = Simulation.RUNNING
        self.thread = Thread(target=self.kernel, kwargs={
                            'time': time, 'timeout': timeout})
        self.thread.start()

    def isRunning(self):
        """ Returns True if an existing Thread is running
        """
        if self.thread is None:
            return False
        else:
            return self.thread.isAlive()

    def wait(self, thread_timeout):
        self.thread.join(thread_timeout)
        self.status = Simulation.STOPPED

    def stop(self):
        """ Stops the simulation even if running
        """
        self.kernel.stop()

    def resume(self, time, thread_timeout=1e5):
        """ Resumes the simulation from the last moment orientation
        """
        self.run(time, None, thread_timeout)

    def stabilize(self, timeout=1e-4, thread_timeout=1e5):
        """ Runs the simulation until the moment is stable or the timeout
        is reached in simulation time
        """
        self.run(None, timeout, thread_timeout)


class FieldSweep(object):
    """
    """
    def __init__(self, kernel, fields=np.array([0,0,0])):
        self.kernel = kernel
        self.fields = fields


    def sweep_linear(self, start_field, end_field, points=1e3, reverse=True, **kwargs):
        """ Sweep the applied field that go from the start_field
        to the end_field, with the default option to also include the reverse

        start_field: starting field
        end_field: ending field
        points: number of field points (including the start_ and end_field)
        reverse: whether to add sweeping the reverse direction

        Extra keyword params to be passed to the kernels.run function:
        return_time: whether to return the relaxation time (default: False)
        precision: energy's relative error for halting condition (default: 1e-3)
        iter_time: time per iteration (default: 1e-9)
        max_time: maximum simulation time (stopping condition) (default: 1e-7)
        """
        fields = np.linspace(start_field, end_field, num=int(points), dtype=np.float32)
        if reverse:
            fields = np.concatenate([fields,fields[::-1]],axis=0)
        self.fields = fields
        return self.run(**kwargs)


    def sweep_rotation(self, field, theta, phi, coupled=False, reverse=False, **kwargs):
        """ Sweep the applied field (fixed magnitude) in spherical theta and phi angles,
        with the default option to NOT include the reverse.
        If coupled: sweep theta and phi at the same time (have to be of equal length)
        If not coupled and both theta and phi have more than one elements,
        we sweep phi first then theta.

        field: field magnitude (preferably positive)
        theta: sweeping spherical theta [deg]
        phi: sweeping spherical phi [deg]
        coupled: whether to sweep theta and phi at the same time
        reverse: whether to add sweeping the reverse direction

        Extra keyword params to be passed to the kernels.run function:
        return_time: whether to return the relaxation time (default: False)
        precision: energy's relative error for halting condition (default: 1e-3)
        iter_time: time per iteration (default: 1e-9)
        max_time: maximum simulation time (stopping condition) (default: 1e-7)
        """
        # Convert theta and phi to array if not already so
        if len(np.array(theta).shape)==0:
            theta = np.array([theta])
        if len(np.array(phi).shape)==0:
            phi = np.array([phi])
        # Convert angles to radians
        theta = np.array(theta)*np.pi/180
        phi = np.array(phi)*np.pi/180

        fields = []
        if coupled:
            if len(theta)!=len(phi):
                raise ValueError("Lengths of theta and phi must match: %d != %d"
                                %(len(theta),len(phi)))
            for t,p in zip(theta,phi):
                fields.append(field*sphere2cartesian(t,p))
        else:
            # We alway sweep phi first, then theta
            for t in theta:
                for p in phi:
                    fields.append(field*sphere2cartesian(t,p))
        if reverse:
            fields = np.concatenate([fields,fields[::-1]],axis=0)
        self.fields = np.array(fields)
        return self.run(**kwargs)


    def run(self, return_time=False, **kwargs):
        """ Runs through each field and stabilizes the moment, returning
        the fields, stabilization time, and moment orientation

        return_time: whether to return the relaxation time (default: False)

        Extra keyword params to be passed to the kernels.run function:
        precision: energy's relative error for halting condition (default: 1e-3)
        iter_time: time per iteration (default: 1e-9)
        max_time: maximum simulation time (stopping condition) (default: 1e-7)
        """
        size = self.fields.shape[0]
        times = np.zeros((size, 1), dtype=np.float32)
        moments = np.zeros((size, 3), dtype=np.float32)
        self.kernel.reset()

        for i, field in enumerate(self.fields):
            ti = self.kernel.t_sec
            self.kernel.set_field(field)
            self.kernel.relax(**kwargs)
            times[i] = self.kernel.t_sec - ti
            moments[i] = self.kernel.m
        if return_time:
            return self.fields, moments, times
        else:
            return self.fields, moments
