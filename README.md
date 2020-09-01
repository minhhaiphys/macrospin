# pymacrospin Python package #

Author: Colin Jermain, Minh-Hai Nguyen

The `pymacrospin` package simulates isolated macrospins in a simple and efficient way. The Landau-Lifshitz equation is solved by either a core Python or a numba kernel. Both are accessible directly from Python to allow easy simulations in and out of the IPython Notebook.

## Basic example of pymacrospin in applied field ##

```python
from pymacrospin.kernels import BasicKernel
# Or use numba kernel which is much faster
# from pymacrospin.numba.kernels import BasicKernel

parameters = {
    'Ms': 140, # Saturation Magnetization (emu/cc)
    'dt': 5e-13, # Timestep (sec)
    'damping': 0.01, # Gilbert damping
    'Hext': [0., 1e3, 0.], # External field (Oe)
    'm0': [-0.999, 0.001, 0.001], # Initial moment (normalized)
    'Nd': [0, 0, 0], # Demagnetization diagonal tensor elements
}

kernel = BasicKernel(**parameters)

# Run simulation for 20ns, read out 1000 data points
# times: Numpy array of simulation times
# moments: Numpy array of moment orientations
times, moments = kernel.run(time_total=2e-8, num_points=1000)
```
