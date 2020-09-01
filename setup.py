from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("pymacrospin.cython.kernels",
        ["pymacrospin/cython/kernels.pyx"],
        language='c++',
        include_dirs=[np.get_include(), '.'],
    ),
]

setup(
    name='pymacrospin',
    version='0.2.0',
    author='Colin Jermain, Minh-Hai Nguyen',
    packages=['pymacrospin'],
    scripts=[],
    license='MIT License',
    description='Macrospin simulations using Python',
    long_description=open('README.md').read(),
    # ext_modules=cythonize(extensions),
    # package_data={'macrospin': ['*.pxd', '*.h']},
)
