from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
import sysconfig

# Get include directories for Python and NumPy
include_dirs = [
    sysconfig.get_paths()["include"],  # This fetches the Python include directory
    numpy.get_include(),
]

setup(
    ext_modules=cythonize("/Users/brennanlee/Desktop/opencv-healthcare/Cython_algorithms/sit_stand_overall.pyx"),
    include_dirs=include_dirs,    
)
