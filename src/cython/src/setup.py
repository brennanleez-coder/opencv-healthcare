from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension module
extensions = [
    Extension(
        "sit_stand_overall",  # name of the generated module
        ["cython_src/sit_stand_overall.pyx"],  # list of source files
        include_dirs=[numpy.get_include()],  # include numpy headers
        language="c++"  # generate C++ code
    )
]

# Define the setup
setup(
    name="sit_stand_overall",
    ext_modules=cythonize(extensions),
    language_level="3"
)
