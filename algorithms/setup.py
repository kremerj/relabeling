from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("_logistic_noise_sigmoid",
                 sources=["_logistic_noise_sigmoid.pyx"],
                 include_dirs=[numpy.get_include()])],
)
