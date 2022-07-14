# Copyright (C) 2022  Marcelo R. H. Maia <mmaia@ic.uff.br, marcelo.h.maia@ibge.gov.br>
# License: GPLv3 (https://www.gnu.org/licenses/gpl-3.0.html)


from setuptools import setup, Extension

import numpy
from Cython.Build import cythonize

extensions = [Extension('*', ['*.pyx'], libraries=['gsl', 'gslcblas'],
                        library_dirs=['../external_lib/gsl_x64-windows/lib'])]

setup(
    ext_modules=cythonize(extensions, language_level='2'),
    include_dirs=[numpy.get_include(), '../external_lib/gsl_x64-windows/include'],
    zip_safe=False,
)
