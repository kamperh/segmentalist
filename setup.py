from distutils.core import setup, Extension
from Cython.Distutils import build_ext

import numpy as np

ext = Extension(
    "segmentalist._cython_utils",
    ["segmentalist/_cython_utils.pyx"],
    include_dirs=[np.get_include()]
    )

setup(
    cmdclass = {"build_ext": build_ext},
    ext_modules = [ext],
    packages=["segmentalist"],
    package_dir={"segmentalist": "segmentalist"}
    )
