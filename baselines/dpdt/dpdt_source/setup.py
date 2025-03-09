import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    name="dpdt",
    version="4.8",
    packages=["dpdt", "dpdt.solver", "dpdt.utils"],
    install_requires=["scipy", "scikit-learn", "numpy", "binarytree", "cython"],
    ext_modules=cythonize(
        Extension(
            name="dpdt.utils.cy_feature_select",
            sources=["dpdt/utils/cy_feature_select.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        annotate=False,
    ),
)
