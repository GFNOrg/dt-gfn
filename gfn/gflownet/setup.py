import os
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

print("Starting setup process...")
NAME = "gflownet"
VERSION = "0.1.0"
DESCRIPTION = "GFlowNet implementation with Cython acceleration"


if sys.platform == "darwin":
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]
elif sys.platform == "win32":
    extra_compile_args = ["/O2", "/openmp"]
    extra_link_args = []
else:
    extra_compile_args = ["-O3", "-fopenmp"]
    extra_link_args = ["-fopenmp"]


# Ensure the target directory exists
os.makedirs("envs", exist_ok=True)
print(f"Current directory: {os.getcwd()}")
print(f"Contents of current directory: {os.listdir('.')}")
print(f"Contents of envs directory: {os.listdir('envs')}")

extensions = [
    Extension(
        "envs.tree_acc_cython",
        ["envs/tree_acc_cython.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

print(f"Extensions: {extensions}")

setup(
    name="tree_acc_cython",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
)

print("Setup process completed.")
