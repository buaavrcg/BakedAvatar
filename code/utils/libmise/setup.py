from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

# mise (efficient mesh extraction)
mise_module = Extension(
    "mise",
    sources=["mise.pyx"],
)

# Gather all extension modules
ext_modules = [
    mise_module,
]

setup(name="libmise", ext_modules=cythonize(ext_modules),)