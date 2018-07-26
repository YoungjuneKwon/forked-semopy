from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy
import os


def find_pyx(path='SEMOpt'):
    pyx_files = []
    for root, dirs, filenames in os.walk(path):
        for fname in filenames:
            if fname.endswith('.pyx'):
                pyx_files.append(os.path.join(root, fname))
    return pyx_files


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      name="SEMOpt",
      version="1.0.0",
      author="Meshcheryakov Georgy Andreyevich",
      author_email="metsheryakov_ga@spbstu.ru",
      description="Structural Equation Modelling optimization package.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://https://bitbucket.org/herrberg/semopt/",
      ext_modules=cythonize(find_pyx(), language_level=3),
      include_dirs=[numpy.get_include()],
      packages=find_packages(),
      classifiers=(
        	   "Programming Language :: Python :: 3",
        	   "License :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent",
    		   ),
)