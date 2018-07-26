from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
      install_requires=['portmin', 'scipy', 'numpy', 'pandas'],
      name="semopt",
      version="1.0.2",
      author="Meshcheryakov Georgy Andreyevich",
      author_email="metsheryakov_ga@spbstu.ru",
      description="Structural Equation Modelling optimization package.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://bitbucket.org/herrberg/semopt/",
      packages=find_packages(),
      classifiers=(
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: MIT License",
              "Operating System :: OS Independent"))
