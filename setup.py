# contents of setup.py
import setuptools

PROJECT_NAME = "example_project"  # change this!

setuptools.setup(
    name=PROJECT_NAME,
    version="1.0",
    packages=setuptools.find_packages(),
)