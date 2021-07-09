from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

# Setting up
setup(
        name="PandaZam", 
        description=Song recognition,
        install_requires=[numpy, matplotlib, numba, typing, pyaudio, ffmeg],

)