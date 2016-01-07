"""
Created on Dec 18, 2015

@author: sohara
"""
import os
import inspect
from pkg_resources import resource_filename, Requirement
import pyvision as pv3

PACKAGE_DIR = os.path.dirname(inspect.getfile(pv3))

# It's useful to have some built-in constants referring to
# sample data files to make demonstrations and tests easy.
#tmp = Requirement.parse(__name__)
IMG_DRIVEWAY = resource_filename("pyvision", "data/driveway.jpg")
IMG_PRIUS = resource_filename("pyvision", "data/prius_gray.jpg")
IMG_SLEEPYCAT = resource_filename("pyvision", "data/sleepycat.jpg")
VID_PRIUS = resource_filename("pyvision", "data/prius_movie.mov")

if __name__ == '__main__':
    pass
