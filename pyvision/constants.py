'''
Created on Dec 18, 2015

@author: sohara
'''
import os
import inspect

import pyvision as pv3

PACKAGE_DIR = os.path.dirname(inspect.getfile(pv3))
SAMPLE_DATA_DIR = os.path.join(PACKAGE_DIR, "..", "data")

# It's useful to have some built-in constants to sample data files
# to make demonstrations and tests easy.
IMG_DRIVEWAY = os.path.join(SAMPLE_DATA_DIR, "driveway.jpg")
IMG_PRIUS = os.path.join(SAMPLE_DATA_DIR, "prius_gray.jpg")
IMG_SLEEPYCAT = os.path.join(SAMPLE_DATA_DIR, "sleepycat.jpg")
VID_PRIUS = os.path.join(SAMPLE_DATA_DIR, "prius_movie.mov")

if __name__ == '__main__':
    pass
