'''
Created on Dec 18, 2015

@author: sohara
'''
import os
import inspect

import pyvision as pv3

PACKAGE_DIR = os.path.dirname(inspect.getfile(pv3))
SAMPLE_DATA_DIR = os.path.join(PACKAGE_DIR,"..","data")

IMG_DRIVEWAY = os.path.join(SAMPLE_DATA_DIR, "driveway.jpg")
IMG_PRIUS = os.path.join(SAMPLE_DATA_DIR, "prius_gray.jpg")

if __name__ == '__main__':
    pass