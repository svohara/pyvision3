"""
Created on Dec 18, 2015
@author: Stephen O'Hara
"""
import os
import inspect
from pkg_resources import resource_filename
import pyvision3 as pv3
import shapely.geometry as sg
import json


INFO_JSON_FILE = resource_filename('pyvision3', os.path.join('data', 'project_info.json'))
with open(INFO_JSON_FILE, 'r') as infile:
    _info = json.load(infile)
VERSION_TUPLE = _info['version_tuple']

PACKAGE_DIR = os.path.dirname(inspect.getfile(pv3))

# Common RGB colors (for use with pyvision3 annotations)
RGB_BLACK = (0, 0, 0)  # not zeros, because that can be confusing with masks
RGB_GRAY = (128, 128, 128)
RGB_WHITE = (255, 255, 255)
RGB_BLUE = (0, 0, 255)
RGB_LTBLUE = (135, 205, 250)
RGB_TEAL = (0, 128, 128)
RGB_CYAN = (0, 255, 255)
RGB_GREEN = (0, 255, 0)
RGB_YELLOW = (255, 255, 0)
RGB_ORANGE = (255, 150, 0)
RGB_BROWN = (140, 70, 20)
RGB_RED = (255, 0, 0)
RGB_PINK = (255, 105, 180)
RGB_PURPLE = (150, 110, 220)
RGB_MAGENTA = (255, 0, 255)
RGB_KHAKI = (240, 230, 140)

RGB_COLORS = {"black": RGB_BLACK, "white": RGB_WHITE, "gray": RGB_GRAY,
              "blue": RGB_BLUE, "light blue": RGB_LTBLUE, "teal": RGB_TEAL,
              "cyan": RGB_CYAN, "green": RGB_GREEN, "yellow": RGB_YELLOW,
              "orange": RGB_ORANGE, "brown": RGB_BROWN, "red": RGB_RED,
              "pink": RGB_PINK, "purple": RGB_PURPLE, "magenta": RGB_MAGENTA,
              "khaki": RGB_KHAKI}

# Common colors as BGR tuples (for use with cv2 functions)
BGR_BLACK = (0, 0, 0)
BGR_GRAY = (128, 128, 128)
BGR_WHITE = (255, 255, 255)
BGR_BLUE = (255, 0, 0)
BGR_LTBLUE = (250, 205, 135)
BGR_TEAL = (128, 128, 0)
BGR_CYAN = (255, 255, 0)
BGR_GREEN = (0, 255, 0)
BGR_YELLOW = (0, 255, 255)
BGR_ORANGE = (0, 150, 255)
BGR_BROWN = (20, 70, 140)
BGR_RED = (0, 0, 255)
BGR_PINK = (180, 105, 255)
BGR_PURPLE = (220, 110, 150)
BGR_MAGENTA = (255, 0, 255)
BGR_KHAKI = (140, 230, 240)

BGR_COLORS = {"black": BGR_BLACK, "white": BGR_WHITE, "gray": BGR_GRAY,
              "blue": BGR_BLUE, "light blue": BGR_LTBLUE, "teal": BGR_TEAL,
              "cyan": BGR_CYAN, "green": BGR_GREEN, "yellow": BGR_YELLOW,
              "orange": BGR_ORANGE, "brown": BGR_BROWN, "red": BGR_RED,
              "pink": BGR_PINK, "purple": BGR_PURPLE, "magenta": BGR_MAGENTA,
              "khaki": BGR_KHAKI}

#  It's useful to have some built-in constants referring to
#  sample data files to make demonstrations and tests easy.
IMG_DRIVEWAY = resource_filename("pyvision3", "data/driveway.jpg")
IMG_PRIUS = resource_filename("pyvision3", "data/prius_gray.jpg")
IMG_SLEEPYCAT = resource_filename("pyvision3", "data/sleepycat.jpg")
IMG_MASK = resource_filename("pyvision3", "data/test_mask.tif")
IMG_MASK_RESULT = resource_filename("pyvision3", "data/test_mask_result2.tif")
VID_PRIUS = resource_filename("pyvision3", "data/prius_movie.mov")

# A sample polygon roughly outlining the sleepy cat
COORDS_SLEEPYCAT = [(285, 460), (408, 323), (545, 368), (561, 352), (530, 280), (645, 184), (714, 240),
                    (713, 114), (784, 91), (808, 286), (871, 450), (816, 651), (664, 735), (428, 656),
                    (388, 573), (350, 565), (329, 516), (277, 493)]
POLY_SLEEPYCAT = sg.Polygon(COORDS_SLEEPYCAT)

if __name__ == '__main__':
    pass
