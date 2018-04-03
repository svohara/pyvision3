"""
Some simple geometry primitives (Point, Rect) that basically wrap
shapely geometries, but are used for convenience. For example,
shapely.geometry.box takes (minx, miny, maxx, maxy) coordinates, but
it's often much more convenient to specify a rectangle as (x, y, width, height)...
"""

import shapely.geometry as sg
from shapely.geometry.point import Point as sgPoint
import numpy as np


def Point(x, y):
    """
    Creates a shapely Point object from the input x,y coords.

    Parameters
    ----------
    x:  x-coordinate
    y:  y-coordinate

    Returns
    -------
    shapely.geometry.point.Point object
    """
    return sgPoint(x, y)


def Rect(x, y, w, h):
    """

    Parameters
    ----------
    x: int
        Upper left corner, x position
    y: int
        Upper left corner, y position
    w: int
        Width
    h: int
        Height

    Returns
    -------
    shapely.geometry.Polygon object representing the rectangle
    """
    return sg.box(x, y, x+w-1, h+y-1)


def CenteredRect(cx, cy, w, h):
    """
    Construct a rectangle shape from the center point
    and the width and height.

    Parameters
    ----------
    cx: center point, x
    cy: center point, y
    w: width
    h: height

    Returns
    -------
    shapely shape (polygon) object
    """
    x = cx - w//2
    y = cy - h//2
    return Rect(x, y, w, h)


def in_bounds(rect, image):
    """
    Returns true if the rectangle is entirely within the bounds of the image.

    Parameters
    ----------
    rect: shapely rectangle with integer coordinates, as per this module's Rect() output.
    image: pyvision3 image

    Returns
    -------
    Boolean, true if no part of rect is outside the bounds of image.
    """
    image_bounds = Rect(0, 0, image.width, image.height)
    return image_bounds.contains(rect)


def integer_coords_array(shape):
    """
    Returns the coordinates (for a point/line) or the exterior coordinates (for a polygon)
    as an numpy array, forcing the coordinates to integers.

    Parameters
    ----------
    shape: a shapely geometry that has either a .coords or .exteror.coords member

    Returns
    -------
    coordinates as an ndarray with integer values
    """
    try:
        coords = shape.exterior.coords
    except AttributeError:
        coords = shape.coords
    return np.array(coords).astype('int')


def integer_bounds(shape):
    """
    Returns the bounds of the given shape using integer values.

    Parameters
    ----------
    shape: a shapely geometry supporting the .bounds attribute

    Returns
    -------
    (minx, miny, maxx, maxy) as integer values.
    """
    return tuple(np.array(shape.bounds, dtype='int'))
