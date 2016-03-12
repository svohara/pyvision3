"""
Some simple geometry primitives (Point, Rect) that basically wrap
shapely geometries, but are used for convenience. For example,
shapely.geometry.box takes (minx, miny, maxx, maxy) coordinates, but
it's often much more convenient to specify a rectangle as (x, y, width, height)...
"""

import shapely.geometry as sg
from shapely.geometry.point import Point as sgPoint


def Point(x, y):
    """
    Creates a shapely Point object from the input x,y coords.

    Parameters
    ----------
    x:  int
    y:  int

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
    return sg.box(x, y, x+w, h+y)

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