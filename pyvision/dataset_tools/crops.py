"""
This module contains tools to help in extracting training crops
from images, where a "crop" is a relatively small rectangular ROI from
within the bounds of an image.

author: Stephen O'Hara
created: April 14, 2016
"""
import pyvision as pv3
import shapely.geometry as sg


def crop_regions(image, shapes, crop_size=None):
    """
    Crops are generated from within the provided image by
    using the bounding box around a set of provided shapes,
    or at a fixed size, centered on the centroids of the
    provided shapes.

    Parameters
    ----------
    image: pyvision image
    shapes: shapely polygons list
        Only the exterior coordinates of the polygons will be
        used to find either the bounding rectangle or the center
        point.
    crop_size: tuple (width, height) or None
        If None, then the bounding box for each shape will be
        used, and thus the generated crops may vary in size.
        Otherwise, if a single common size is desired, then
        a centered rectangle of this size will be extracted from
        the centroids of the shapes. It's possible with this latter
        strategy that you might have a shape too big for the crop_size.

    Returns
    -------
    A list of crops, where each crop is a pyvision image
    """

    if crop_size is not None:
        cw, ch = crop_size
        centers = [(shp.centroid.x, shp.centroid.y) for shp in shapes]
        rects = [pv3.CenteredRect(cx, cy, cw, ch) for (cx, cy) in centers]
    else:
        rects = [sg.box(*shp.bounds) for shp in shapes]

    crops = [image.crop(r) for r in rects]
    return crops
