"""
This module contains tools to help in extracting training crops
from images, where a "crop" is a relatively small rectangular ROI from
within the bounds of an image.

author: Stephen O'Hara
created: April 14, 2016
"""
import pyvision3 as pv3
import shapely.geometry as sg
import shapely.ops as so
import numpy as np


def crop_regions(image, shapes, crop_size=None):
    """
    Crops are generated from within the provided image by
    using the bounding box around a set of provided shapes,
    or at a fixed size, centered on the centroids of the
    provided shapes.

    Parameters
    ----------
    image: pyvision3 image
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
    A list of crops, where each crop is a pyvision3 image
    """

    if crop_size is not None:
        cw, ch = crop_size
        centers = [(shp.centroid.x, shp.centroid.y) for shp in shapes]
        rects = [pv3.CenteredRect(cx, cy, cw, ch) for (cx, cy) in centers]
    else:
        rects = [sg.box(*shp.bounds) for shp in shapes]

    # crops = [image.crop(r) for r in rects]
    crops = []
    for r in rects:
        try:
            crop = image.crop(r)
        except pv3.OutOfBoundsError:
            print("{} is out of bounds in {}".format(str(r.bounds), image.desc))
            crop = None
        crops.append(crop)
    return crops


def crop_negative_regions(image, shapes, crop_size, N=10):
    """
    This function is useful for creating negative or 'background'
    samples from an image where you already have known foreground
    regions. Basically, it generates randomly located rectangles of
    the specified crop size in the image, then removes any that intersect
    with any of the foreground shapes. The user specifies how many
    negative samples to crop from the image.

    Parameters
    ----------
    image:  pyvision3 image
    shapes: shapely polygons list
        The shapes are the places NOT to crop negative samples from.
    crop_size: (w,h)
        The fixed size rectangles to be used for background crops
    N: integer
        The number of crops to generate from this image

    Returns
    -------
    A list of crops, where each is a pyvision3 image
    """
    positive_area = so.cascaded_union(shapes)
    validated_crops = []

    while len(validated_crops) < N:
        rect_gen = random_rect_gen(image.size, crop_size, N=N*2)
        for rect in rect_gen:
            if not rect.intersects(positive_area):
                validated_crops.append(image.crop(rect))
            if len(validated_crops) >= N:
                break

    return validated_crops


def random_rect_gen(image_size, crop_size, N=1):
    """
    Generates random rectangles (crop boundaries) of the specified size from within
    the image_size bounds.

    Parameters
    ----------
    image_size:  tuple
        The (w, h) of the image, serves as the container size from which the smaller
        crop rectangles will be generated
    crop_size: tuple
        The (w, h) of the crop rectangles, must be smaller than the image_size
    N: integer
        The number of random crop rects to create, default is 1

    Returns
    -------
    The rectangles as shapely polygons
    """
    img_w, img_h = image_size
    c_w, c_h = crop_size

    offset_x = c_w // 2
    offset_y = c_h // 2

    rand_xs = np.random.randint(offset_x, high=img_w-offset_x, size=N)
    rand_ys = np.random.randint(offset_y, high=img_h-offset_y, size=N)
    for (cx, cy) in zip(rand_xs, rand_ys):
        rect = pv3.CenteredRect(cx, cy, crop_size[0], crop_size[1])
        yield rect
