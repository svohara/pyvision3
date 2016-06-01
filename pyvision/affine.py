"""
This module provides convenient wrappers to apply or construct
affine transformations from pyvision images.
"""

import cv2
import pyvision as pv3


class AffineTransformer(object):
    """
    This class defines a callable object that can be applied to
    pyvision images to apply an affine or inverse affine transformation.
    """
    def __init__(self, affine_matrix):
        """
        Constructor

        Parameters
        ----------
        affine_matrix: numpy array (2x3)

        """
        self.affine_matrix = affine_matrix

    def __call__(self, source_img, dest_size=None, invert=False):
        """

        Parameters
        ----------
        source_img: pyvision image
        dest_size: tuple (w, h)
            If None, then destination size will be set as source_img size
        invert: boolean
            If true, the inverse transformation is applied to the source_img
            instead of the forward transformation.

        Returns
        -------
        A pyvision image resulting from applying the transformation
        to the source_img.
        """

        input_array = source_img.data
        if dest_size is None:
            dest_size = source_img.size
        warped = cv2.warpAffine(input_array, self.affine_matrix, dest_size)
        return pv3.Image(warped)
