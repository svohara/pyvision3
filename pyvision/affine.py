"""
This module provides convenient wrappers to apply or construct
affine transformations from pyvision images.
"""

import cv2
import pyvision as pv3
import math
import numpy as np


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
        dest_size: tuple (w, h) or 'fit'
            If None, then destination size will be set as source_img size.
            If 'fit', then the destination will be large enough to fit all
            the original pixels. NOTE: this could make the image quite huge!
            Also, if 'fit' then the affine transform will be modified to
            center the result within the new bounds...which is not what
            you'd want for a pure translation...
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
        elif dest_size == 'fit':
            # figure out max size of image by taking transformations
            # of the four corners
            maxx = source_img.size[0] - 1
            maxy = source_img.size[1] - 1
            points = np.array([[0, 0, 1], [0, maxy, 1], [maxx, maxy, 1], [maxx, 0, 1]]).T
            aug_mat = self.get_augmented_matrix()
            new_points = np.dot(aug_mat, points).astype('int')
            fit_width = new_points[0, :].max() - new_points[0, :].min()
            fit_height = new_points[1, :].max() - new_points[1, :].min()
            dest_size = (fit_width+1, fit_height+1)
            self.affine_matrix[0, 2] -= new_points[0, :].min()
            self.affine_matrix[1, 2] -= new_points[1, :].min()

        flags = cv2.WARP_INVERSE_MAP if invert else 0
        warped = cv2.warpAffine(input_array, self.affine_matrix, dest_size, flags=flags)

        return pv3.Image(warped)

    def get_augmented_matrix(self):
        """
        Returns
        -------
        The affine transformation matrix in 'augmented' form, as 3x3 ndarray
        """
        aug_row = np.array([[0.0, 0.0, 1.0]])
        return np.vstack((self.affine_matrix, aug_row))

    def combine(self, aff2):
        """
        Combines the input affine transformation with the current, and returns a new
        AffineTransformer object that is the product of the two.

        Example:
        t1 = AffineTranslate(20, 30)
        t2 = t1.combine( AffineTranslate(-20, -30) )
        t2 would be the identity transformation

        Parameters
        ----------
        aff2: AffineTransformer (or subclass)

        Returns
        -------
        AffineTransformer resulting from multiplying the (augmented) matrices
        of aff2 and self.
        """
        m1 = self.get_augmented_matrix()
        m2 = aff2.get_augmented_matrix()
        m3 = np.dot(m2, m1)
        aff_mat = m3[0:2, :]  # strip the augmented row now...
        return AffineTransformer(aff_mat)


class AffineTranslate(AffineTransformer):
    """
    Convenience subclass of AffineTransformer for performing simple
    translation by dx, dy.
    """
    def __init__(self, dx, dy):
        """
        Parameters
        ----------
        dx: int or float
        dy: int or float
        """
        translate_mat = np.array([[1, 0, dx],
                                  [0, 1, dy]], dtype='float32')
        AffineTransformer.__init__(self, translate_mat)


class AffineRotation(AffineTransformer):
    """
    Convenience class to construct an affine transformation that
    rotates an image clockwise about the center of the image by
    theta degrees.

    Note, for 90, 180, 270 rotations, it would be much faster
    to apply matrix flipping and transposing.
    """
    def __init__(self, theta_degrees, image_size):
        """
        Constructor

        Parameters
        ----------
        theta_degrees: int or float
            How many degrees clockwise about the center should the
            image be rotated?
        img_size: tuple (w,h)
            The size of the image, from which we can determine how
            to rotate about the center
        """
        self.theta_degrees = theta_degrees
        self.theta = float(theta_degrees)*math.pi/180.0
        self.image_size = image_size
        self.center = tuple(np.array(image_size, dtype='float32')/2)

        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        origin_rotation = np.array([[cos_t, -sin_t, 0],
                                    [sin_t, cos_t, 0],
                                    [0, 0, 1]])

        # we are rotating about the center, so we
        # first translate the center to the origin,
        # then apply the origin rotation
        # then translate the origin back to the original
        # image center
        trans_1 = np.array([[1, 0, -self.center[1]],
                            [0, 1, -self.center[0]],
                            [0, 0, 1]])
        trans_2 = np.array([[1, 0, self.center[1]],
                            [0, 1, self.center[0]],
                            [0, 0, 1]])
        mat = np.dot(trans_2, np.dot(origin_rotation, trans_1))
        mat = mat[0:2, :]  # drop the augmented row

        AffineTransformer.__init__(self, mat)


