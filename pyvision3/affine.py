"""
This module provides convenient wrappers to apply or construct
affine transformations from pyvision images.
"""

import cv2
import pyvision3 as pv3
import math
import numpy as np


class AffineTransformer(object):
    """
    This class defines a callable object that can be applied to
    pyvision3 images to apply an affine or inverse affine transformation.
    """
    def __init__(self, affine_matrix, dest_size=None):
        """
        Constructor

        Parameters
        ----------
        affine_matrix: numpy array (2x3)
        dest_size: tuple (w, h) or 'fit'
            If None, then destination size will be set as source_img size
            when applying the transformation.

            If 'fit', then the destination will be large enough to fit all
            the original pixels. Also, if 'fit' then the affine transform
            will be modified to add enough translation so that all pixels are
            located with positive coordinates.

            Otherwise, dest_size should be a tuple specifying the output
            width and height. If your output is all black, then you've likely
            transformed the source image out of bounds of the destination size.
        """
        self.affine_matrix = affine_matrix
        self.dest_size = dest_size
        self._fit_points = None

    def _get_fit(self, source_size, invert=False):
        # figure out max size of destination image by taking transformations
        # of the four corners
        maxx = source_size[0] - 1
        maxy = source_size[1] - 1
        points = np.array([[0, 0, 1], [0, maxy, 1], [maxx, maxy, 1], [maxx, 0, 1]]).T
        aug_mat = self.get_augmented_matrix()

        if invert:
            msg = "You can't perform an inverse transform with the 'fit' option "
            msg += "until you have performed a forward transform to compute the fit."
            if self._fit_points is None:
                raise ValueError(msg)
            aug_mat = np.linalg.inv(aug_mat)
            new_points = np.dot(aug_mat, self._fit_points)
        else:
            new_points = np.dot(aug_mat, points)
            self._fit_points = new_points

        new_maxx = new_points[0, :].max()
        new_maxy = new_points[1, :].max()
        offset_x = new_points[0, :].min()
        offset_y = new_points[1, :].min()

        # new image size to fit
        fit_width = int(new_maxx - offset_x) + 1
        fit_height = int(new_maxy - offset_y) + 1

        # update the transformation so no negative pixel locations
        # only do this for forward transformations.
        if not invert:
            self.affine_matrix[0, 2] -= offset_x
            self.affine_matrix[1, 2] -= offset_y

        return fit_width, fit_height

    def __call__(self, source_img, invert=False):
        """

        Parameters
        ----------
        source_img: pyvision3 image

        invert: boolean
            If true, the inverse transformation is applied to the source_img
            instead of the forward transformation.

        Returns
        -------
        A pyvision3 image resulting from applying the transformation
        to the source_img.
        """
        input_array = source_img.data
        if self.dest_size is None:
            dest_size = source_img.size
        elif self.dest_size == 'fit':
            dest_size = self._get_fit(source_img.size, invert=invert)
        else:
            dest_size = self.dest_size

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
    def __init__(self, dx, dy, **kwargs):
        """
        Parameters
        ----------
        dx: int or float
        dy: int or float
        """
        translate_mat = np.array([[1, 0, dx],
                                  [0, 1, dy]], dtype='float32')
        AffineTransformer.__init__(self, translate_mat, **kwargs)


class AffineRotation(AffineTransformer):
    """
    Convenience class to construct an affine transformation that
    rotates an image clockwise about the center of the image by
    theta degrees.

    Note, for 90, 180, 270 rotations, it would be much faster
    to apply matrix flipping and transposing.
    """
    def __init__(self, theta_degrees, image_size, **kwargs):
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

        AffineTransformer.__init__(self, mat, **kwargs)


# TODO AffineFromPoints, AffineFromRect
