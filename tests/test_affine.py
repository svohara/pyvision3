import unittest
import pyvision3 as pv3
import numpy as np


class TestAffine(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_img = pv3.Image(pv3.IMG_SLEEPYCAT)
        (w, h) = self.test_img.size

        # width, height of test image
        self.test_w = w
        self.test_h = h

        # center point of test image
        self.test_cx = w/2.0
        self.test_cy = h/2.0

    def test_affine_identity(self):
        print("\nTesting identity affine transform")
        affine_mat = np.array([[1, 0, 0],
                               [0, 1, 0]], dtype='float32')  # identity transform
        aff = pv3.AffineTransformer(affine_matrix=affine_mat)
        out = aff(self.test_img)
        self.assertTupleEqual(out.size, self.test_img.size)
        self.assertTrue(np.allclose(self.test_img.data, out.data))

    def test_affine_translate(self):
        print("\nTesting affine translation")
        # manual method using base class
        affine_mat = np.array([[1, 0, 200],
                               [0, 1, 100]], dtype='float32')  # translate +200x, +100y
        aff = pv3.AffineTransformer(affine_matrix=affine_mat, dest_size=(800, 400))
        out = aff(self.test_img)

        # using convenience class
        aff2 = pv3.AffineTranslate(200, 100, dest_size=(800, 400))
        out2 = aff2(self.test_img)

        # output of the two should be identical
        self.assertTrue(np.allclose(out.data, out2.data))

        # the 10x10 block of pixels in out located at x=200, y=100
        # should be the same as the 10x10 block of source located at 0, 0
        block_1 = out.data[100:110, 200:210, :]
        block_2 = self.test_img.data[0:10, 0:10, :]
        self.assertTrue(np.allclose(block_1, block_2))

    def test_affine_rotation(self):
        print("\nTesting affine rotation degrees")

        # use the 'fit' option for the dest_size to ensure nothing gets
        # chopped off at the corners
        aff = pv3.AffineRotation(theta_degrees=90, image_size=self.test_img.size, dest_size='fit')

        # rotate forward, 'fit' ensures buffer space for entire image
        out = aff(self.test_img)

        # rotate back, we should recover the test_img, more or less
        out2 = aff(out, invert=True)

        # this should be true when we rotate by 90, 180, 270, etc., otherwise
        # interpolation effects will cause differences between source and out2
        self.assertTrue(np.allclose(self.test_img.data, out2.data))

        # test that a ValueError is raised if we try to apply an inverse transform
        # when using the 'fit' option, but we haven't yet computed a fit (i.e., a forward pass)
        aff2 = pv3.AffineRotation(theta_degrees=30, image_size=self.test_img.size, dest_size='fit')
        self.assertRaises(ValueError, aff2, self.test_img, invert=True)

    def test_affine_invert(self):
        print("\nTesting inverting affine transformation")
        # translate the test image
        aff = pv3.AffineTranslate(200, 100)
        tmp = aff(self.test_img)

        # invert the result, to translate the image back
        out = aff(tmp, invert=True)

        # a block in the upper left area of the image should be unchanged
        # after applying the forward and inverse transformation of a translation
        block_1 = out.data[100:110, 200:210, :]
        block_2 = self.test_img.data[100:110, 200:210, :]
        self.assertTrue(np.allclose(block_1, block_2))


if __name__ == '__main__':
    unittest.main()
