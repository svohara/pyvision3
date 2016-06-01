import unittest
import pyvision as pv3
import numpy as np
import math


class TestAffine(unittest.TestCase):
    def setUp(self):
        self.test_img = pv3.Image(pv3.IMG_SLEEPYCAT)

    def test_affine_identity(self):
        print("\nTesting identity affine transform")
        affine_mat = np.array([[1, 0, 0],
                               [0, 1, 0]], dtype='float32')  # identity transform
        aff = pv3.AffineTransformer(affine_matrix=affine_mat)
        out = aff(self.test_img)
        self.assertTupleEqual(out.size, self.test_img.size)
        self.assertTrue(np.allclose(self.test_img.data, out.data))
        out.save("aff_test_identity.jpg", as_annotated=False)

    def test_affine_translate(self):
        print("\nTesting affine translation")
        affine_mat = np.array([[1, 0, 200],
                               [0, 1, 100]], dtype='float32')  # translate +200x, +100y
        aff = pv3.AffineTransformer(affine_matrix=affine_mat)
        out = aff(self.test_img, dest_size=(800, 400))
        # the 10x10 block of pixels in out located at x=200, y=100
        # should be the same as the 10x10 block of source located at 0, 0
        block_1 = out.data[100:110, 200:210]
        block_2 = self.test_img.data[0:10, 0:10]
        self.assertTrue(np.allclose(block_1, block_2))
        # out.save("aff_test_translate.jpg", as_annotated=False)

    def test_affine_rotation(self):
        print("\nTesting affine rotation by 90 degrees")
        theta = math.pi/2
        (img_w, img_h) = self.test_img.size
        affine_mat = np.array([[math.cos(theta), -math.sin(theta), img_h],
                               [math.sin(theta), math.cos(theta), 0]], dtype='float32')

        aff = pv3.AffineTransformer(affine_matrix=affine_mat)
        out = aff(self.test_img, dest_size=(img_h, img_w))
        # the output data should be the transpose of the input
        # self.assertTrue(np.allclose(out.data.T, self.test_img.data))
        # out.save("aff_test_rotate.jpg", as_annotated=False)

if __name__ == '__main__':
    unittest.main()
