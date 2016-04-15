from unittest import TestCase
import pyvision as pv3
import numpy as np


class TestImage(TestCase):
    def test_resize(self):
        print("\nTest Image 'resize' Method")
        img = pv3.Image(pv3.IMG_PRIUS)
        img_thumb = img.resize((64, 64), keep_aspect=True)

        # letter box should exist on top and bottom
        self.assertTrue(np.all(img_thumb[0, :] == 0))

        img_thumb = img.resize((64, 64))  # not keeping aspect ratio
        self.assertFalse(np.all(img_thumb[0, :] == 0))

    def test_crop(self):
        print("\nTest Image 'crop' Method")
        img = pv3.Image(pv3.IMG_DRIVEWAY)
        rect = pv3.Rect(20, 50, 100, 100)
        tile = img.crop(rect)
        self.assertTupleEqual(tile.data.shape, (100, 100, 3))

        # Check that OutOfBoundsError is raised when crop is invalid
        bad_rect = pv3.Rect(-40, 300, 80, 80)
        self.assertRaises(pv3.OutOfBoundsError, img.crop, bad_rect)
