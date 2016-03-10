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

