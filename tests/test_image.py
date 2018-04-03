import unittest
import pyvision3 as pv3
import numpy as np
import cv2


class TestImage(unittest.TestCase):
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

    def test_mask(self):
        print("\nTest Image 'annotate_mask' Method")
        img = pv3.Image(pv3.IMG_DRIVEWAY)
        mask = cv2.imread(pv3.IMG_MASK)
        mask_target = cv2.imread(pv3.IMG_MASK_RESULT)   # known good result
        img.annotate_mask(mask)
        masked_image = img.as_annotated()  # what we get
        self.assertTrue(np.allclose(masked_image.data, mask_target))

    def test_inset(self):
        print("\nTest Image 'annotate_inset_image' Method")
        img = pv3.Image(pv3.IMG_DRIVEWAY)

        # generate inset image
        dat = np.ones((50, 50, 3), dtype='uint8')  # (1,1,1) is the transparent color by default
        cv2.rectangle(dat, (5, 5), (20, 20), (255, 0, 0), thickness=-1)
        cv2.rectangle(dat, (30, 30), (45, 45), (0, 255, 255), thickness=-1)
        inset = pv3.Image(dat)

        img.annotate_inset_image(inset, (250, 250), size=None)
        self.assertTupleEqual(tuple(img.annotation_data[256, 256, :]), (255, 0, 0))
        self.assertTupleEqual(tuple(img.annotation_data[281, 281, :]), (0, 255, 255))

if __name__ == '__main__':
    unittest.main()