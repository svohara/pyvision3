"""
Tests for the dataset tools package
author: Stephen O'Hara
created: April 14, 2016
"""

import unittest
import pyvision3 as pv3
import shapely.geometry as sg


class TestDatasetTools(unittest.TestCase):
    def test_crop_regions(self):
        print("\nTest 'crop regions' function")
        img = pv3.Image(pv3.IMG_SLEEPYCAT)

        p1 = sg.Polygon([(200, 200), (200, 400), (380, 380), (395, 210)])
        p2 = sg.Polygon([(400, 400), (350, 500), (400, 600), (500, 600), (575, 425)])

        shapes_list = [p1, p2]
        for shp in shapes_list:
            img.annotate_shape(shp, fill_color=pv3.RGB_YELLOW)

        # Extract crops of different sizes based on polygon bounds
        crops = pv3.crop_regions(img, shapes_list)
        self.assertTrue(len(crops) == 2)
        self.assertTupleEqual(crops[0].size, (196, 201))

        # Extract crops of a fixed size centered on the polygon centroids
        crops2 = pv3.crop_regions(img, shapes_list, crop_size=(300, 300))
        self.assertTrue(len(crops2) == 2)
        self.assertTupleEqual(crops2[0].size, (300, 300))


if __name__ == '__main__':
    unittest.main()
