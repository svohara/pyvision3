import unittest
import pyvision3 as pv3


class TestImageBuffer(unittest.TestCase):
    def test_buffer_fill(self):
        print("\nTesting Image Buffer 'fill' Method")

        # Fill buffer from a video object
        vid = pv3.Video(pv3.VID_PRIUS, size=(320, 240))
        ib = pv3.ImageBuffer(N=75)
        ib.fill(vid)
        img = ib.middle()  # middle image of the buffer
        self.assertTupleEqual(img.size, (320, 240))

        # Fill buffer from a list of pv images
        img_files = [pv3.IMG_SLEEPYCAT, pv3.IMG_DRIVEWAY, pv3.IMG_SLEEPYCAT, pv3.IMG_DRIVEWAY]
        img_list = [pv3.Image(x) for x in img_files]  # list of 4 images

        ib = pv3.ImageBuffer(N=10)
        ib.fill(img_list)  # should repeat these four images to fill buffer of 10
        img1 = ib.last()
        img2 = img_list[1]
        self.assertTrue(img1 == img2)  # should 'point' to same object

    def test_buffer_as_montage(self):
        print("\nTesting Image Buffer 'as_montage' Method")
        # Fill buffer from a video object
        vid = pv3.Video(pv3.VID_PRIUS, size=(320, 240))
        ib = pv3.ImageBuffer(N=75)
        ib.fill(vid)
        im = ib.as_montage(layout=(4, 6), tile_size=(80, 60))
        self.assertIsInstance(im, pv3.ImageMontage)
        # im_img = im.as_image()
        # im_img.save("test.jpg")

if __name__ == '__main__':
    unittest.main()
