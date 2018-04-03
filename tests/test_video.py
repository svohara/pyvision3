import unittest
import pyvision3 as pv3
import numpy as np


class TestVideo(unittest.TestCase):
    def test_video_resize(self):
        print("\nTest Video 'size' Parameter")
        vid = pv3.Video(pv3.VID_PRIUS, size=(200, 250))
        for _ in range(10):
            img = vid.next()
        self.assertTupleEqual((200, 250), img.size)

    def test_video_seek(self):
        print("\nTest Video 'seek_to' Method")
        vid = pv3.Video(pv3.VID_PRIUS, size=(320,240))

        # seek forward to frame index 200 (201st image)
        # then call next() 10 times
        img = vid.seek_to(200)
        for _ in range(10):
            img = vid.next()

        # we should end up at frame number 210
        self.assertTrue(vid.current_frame_num == 210)

        # test when desired frame is earlier than current
        vid.seek_to(12)
        self.assertTrue(vid.current_frame_num == 12)

    def test_video_from_image_stack(self):
        print("\nTest VideoFromImageStack using an Image Buffer")
        vid = pv3.Video(pv3.VID_PRIUS, size=(320, 240))

        # fill an image buffer using video as input, starting with frame 401
        vid.seek_to(400)
        ib = pv3.ImageBuffer(N=120)
        ib.fill(vid)

        # convert image buffer to a stack of grayscale images
        X = ib.as_image_stack_BW()  # X is a 3D ndarray

        # instantiate a video object from the image stack
        # and do a few simple tests
        vid2 = pv3.VideoFromImageStack(X)
        imgA = vid2.seek_to(30)
        self.assertTupleEqual(imgA.size, (320, 240))
        self.assertTrue(np.all(imgA.data == X[30, :, :]))

if __name__ == '__main__':
    unittest.main()
