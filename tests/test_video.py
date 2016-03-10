import unittest
import pyvision as pv3


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

if __name__ == '__main__':
    unittest.main()
