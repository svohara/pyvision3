"""
Demonstrates using motion detection on a video
"""
import pyvision as pv3
import numpy as np


def each_frame(cur_frame, frame_num, key=None, image_buffer=None, motion=None):
    motion.detect(cur_frame)
    out = motion.annotate_frame(convex_hull_color=None)
    fg = motion.foreground_pixels()
    if (out is not None) and (fg is not None):
        montage = pv3.ImageMontage([out, fg], layout=(1, 2), tile_size=(320, 240),
                                   labels=['Annotation', 'Foreground Pix'])
        out_img = montage.as_image()
        out_img.show(window_title="Motion Detection", delay=1)


def md_factory_static(vid):
    ib = pv3.ImageBuffer(30)
    ib.fill(vid)
    bg_stack = ib.as_image_stack_BW()
    bg_array = np.median(bg_stack, axis=0)
    bg_image = pv3.Image(bg_array)
    md = pv3.MotionDetector(method=pv3.BG_SUBTRACT_STATIC, bg_image=bg_image, thresh=80)
    return md


def md_factory_approx_median():
    md = pv3.MotionDetector(method=pv3.BG_SUBTRACT_APPROX_MEDIAN, buff_size=60, thresh=80)
    return md


def md_factory_median():
    md = pv3.MotionDetector(method=pv3.BG_SUBTRACT_MEDIAN, buff_size=30, thresh=80)
    return md


def md_factory_n_frame_diff():
    md = pv3.MotionDetector(method=pv3.BG_SUBTRACT_FRAME_DIFF, buff_size=5, thresh=80)
    return md


def demo():
    vid = pv3.Video(pv3.VID_PRIUS, size=(320, 240))

    # Try one of the following...
    md = md_factory_static(vid)
    # md = md_factory_approx_median()
    # md = md_factory_n_frame_diff()
    # md = md_factory_median()

    vid.play(window=None, annotate=False,
             start_frame=100, end_frame=1000,
             on_new_frame=each_frame, motion=md)


if __name__ == '__main__':
    demo()
