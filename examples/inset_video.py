"""
Demonstration of showing an video as an
inset into a static image
"""
import pyvision3 as pv3


def on_frame(frame, num, background_img=None, **kwargs):
    """
    callback function, called by vid.play(...) on
    each new frame of the video
    """
    assert(background_img is not None)
    img_w, img_h = background_img.size
    inset_width = 225
    inset_height = 150
    inset_x = img_w - inset_width - 10
    inset_y = img_h - inset_height - 10
    background_img.annotate_inset_image(frame, pos=(inset_x, inset_y), size=(inset_width, inset_height))
    background_img.annotate_text("{}".format(str(num).zfill(5)), (10, 10),
                                 color=pv3.RGB_WHITE, bg_color=pv3.RGB_BLACK)
    background_img.show(window_title="Picture-in-picture", delay=5, annotations_opacity=1.0)
    background_img.annotate_rect((inset_x-2, inset_y-2),
                                 (inset_x+inset_width+2, inset_y+inset_height+2),
                                 color=pv3.RGB_BLACK, thickness=-1)


def demo():
    vid = pv3.Video(pv3.VID_PRIUS)
    img = pv3.Image(pv3.IMG_SLEEPYCAT).resize((640, 480))

    vid.play(
        window_title=None,  # suppress main video output
        delay=1,
        on_new_frame=on_frame,  # this will take care of output
        background_img=img,  # kwarg passed to on_frame
        start_frame=250)


if __name__ == "__main__":
    print("=================================================================")
    print("With focus on video window, use the keyboard to control playback.")
    print("During playback, with focus on window, hold SPACEBAR to pause.")
    print("=================================================================")
    demo()
