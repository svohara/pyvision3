"""
Demonstrates using some of the background subtraction methods
"""
import pyvision3 as pv3


def demo():
    vid = pv3.Video(pv3.VID_PRIUS, size=(320, 240))
    vid.seek_to(400)  # fast forward to a place that's interesting

    # Frame Difference model keys on the middle image in the buffer,
    # and looks at differences between the middle and first, middle and last,
    # and generates the absolute value of the minimum difference
    ib = pv3.ImageBuffer(N=60)  # frame differencing on short-temporal diff
    ib.fill(vid)
    bg = pv3.FrameDifferenceModel(ib, thresh=80, soft_thresh=False)
    fd_mask = bg.foreground_mask()
    fd_key = ib.middle()

    # Median Model
    ib = pv3.ImageBuffer(N=400)  # median stats are better with many samples
    vid.reset()
    ib.fill(vid)
    bg = pv3.MedianModel(ib, thresh=80, soft_thresh=False)
    md_mask = bg.foreground_mask()
    md_key = ib.last()

    # Approximate Median
    # Use same buffer as created for Median model
    # NOTE: This will be exact same as Median Model, the difference is that
    # if we were to continue the video, updating the model at each new frame,
    # the computation would be a fast update (approx) to the median...but with
    # a static set of images in a buffer, the exact median will be computed.
    bg = pv3.ApproximateMedianModel(ib, thresh=80, soft_thresh=False)
    am_mask = bg.foreground_mask()
    am_key = ib.last()

    # display result side-by-side with key image to which it applies
    img_list = [fd_key, fd_mask, md_key, md_mask, am_key, am_mask]
    img_lbls = ["FD: Img", "FD: Mask", "MD: Img", "MD: Mask", "AM: Img", "AM: Mask"]
    imontage = pv3.ImageMontage(
        img_list, layout=(3, 2), tile_size=(320, 240), labels=img_lbls
    )
    imontage.show(window_title="Background Subtraction", delay=0)
    return bg


if __name__ == "__main__":
    print("=================================================================")
    print("Focus on montage image, and hit any key to exit.")
    print("=================================================================")
    demo()
