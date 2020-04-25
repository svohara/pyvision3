import pyvision3 as pv3


def main():
    """
    Demonstrates several annotation methods of the Image class
    """
    img1 = pv3.Image(pv3.IMG_DRIVEWAY)
    img1.annotate_text("sample annotation", (200, 200), color=pv3.RGB_CYAN, bg_color=pv3.RGB_BLACK, font_scale=2)
    img2 = pv3.Image(pv3.IMG_PRIUS)
    img3 = pv3.Image(pv3.IMG_SLEEPYCAT)
    img_list = [img1, img2, img3] * 4
    imontage = pv3.ImageMontage(img_list,
                                layout=(1, 3),
                                tile_size=(480, 360),
                                labels="index",
                                keep_aspect=True,
                                highlight_selected=True,
                                alpha=0.6)
    imontage.show()  # event loop, blocks until user quits montage
    sel = imontage.get_highlighted()
    sel_str = ",".join([str(x) for x in sel])
    print("You selected these images: {}".format(sel_str))


if __name__ == '__main__':
    print("====================================================")
    print("Click on some of the image tiles in the montage")
    print("With montage window in focus, hit spacebar to quit")
    print("====================================================")
    main()
