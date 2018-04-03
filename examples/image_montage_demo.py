import pyvision3 as pv3


def main():
    """
    Demonstrates several annotation methods of the Image class
    """
    img1 = pv3.Image(pv3.IMG_DRIVEWAY)
    img2 = pv3.Image(pv3.IMG_PRIUS)
    img3 = pv3.Image(pv3.IMG_SLEEPYCAT)
    img_list = [img1, img2, img3] * 4
    imontage = pv3.ImageMontage(img_list, layout=(2, 3), tile_size=(128, 128),
                                labels="index", keep_aspect=True, highlight_selected=True)
    imontage.show()  #event loop, blocks until user quits montage
    sel = imontage.get_highlighted()
    sel_str = ",".join([str(x) for x in sel])
    print("You selected these images: {}".format(sel_str))

if __name__ == '__main__':
    print("====================================================")
    print("Click on some of the image tiles in the montage")
    print("With montage window in focus, hit spacebar to quit")
    print("====================================================")
    main()