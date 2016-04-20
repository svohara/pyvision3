"""
Demonstration of the crop_regions function.
author: Stephen O'Hara
created: April 14, 2016
"""
import pyvision as pv3
import shapely.geometry as sg


def demo():
    img = pv3.Image(pv3.IMG_SLEEPYCAT)
    p_cat = pv3.POLY_SLEEPYCAT
    img.annotate_shape(p_cat, color=pv3.RGB_BLACK, fill_color=pv3.RGB_WHITE)

    # Two polygons that intersect part of sleepy cat
    p1 = sg.Polygon([(575, 200), (575, 400), (755, 380), (770, 210)])
    p2 = sg.Polygon([(400, 400), (350, 500), (400, 600), (500, 600), (575, 425)])

    shapes_list = [p1, p2]
    for shp in shapes_list:
        img.annotate_shape(shp, color=pv3.RGB_BLACK, fill_color=pv3.RGB_WHITE)
        r1 = sg.box(*shp.bounds)
        img.annotate_shape(r1, color=pv3.RGB_CYAN, thickness=5)
        pt = (shp.centroid.x, shp.centroid.y)
        r2 = pv3.CenteredRect(pt[0], pt[1], 300, 300)
        img.annotate_shape(r2, color=pv3.RGB_YELLOW, thickness=5)

    # Extract crops of different sizes based on polygon bounds
    crops = pv3.crop_regions(img, shapes_list)
    montage1 = pv3.ImageMontage(crops, layout=(1, 2), tile_size=(150, 150))
    mnt_image = montage1.as_image()

    # Extract crops of a fixed size centered on the polygon centroids
    crops2 = pv3.crop_regions(img, shapes_list, crop_size=(300, 300))
    montage2 = pv3.ImageMontage(crops2, layout=(1, 2), tile_size=(150, 150))
    mnt_image2 = montage2.as_image()

    # Extract background crops that do not intersect sleepy cat
    neg_crops = pv3.crop_negative_regions(img, [p_cat], (150, 150), N=10)
    montage3 = pv3.ImageMontage(neg_crops, layout=(2, 5), tile_size=(150, 150))
    mnt_image_negs = montage3.as_image()

    img.imshow(window_title="Source image and regions")
    mnt_image.show(window_title="Crops from bounding boxes", delay=1, pos=(10, 10))
    mnt_image2.show(window_title="Crops from centroids and fixed size", delay=1, pos=(10, 200))
    mnt_image_negs.show(window_title="Random background crops", delay=0, pos=(10, 400))

if __name__ == '__main__':
    print("=================================================================")
    print("Demonstrating extracting crops from a source image")
    print("Focus on either montage image, and hit any key to exit.")
    print("=================================================================")
    demo()
