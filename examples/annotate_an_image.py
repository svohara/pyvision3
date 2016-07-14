"""
Created on Dec 20, 2015

@author: sohara
"""
import shapely.geometry as sg
import pyvision as pv3


def annotation_demo():
    """
    Demonstrates several annotation methods of the Image class
    """
    img = pv3.Image(pv3.IMG_DRIVEWAY, desc="Annotations Demo")  # load a sample image

    # annotate a point as a small filled circle
    img.annotate_point((250, 250), color=(0, 0, 0))
    img.annotate_point((300, 300))  # default color is red

    # annotate text next to the black point, blue on white background
    img.annotate_text("Waldo was here", (254, 246), color=(0, 0, 255), bg_color=(255, 255, 255))

    # complex polygon with an exterior and an interior 'hole'
    exterior = ((50, 50), (25, 125), (50, 200), (200, 200), (200, 50), (50, 50))
    interior = ((100, 100), (100, 150), (150, 150), (175, 125), (150, 100), (100, 100))
    poly = sg.Polygon(exterior, [interior])
    img.annotate_shape(poly, color=(255, 255, 0), fill_color=(0, 0, 255), thickness=2)

    # two concentric circles
    img.annotate_circle((400, 400), 50, color=(0, 255, 0), thickness=2)
    img.annotate_circle((400, 400), 25, color=(0, 0, 255), thickness=-1)  # filled circle

    # a LineString
    lines = sg.LineString(((10, 25), (382, 122), (251, 470)))
    img.annotate_shape(lines)

    # a Rectangle
    img.annotate_rect((538, 212), (600, 300), color=(255, 0, 255), thickness=3)  # -1 for filled

    # show the image with annotations
    # NOTE: to NOT show the annotations, set annotations=False in the following
    img.show(highgui=True, delay=0, annotations_opacity=0.75)


if __name__ == '__main__':
    print("====================================================")
    print("Set focus to display window and hit any key to exit.")
    print("====================================================")
    annotation_demo()
