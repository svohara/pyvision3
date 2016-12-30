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
    img.annotate_point((250, 250), color=pv3.RGB_BLACK)
    img.annotate_point((300, 300))  # default color is red

    # annotate text next to the brown point, blue on white background
    img.annotate_text("Waldo was here", (254, 246),
                      color=pv3.RGB_BLUE,
                      bg_color=pv3.RGB_WHITE)

    # complex polygon with an exterior and an interior 'hole'
    exterior = ((50, 50), (25, 125), (50, 200), (200, 200), (200, 50), (50, 50))
    interior = ((100, 100), (100, 150), (150, 150), (175, 125), (150, 100), (100, 100))
    poly = sg.Polygon(exterior, [interior])
    img.annotate_shape(poly, color=pv3.RGB_CYAN, fill_color=pv3.RGB_GREEN, thickness=2)

    # two concentric circles
    img.annotate_circle((400, 400), 75, color=pv3.RGB_PURPLE, thickness=5)
    img.annotate_circle((400, 400), 60, color=pv3.RGB_PURPLE, thickness=-1)  # filled circle

    # a LineString
    lines = sg.LineString(((10, 25), (382, 122), (251, 470)))
    img.annotate_shape(lines, thickness=4, color=pv3.RGB_RED)

    # a Rectangle
    img.annotate_rect((538, 212), (600, 300), color=pv3.RGB_ORANGE, thickness=3)  # -1 for filled

    # show the image with annotations
    # NOTE: to NOT show the annotations, set annotations=False in the following
    img.show_annotation(window_title="Mask Layer", highgui=True, delay=1)
    img.show(window_title="Image Layer", highgui=True, delay=0, annotations_opacity=0.5)

if __name__ == '__main__':
    print("====================================================")
    print("Set focus to display window and hit any key to exit.")
    print("====================================================")
    annotation_demo()
