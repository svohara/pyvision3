"""
Created on Oct 31, 2011
@author: bolme and sohara
Original version by Stephen O'Hara.
Modified 2014 by Stephen O'Hara to support additional capabilities,
and an addition of an interface to capture polygons.
Modified 2017 by Stephen O'Hara to forward port to Pyvision3, PEP8 compliance
"""
import pyvision3 as pv3
import shapely.geometry as sg
import cv2


class CapturePolygons(pv3.CaptureClicks):
    """
    This object handles the data management and display of the capture polygons window.
    """

    def __init__(self, im, default_polygons=None, keep_window_open=False,
                 window="PyVision Capture Polygons", pos=None):
        """
        Initialize the data.
        """
        super().__init__(im, default_points=[], keep_window_open=keep_window_open, window=window, pos=pos)

        # polygons that were input and must always show
        self.default_polygons = [] if default_polygons is None else default_polygons
        self.current_polygons = []  # completed, closed polygons

    def _close_polygon(self):
        if len(self.current_points) >= 3:
            # must have 3 points to close the polygon
            new_poly = sg.Polygon(self.current_points)
            self.current_polygons.append(new_poly)
            self.current_points = []
        return

    def _clear_last_point(self):
        if self.current_points:
            _ = self.current_points.pop()  # remove most recent element from list and discard
        return

    @staticmethod
    def _draw_instructions(canvas):
        canvas.annotate_rect((2, 2), (320, 80), color=pv3.RGB_BLUE, thickness=-1)

        text_messages = ["Click anywhere in the image to select a point.",
                         "Press 'r' to reset.",
                         "Press 'x' to delete the recent point.",
                         "Press 'c' to close the in-progress polygon.",
                         "Press the space bar when finished.",
                         "Press 'h' to toggle display of this help text."]

        for idx, txt in enumerate(text_messages):
            canvas.annotate_text(txt, (10, 10*(idx+1)), color=pv3.RGB_WHITE,
                                 font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                 font_scale=0.5)

    @staticmethod
    def _draw_polys(canvas, polygons, color=pv3.RGB_YELLOW):
        for idx, poly in enumerate(polygons):
            kx, ky = poly.exterior.coords[0]
            key_point = int(kx), int(ky)
            canvas.annotate_shape(poly, color=color, thickness=3, fill_color=None)
            canvas.annotate_point(key_point, color=color)
            canvas.annotate_text(str(idx + 1), key_point, color=color)

    def _update_image(self):
        """
        Renders the annotations on top of the current image
        """
        canvas = self.im.copy()
        if self._show_help:
            self._draw_instructions(canvas)
        self._draw_polys(canvas, self.default_polygons, color=pv3.RGB_YELLOW)
        self._draw_polys(canvas, self.current_polygons, color=pv3.RGB_RED)

        if self.current_points:
            for pt in self.current_points:
                canvas.annotate_point(pt, color=pv3.RGB_BLUE)

        if len(self.current_points) > 1:
            # draw lines connecting the in-progress points of a new polygon
            for idx in range(1, len(self.current_points)):
                pt1 = self.current_points[idx]
                pt2 = self.current_points[idx - 1]
                canvas.annotate_line(pt1, pt2, color=pv3.RGB_BLUE, thickness=2)
        self.canvas = canvas

    def display(self):
        """
        Display the window and run the main event loop.
        """
        # Setup the mouse callback to handle mouse events (optional)
        cv2.namedWindow(self.window)
        if self.pos:
            cv2.moveWindow(self.window, *self.pos)
        cv2.setMouseCallback(self.window, self.mouse_callback)

        while True:
            self._update_image()
            key_press = self.canvas.show(self.window, delay=100, annotations_opacity=0.75)
            key_press = key_press % 256

            # Handle key press events.
            if key_press == ord(' '):
                break

            if key_press == ord('h'):
                self._show_help = not self._show_help

            if key_press == ord('q'):
                self._user_quit = True
                break

            if key_press == ord('c'):
                self._close_polygon()

            if key_press == ord('x'):
                self._clear_last_point()

            if key_press == ord('r'):
                self.reset()

        if not self.keep_window_open:
            cv2.destroyWindow(self.window)

        all_polys = self.default_polygons + self.current_polygons
        return all_polys

    def reset(self):
        """
        Clear the points and start over.
        """
        self.current_points = []
        self.current_polygons = []
