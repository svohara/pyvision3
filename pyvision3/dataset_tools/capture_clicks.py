"""
Created on Oct 31, 2011
@author: bolme and sohara
Original version by David Bolme.
Modified 2014 by Stephen O'Hara to support additional capabilities,
and an addition of an interface to capture polygons.
Modified 2017 by Stephen O'Hara to forward port to Pyvision3, PEP8 compliance
"""
import pyvision3 as pv3
import cv2


def null_callback(*args, **kwargs):
    pass


class CaptureClicks:
    """
    This object handles the data management and display of the capture clicks window.
    """

    def __init__(self, im, default_points=None, keep_window_open=False,
                 window="PyVision Capture Points", pos=None):
        """
        Initialize the data.
        """
        self.window = window
        self.im = im
        self.keep_window_open = keep_window_open
        self._user_quit = False
        self.pos = pos  # position of window
        self.current_points = []
        self.default_points = [] if default_points is None else default_points
        self._show_help = True

    def _clear_last_point(self):
        if self.current_points:
            _ = self.current_points.pop()  # remove most recent element from list and discard
        return

    @staticmethod
    def _draw_instructions(canvas):
        canvas.annotate_rect((2, 2), (320, 70), color=pv3.RGB_BLUE, thickness=-1)

        text_messages = ["Click anywhere in the image to select a point.",
                         "Press 'r' to reset.",
                         "Press 'x' to delete the recent point.",
                         "Press the space bar when finished.",
                         "Press 'h' to toggle display of this help text."]

        for idx, txt in enumerate(text_messages):
            canvas.annotate_text(txt, (10, 10*(idx+1)), color=pv3.RGB_WHITE,
                                 font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                 font_scale=0.5)

    @staticmethod
    def _draw_points(canvas, points, color=pv3.RGB_YELLOW):
        for idx, pt in enumerate(points):
            canvas.annotate_point(pt, color=color)
            canvas.annotate_text(str(idx+1), pt, color=color)

    def _update_image(self):
        """
        Renders the annotations on top of the current image
        """
        canvas = self.im.copy()
        if self._show_help:
            self._draw_instructions(canvas)

        if self.default_points:
            self._draw_points(canvas, self.default_points, color=pv3.RGB_YELLOW)

        if self.current_points:
            self._draw_points(canvas, self.current_points, color=pv3.RGB_RED)

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

            if key_press == ord('x'):
                self._clear_last_point()

            if key_press == ord('r'):
                self.reset()

        if not self.keep_window_open:
            cv2.destroyWindow(self.window)

        all_points = self.default_points + self.current_points
        return all_points

    def reset(self):
        """
        Clear the points and start over.
        """
        self.current_points = []

    def mouse_callback(self, event, x, y, flags, param):
        """
        Call back function for mouse events.
        """
        if event in [cv2.EVENT_LBUTTONDOWN]:
            point = (x, y)
            self.current_points.append(point)

