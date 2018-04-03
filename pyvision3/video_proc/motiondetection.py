"""
Created: Nov 9, 2010
Author: Stephen O'Hara
Modified: Mar 11, 2016
    For compatibility with pyvision 3
"""

import pyvision3 as pv3
from pyvision3 import BG_SUBTRACT_STATIC, BG_SUBTRACT_FRAME_DIFF, \
    BG_SUBTRACT_MEDIAN, BG_SUBTRACT_APPROX_MEDIAN

import cv2
import numpy as np

try:
    import shapely.geometry as sg
except ImportError:
    print("Unable to import shapely.")
    print("MotionDetector will fail to produce polygon regions.")

MD_BOUNDING_RECTS = "BOUNDING_RECTS"
MD_STANDARDIZED_RECTS = "STANDARDIZED_RECTS"


class MotionDetector(object):
    """
    Uses background subtraction from an image buffer to detect
    areas of motion in a video.
    
    The general process is to update the image buffer and then
    call the MotionDetector's detect() method.
    """
    def __init__(self, image_buffer=None, thresh=80, method=BG_SUBTRACT_APPROX_MEDIAN, min_area=400,
                 rect_filter=None, buff_size=5, rect_type=MD_BOUNDING_RECTS, rect_sigma=2.0, **kwargs):
        """
        Parameters
        ----------
        image_buffer: a pv.ImageBuffer object to be used in the background subtraction
            step of the motion detection. If None, then this object will create an empty
            5-frame buffer, and until the buffer is full, the results of the motion detection
            will be nothing.
        thresh: Used by the background subtraction to eliminate noise.
        method: Select background subtraction method. See constants defined in
          BackgroundSubtraction module
        min_area: minimum foreground contour area required for detection
        rect_filter: a function reference that takes a list of rectangles and
          returns a list filtered in some way. This allows the user to arbitrarily
          define rules to further limit motion detection results based on the geometry
          of the bounding boxes.
        buff_size: Only used if image_buffer==None. This controls the size of the
          internal image buffer.
        kwargs: additional keyword args will be passed onto the constructor of the background
            subtraction object

        Notes
        -----
        Until the image buffer is full, the result of the motion detection will be
          nothing. See documentation on the detect(img) method of this class.
        """
        self._fgMask = None        
        self._minArea = min_area
        self._filter = rect_filter
        self._threshold = thresh
        self._softThreshold = False  # soft_thresh
        
        if image_buffer is None:
            self._image_buffer = pv3.ImageBuffer(N=buff_size)
        else:
            self._image_buffer = image_buffer
        
        self._method = method      
        self._bgSubtract = None  # can't initialize until buffer is full...so done in detect()
        self._contours = []
        self._annotateImg = None # a pyvision3 image for annotation motion detections
        self._rect_type = rect_type
        self._rect_sigma = rect_sigma

        self._kwargs = kwargs  # passed onto background subtractor initialization
        
    def _init_bg_subtract(self):
        kwargs = self._kwargs
        kwargs = {"thresh": self._threshold,
                  "soft_thresh": False}
        kwargs.update(self._kwargs)

        if self._method == BG_SUBTRACT_FRAME_DIFF:
            self._bgSubtract = pv3.FrameDifferenceModel(self._image_buffer, **kwargs)
        elif self._method == BG_SUBTRACT_STATIC:
            self._bgSubtract = pv3.StaticModel(self._image_buffer, **kwargs)
        # elif self._method == BG_SUBTRACT_MCFD:
        #    self._bgSubtract = pv.MotionCompensatedFrameDifferencer(self._image_buffer, self._threshold)
        elif self._method == BG_SUBTRACT_MEDIAN:
            self._bgSubtract = pv3.MedianModel(self._image_buffer, **kwargs)
        elif self._method == BG_SUBTRACT_APPROX_MEDIAN:
            self._bgSubtract = pv3.ApproximateMedianModel(self._image_buffer, **kwargs)
        else:
            raise ValueError("Unknown Background Subtraction Method specified.")
                  
    def _compute_contours(self):
        mask_array = self._fgMask.as_grayscale(as_type="CV")
        _, contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._contours = contours
            
    def _compute_convex_hulls(self):
        hulls = []
        for contour in self._contours:
            hull = cv2.convexHull(contour, returnPoints=True)
            hulls.append(hull)
        self._convexHulls = hulls
        
    def __call__(self, img, **kwargs):
        self.detect(img, **kwargs)
        return self.get_rects()
            
    def detect(self, img, convex_hulls=False):
        """
        You call this method to update detection results, given the new
        image in the stream. After updating detection results, use one
        of the get*() methods, such as get_rects() to see the results in the
        appropriate format.

        Parameters
        ----------
        img: pyvision3 image
            An image to be added to the buffer as the most recent image,
            and that triggers the new motion detection. Note that, depending on the
            background subtraction method, this may not be the "key frame" for the
            detection. The Frame Differencer returns a background model based on the
            middle image, but Median and Approx. Median Filters return a background
            model based on the most recent (last) image in the buffer.
        
        convex_hulls: boolean
            If true, then the detected foreground pixels are
            grouped into convex hulls, which can have the effect of removing internal
            "holes" in the detection.

        Returns
        -------
        The number of detected components in the current image. To get
        more details, use the various getX() methods, like foreground_mask(),
        after calling detect().
        
        Notes
        -----
        Until the image buffer is full, this method will make no detections.
        In which case, the return value will be -1, indicating this status. Also,
        the getKeyFrame() method should be used to retrieve the key frame from
        the buffer, which is not always the most recent image, depending on background
        subtraction method.
        """
        self._image_buffer.add(img)
        if not self._image_buffer.is_full():
            return -1
        
        # initialize background subtraction object only after buffer is full.
        if self._bgSubtract is None:
            self._init_bg_subtract()

        # update current annotation image from buffer, as appropriate for
        # the different methods
        if self._method == BG_SUBTRACT_FRAME_DIFF:
            self._annotateImg = self._image_buffer.middle()
        # elif self._method == BG_SUBTRACT_MCFD:
        #    self._annotateImg = self._image_buffer.middle()
        else:
            self._annotateImg = self._image_buffer.last()

        mask = self._bgSubtract.foreground_mask()
        cv_binary = mask.as_grayscale(as_type="CV")

        # morphology
        cv_binary = cv2.blur(cv_binary, (5, 5))
        cv_binary = cv2.dilate(cv_binary, (5, 5))
        cv_binary = cv2.erode(cv_binary, (5, 5))

        # update the foreground mask
        self._fgMask = pv3.Image(cv_binary)

        # update the detected foreground contours
        self._compute_contours()
        self._compute_convex_hulls()

        if convex_hulls:
            for hull in self._convexHulls:
                cv2.fillConvexPoly(cv_binary, hull, (255, 255, 255))

        return len(self._contours)

    def key_frame(self):
        """
        Returns
        -------
        The "key frame" of the motion detector's buffer. This is the image
        upon which detected motion rectangles, for example, should be overlaid. This
        is not always the last image in the buffer because some background subtraction
        methods (notably N-Frame Differencer) use the middle frame of the buffer.
        """
        return self._annotateImg  # computed already by the detect() method
    
    def foreground_mask(self):
        """
        Returns
        -------
        A binary pv.Image representing the foreground pixels
        as determined by the selected background subtraction method.
        @note: You must call the detect() method before foreground_mask() to
        get the updated mask.
        """
        return self._fgMask
    
    def foreground_pixels(self, bg_color=None):
        """
        Parameters
        ----------
        bg_color: (r,g,b) or None
        The background color to use. Specify as an (R,G,B) tuple.
        Specify None for a blank/black background.

        Returns
        -------
        The full color foreground pixels on either a blank (black)
        background, or on a background color specified by the user.

        Notes
        -----
        You must call detect() before foreground_pixels() to
        get updated information.
        """
        if self._fgMask is None:
            return None
        
        # binary mask selecting foreground regions
        mask = self._fgMask.as_grayscale(as_type="CV")
        
        # full color source image
        image = self._annotateImg.data
        
        # dest image, full color, but initially all zeros (black/background)
        # we will copy the foreground areas from image to here.
        dest = image.copy()
        if bg_color is None:
            dest[:] = 0
        else:
            # BGR color order
            (r, g, b) = bg_color
            dest[:] = (b, g, r)

        dest[mask.nonzero()] = image[mask.nonzero()]
        return pv3.Image(dest)

    def foreground_tiles(self, bg_color=None):
        """
        Parameters
        ----------
        bg_color: tuple (r,g,b)
            The background color to use. Specify as an (R,G,B) tuple.
            Specify None for a blank/black background.

        Returns
        -------
        A list of "tiles", where each tile is a small pyvision3 Image
        representing the clipped area of the annotationImg based on
        the motion detection. Only the foreground pixels are copied, so
        the result are tiles with full-color foreground pixels on the
        specified background color (black by default).

        Notes
        -----
        You must call detect() prior to foreground_tiles() to get
        updated information.
        """
        fg_pix = self.foreground_pixels(bg_color=bg_color)
        rects = self.get_rects()

        tiles = []
        for r in rects:
            # for every rectangle, crop from fg_pix image
            t = fg_pix.crop(r)
            tiles.append(t)

        return tiles

    def get_rects(self):
        """
        Returns
        -------
        The bounding boxes of the external contours of the foreground mask. The
        boxes will either be the bounding rectangles of the contours, or a box fitted to
        the contours based on the center of mass and n-sigma deviations in x and y. This
        preference is selected when initializing the MotionDetector object.
        
        Notes
        -----
        You must call detect() before get_rects() to see updated results.
        """
        if self._rect_type == MD_BOUNDING_RECTS:
            return self.bounding_rects()
        elif self._rect_type == MD_STANDARDIZED_RECTS:
            return self.standardized_rects()
        else:
            raise ValueError("Unknown rect type: "+self._rect_type)

    def bounding_rects(self):
        """
        Returns
        -------
        the bounding boxes of the external contours of the foreground mask.

        Notes
        -----
        You must call detect() before bounding_rects() to see updated results.
        """
        # create a list of the top-level contours found in the contours structure
        rects = [pv3.Rect(*cv2.boundingRect(c)) for c in self._contours
                 if cv2.contourArea(c) > self._minArea]

        if self._filter is not None:
            rects = self._filter(rects)
        
        return rects
    
    def standardized_rects(self):
        """
        Returns
        -------
        the boxes centered on the target center of mass +- n_sigma*std

        Notes
        -----
        You must call detect() before standardized_rects() to see updated results.
        """
        rects = []
        for contour in self._contours:
            if cv2.contourArea(contour) > self._minArea:
                (x, y, w, h) = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                m00 = moments["m00"]
                m01 = moments["m01"]
                m10 = moments["m10"]
                mu02 = moments["mu02"]
                mu20 = moments["mu20"]
                cx = m10/m00
                cy = m01/m00
                w = 2.0*self._rect_sigma*np.sqrt(mu20/m00)
                h = 2.0*self._rect_sigma*np.sqrt(mu02/m00)
                r = pv3.CenteredRect(cx, cy, w, h)
                rects.append(r)
        
        if self._filter is not None:
            rects = self._filter(rects)
        
        return rects
    
    def polygons(self, return_all=False):
        """
        Parameters
        ----------
        return_all: boolean
            return all contours regardless of min area.

        Returns
        -------
        The polygon contours of the foreground mask. The polygons are
        compatible with pv.Image annotate_shape() method, and are instances
        of shapely polygons

        Notes
        -----
        You must call detect() before polygons() to see updated results.
        """
        # create a list of the top-level contours found in the contours (cv.Seq) structure
        polys = []

        for contour in self._contours:
            if return_all or cv2.contourArea(contour) > self._minArea:
                coords = [pt for pt in contour.squeeze()]
                poly = sg.Polygon(coords)
                polys.append(poly)

        return polys
    
    def convex_hulls(self):
        """
        Returns
        -------
        The convex hulls of the contours of the foreground mask, as shapely
        polygon objects.

        Notes
        -----
        You must call detect() before convex_hulls() to see updated results.
        """
        hull_polys = [sg.Polygon(hull.squeeze()) for hull in self._convexHulls
                      if len(hull.squeeze()) > 3]
        return hull_polys
        
    def annotate_frame(self, key_frame=None,
                       rect_color=pv3.RGB_RED,
                       contour_color=pv3.RGB_BLACK,
                       convex_hull_color=pv3.RGB_CYAN):
        """
        Draws detection results on an image (key_frame) specified by the user. Specify
        None as the color for any aspect you wish not drawn.

        Parameters
        ----------
        key_frame: pyvision3 image or None
            if None, then a copy of the internal annotation image will be used,
            otherwise the annotations will be drawn on top of the provided pv image
        rect_color: tuple (r,g,b) for detection rectangles, or None
        contour_color: tuple (r,g,b) for detection contours, or None
        convex_hull_color: tuple (r,g,b) for convex hulls, or None

        Returns
        -------
        Returns the key_frame image with appropriate annotations. Annotations
        will be rendered directly onto key_frame, so the return value should
        "point" to the same object as the input key_frame or the internal
        annotation image, as appropriate.

        Notes
        -----
        1. You must call detect() prior to annotate_frame() to see updated results.
        2. Optical flow is only shown if method was MCFD. Optical flow not yet implemented
        """
        if key_frame is None and self._annotateImg is None:
            return None

        key_frame = self._annotateImg if key_frame is None else key_frame

        if contour_color is not None:
            for poly in self.polygons():
                key_frame.annotate_shape(poly, color=contour_color, thickness=1)

        if rect_color is not None:
            for rect in self.get_rects():
                key_frame.annotate_shape(rect, color=rect_color, thickness=2)

        if convex_hull_color is not None:
            for poly in self.convex_hulls():
                key_frame.annotate_shape(poly, color=convex_hull_color, thickness=1)

        # TODO: Update this after porting OpticFlow and MCFD code to pyvision3 3
        #if (flow_color is not None) and (self._method == pv.BG_SUBTRACT_MCFD):
        #    flow = self._bgSubtract.getOpticalFlow()
        #    flow.annotate_frame(key_frame, type="TRACKING", color=flow_color)

        return key_frame