"""
Created on Dec 18, 2015
@author: Stephen O'Hara

This module defines the pyvision image class, which wraps
the image data (array) in a class with convenient methods.

One very useful aspect of the image class is that annotations
(such as bounding boxes that might be drawn around detection regions)
are kept on a separate layer, as opposed to drawing directly
onto the image array itself.
"""
# The following prevents a bunch of pylint no-member errors
# with the cv2 module.
# pylint: disable=E1101

import cv2
import numpy as np
import io
try:
    import matplotlib.pyplot as plot
except ImportError:
    print("Error importing matplotlib.")
    print("Matplotlib integration will not work")

try:
    import shapely.geometry as sg
except ImportError:
    print("Error importing shapely.")
    print("Shapely is required for annotating shapes (polygons) on images.")
    print("Shapely is also used to determine if a crop is in bounds, etc.")

from .pv_exceptions import OutOfBoundsError, ImageAnnotationError
from .geometry import in_bounds, integer_bounds, Rect


class Image(object):
    """
    A pyvision3 Image object contains the image data, an
    annotations layer, and many convenient methods.
    Supports 1 channel and 3 channel images.
    """

    def __init__(self, source, *args, desc="Pyvision Image", **kwargs):
        """
        The constructor wraps a cv2.imread(...) function,
        passing in the args and kwargs appropriately, and
        then allocating data for the annotations layer.

        Parameters
        ----------
        desc: string
            Provide a short description of this image, that will be used
            by default in window titles and other functions
        source: string, file object, or cv2 image array
            If string, this is the full path to the image file to load.
            If file object, this is an open file handle from which to load
            the image.
            If ndarray, then we assume this is a cv2 image array which we will
            just wrap.
        args: variable
            Other args will be passed through to cv2.imread, the first arg
            should be the image source, like a file name. See the cv2 docs
            on imread for more details.
        kwargs: variable
            Keyword args will be passed through to cv2.imread

        Examples
        --------
        #loading from files
        img = pv3.Image('mypic.jpg')
        img2 = pv3.Image('mypic.jpg', cv2.IMREAD_GRAYSCALE)
        
        #loading from a file handle...this example is silly, but
        # this capability is good for example when loading from a file streamed
        # over a network connection, or S3, etc.
        with open('somepath/somefile.png', 'rb') as infile:
            img3 = pv3.Image(infile)
        
        #Wrapping of a numpy/cv2 ndarray
        img4 = pv3.Image( np.zeros( (480,640), dtype='uint8' ) )
        """
        self.desc = desc
        # metadata dictionary can be used to pass arbitrary info with the image
        self.metadata = {}

        if isinstance(source, np.ndarray):
            self.data = source
            self.metadata['source'] = "np.ndarray"
        elif type(source) == str:
            self.data = cv2.imread(source, *args, **kwargs)
            self.metadata['source'] = "file"
            self.metadata['filename'] = source
        else:
            # assume a file object
            buf = source.read()
            x = np.fromstring(buf, dtype='uint8')
            self.data = cv2.imdecode(x, cv2.IMREAD_UNCHANGED)
            self.metadata['source'] = "file object or buffer"

        self.height, self.width = self.data.shape[0:2]
        self.size = (self.width, self.height)
        self.nchannels = self.data.shape[2] if len(self.data.shape) == 3 else 1

        # Annotation data is a separate BGR image array.
        self.annotation_data = self.data.copy() if self.nchannels == 3 else cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR)

    def __str__(self):
        txt = "Pyvision3 Image: {}".format(self.desc)
        txt += "\nWidth: {}, Height: {}, Channels: {}, Depth: {}".format(self.width, self.height,
                                                                         self.nchannels, self.data.dtype)
        return txt

    def __repr__(self):
        return str(self)

    def __getitem__(self, slc):
        return self.data[slc]

    def as_grayscale(self, as_type="PV"):
        """
        Parameters
        ----------
        as_type: str in ("CV", "PV"), default is "PV"

        Returns
        -------
        A copy of the image (data only, not annotations) as a single channel opencv numpy array,
        or, if as_type is "PV", then a pyvision3 Image wrapped around the same.
        """
        if self.nchannels == 3:
            img_gray = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = self.data.copy()

        if as_type == "CV":
            return img_gray
        else:
            return Image(img_gray)

    def as_annotated(self, alpha=0.5, as_type="PV"):
        """
        Provides an array which represents the merging of the image data
        with the annotations layer.

        Parameters
        ----------
        alpha: float
            Specify a number between 0.0 and 1.0
            This is the alpha blend when merging the annotations layer onto the image data.
            1.0 means that the annotation will be completely opaque and 0.0 completely transparent.
        as_type: string
            Specify either "CV" or "PV" (default) to indicate the return type of the annotated
            image. If "CV", then an ndarray in the normal opencv format is returned.
            If "PV", then a new pyvision3 image is returned with the annotations baked-in.

        Returns
        -------
        A copy of the image array (possibly expanded to 3 channels if the
        original image is single channel) with the color annotations baked in by
        replacing those pixels in the image with the corresponding non-zero pixels
        in the annotation data.

        Return type is either an opencv ndarray or a pyvision3 image depending on as_type.
        """
        # TODO: What if self.data is a floating point image and the annotations
        # are uint8 BGR? We should probably call a normalizing routine of some
        # sort that copies/converts self.data into a 3-channel BGR 8-bit image
        if self.nchannels == 1:
            tmp_img = cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR)
        else:
            tmp_img = self.data.copy()

        # this works because the annotation_data was initialized as a copy of the
        # source data. Annotations draw on this copy, and when we alpha-blend, those pixels
        # that were not changed by an annotation will blend back to full intensity.
        # i.e., if there were no annotations on pixel x, (1-alpha)*I(x) + (alpha)*A(x) = I(x)
        # because A(x) == I(x) where not otherwise annotated.
        tmp_img = cv2.addWeighted(tmp_img, 1.0-alpha, self.annotation_data, alpha, 0.0)

        if as_type == "PV":
            return Image(tmp_img)

        return tmp_img

    def annotate_shape(self, shape, color=(255, 0, 0), fill_color=None, *args, **kwargs):
        """
        Draws the specified shape on the annotation data layer.
        Currently supports LineString, MultiLineString, LinearRing,
        and Polygons.

        Parameters
        ----------
        shape: shapely.geometry shape object
        color:
            An RGB tuple indicating the color, red is (255,0,0)
        fill_color:
            The color used to fill a closed shape (polygon). None means
            that there will be no fill (default).

        *args and **kwargs are for optional line parameters that will
        be passed onto cv2.line(...) which is used at the core for
        drawing the line segments of the shape.

        Note
        ----
        OpenCV stores images in BGR not RGB order, thus the annotation
        layer will be constructed this way. However, to make it easier for the
        user, colors to be used are specified in the normal (r,g,b) tuple order,
        and internally, we handle it.
        """
        # TODO: annotate_shape should support all shapely geometries
        # TODO: annotations should support alpha-channel fills for partial transparency
        if isinstance(shape, sg.LinearRing) or isinstance(shape, sg.LineString):
            self._draw_segments(shape, color, *args, **kwargs)
        elif isinstance(shape, sg.MultiLineString):
            for line_string in shape:
                self._draw_segments(line_string, color, *args, **kwargs)
        elif isinstance(shape, sg.Polygon):
            if fill_color is not None:
                # use cv2.fillPoly
                c = self._fix_color_tuple(fill_color)
                exterior = np.array(shape.exterior.coords, dtype='int')
                interiors = [np.array(x.coords, dtype='int') for x in shape.interiors]
                cv2.fillPoly(self.annotation_data, [exterior] + interiors, color=c)
            # draw external ring of polygon
            self._draw_segments(shape.exterior, color, *args, **kwargs)
            # draw interior rings (holes) if any
            for interior_ring in shape.interiors:
                self._draw_segments(interior_ring, color, *args, **kwargs)

    def annotate_point(self, point, color=(255, 0, 0)):
        """
        Annotates a point by drawing a filled circle of radius-3 pix in the annotation
        layer.

        Parameters
        ----------
        point:  tuple (int: x, int: y) or shapely Point object
        color:  tuple (r,g,b)
        """
        pt = (int(point.x), int(point.y)) if isinstance(point, sg.point.Point) else tuple(point)
        self.annotate_circle(pt, 3, color, thickness=-1)

    def annotate_circle(self, ctr, radius, color=(255, 0, 0), *args, **kwargs):
        """
        Draws a circle on the annotation layer

        Parameters
        ----------
        ctr: (int: x, int: y)
            the center points of the circle

        radius: int
            the radius in pixels of the circle

        color: (r,g,b)

        *args and **kwargs will be passed onto cv2.circle function, and
        can be used to control the thickness and line type. Note that a negative
        thickness indicates that the circle will be filled.
        """
        c = self._fix_color_tuple(color)
        ctr = (int(ctr[0]), int(ctr[1]))
        cv2.circle(self.annotation_data, ctr, radius, c, *args, **kwargs)

    def annotate_line(self, pt1, pt2, color, *args, **kwargs):
        """
        Draws a line segment between two points on the annotation layer.

        Parameters
        ----------
        pt1, pt2: tuples, (int: x, int: y)
            The start and end points of the line segment

        color: tuple (r,g,b)

        *args and **kwargs will be passed onto cv2.line, which can be
        used to control line thickness and style
        """
        c = self._fix_color_tuple(color)
        cv2.line(self.annotation_data, tuple(pt1), tuple(pt2), c, *args, **kwargs)

    def annotate_rect(self, pt1, pt2, color=(255, 0, 0), *args, **kwargs):
        """
        Draws a rectangle using upper left (pt1) and lower right (pt2) coordinates.
        To fill the rectangle, specify thickness=-1 in the kwargs. If you have a
        rectangle as a shapely object (polygon), use the annotate_shape method instead.

        Parameters
        ----------
        pt1, pt2: tuple (int:x, int:y)
            The upper left and lower right corners of the rectangle
        color:  tuple (r,g,b)
            The rgb color of the rectangle
        *args and **kwargs will be passed onto cv2.rectangle, which can be
        used to control line thickness and style.

        Note
        ----
        If your rectangle is a shapely Polygon (which it will be if you use
        pv3.Rect(...) to create it), then use the annotate_shape method instead.
        """
        c = self._fix_color_tuple(color)
        cv2.rectangle(self.annotation_data, tuple(pt1), tuple(pt2), color=c, *args, **kwargs)

    def annotate_text(self, txt, point, color=(0, 0, 0), bg_color=None,
                      font_face=cv2.FONT_HERSHEY_PLAIN, font_scale=1, *args, **kwargs):
        """
        Draws the specified text string to the annotations layer, uses cv2.putText() function
        as the basis, so *args and **kwargs will be passed onto that function.

        Parameters
        ----------
        txt:    str
            The text string to be drawn to the image
        point:  tuple (int: x, int: y)
            The location of the start of the text string
        color:  tuple (r,g,b)
            The color of the text
        bg_color: tuple (r,g,b) or None
            Optional. If specified then a filled box of this color will
            serve as a background for the text. This is handy to ensure
            text legibility.
        font_face: cv2 font selection, defaults to cv2.FONT_HERSHEY_PLAIN
        font_scale: cv2 font Scale, defaults to 1

        *args and **kwargs will be passed onto the underlying cv2.putText
        function.
        """
        c = self._fix_color_tuple(color)
        if bg_color is not None:
            # we will draw a filled rectangle of this color to be behind
            # the annotated text
            ((w, h), _) = cv2.getTextSize(txt, font_face, font_scale, thickness=1)
            point1 = (point[0] - 1, point[1] + 1)
            point2 = (point1[0] + w + 2, point1[1] - h - 2)
            self.annotate_rect(point1, point2, color=bg_color, thickness=-1)

        cv2.putText(self.annotation_data, txt, tuple(point), fontFace=font_face,
                    fontScale=font_scale, color=c, *args, **kwargs)

    def annotate_mask(self, mask_img, transparency=(0, 0, 0)):
        """
        Replaces any and all current annotations on the image with the provided
        3-channel 8-bit mask image. To change what mask pixels are considered
        transparent, use the set_annotation_transparency method.

        Parameters
        ----------
        mask_img:       ndarray (3 channel, uint8 image, same size as self.data, BGR color order)
        transparency:   pixels in the input mask_img that match this color will be set as transparent,
                        and thus the background image's pixel value will be used instead.
                        None means there are no fully transparent pixels. Default is black (0, 0, 0),
                        so anything black in the mask_img will be replaced with the corresponding color
                        value from the underlying image.
        """
        if mask_img.shape[0:2] != self.data.shape[0:2]:
            raise ImageAnnotationError("Invalid mask. Must be same (w,h) as image.")
        self.annotation_data = mask_img.copy()
        if transparency is not None:
            pix = np.nonzero((self.annotation_data == transparency).all(axis=2))
            self.annotation_data[pix] = self.data[pix]

    def annotate_inset_image(self, inset_image, pos, size=None):
        """
        On the annotation layer, composites another image, typically a thumbnail or other graphic,
        at the specified inset position and size. This is a way that an icon image could be drawn
        to a specific location as an annotation, or for an inset graphic like a performance meter,
        to be drawn as an inset onto a corner of the image, etc.

        Parameters
        ----------
        inset_image:    A pyvision3 image to be drawn as an annotation on the current image. This may be
                        a single channel or 3 channel image.
        pos:            The (x, y) coordinates, relative to the current image, where the top-left corner
                        of the inset image will be placed.
        size:           The (width, height) of the inset image in pixels. The inset image will be resized
                        to this specified size before drawing it to the annotation layer. Specify None
                        if no resizing is required of the inset image.

        Note
        -----
        The pixels from the inset image will replace whatever is already at that location in the annotation
        layer. Thus, the transparency color already defined for the current image will apply to the
        inset image as well. The inset image will be subject to the same alpha-blending as the other annotations.
        """
        if size is not None:
            if inset_image.size != size:
                inset_image = inset_image.resize(size, as_type="PV")
        else:
            size = inset_image.size

        cv_tile = inset_image.data
        roi = Rect(pos[0], pos[1], size[0], size[1])
        depth = cv_tile.shape[-1] if len(cv_tile.shape) == 3 else 1

        if depth == 1:
            cv_tile_bgr = cv2.cvtColor(cv_tile, cv2.COLOR_GRAY2BGR)
        else:
            cv_tile_bgr = cv_tile

        # copy pixels of tile onto appropriate location in annotation image
        dest = self.annotation_data
        (minx, miny, maxx, maxy) = integer_bounds(roi)
        dest[miny:(maxy+1), minx:(maxx+1), :] = cv_tile_bgr

    def _draw_segments(self, simple_shape, color, *args, **kwargs):
        """
        Internal method for drawing a "simple" shapely geometric object,
        i.e., one that has a single set of coordinates which will be
        connected in sequence. Examples are LineStrings and LinearRings,
        the latter being a closed sequence of points and the former is not.

        Parameters
        ----------
        simple_shape:   shape
            A shape object that provides a coords member
        color:  tuple (r,g,b)
        *args, **kwargs:
            will be passed on to self.annotate_line for each
            line segment in the simple shape.
        """
        points = [(int(x), int(y)) for (x, y) in simple_shape.coords]
        for i in range(1, len(points)):
            pt1 = points[i - 1]
            pt2 = points[i]
            self.annotate_line(pt1, pt2, color, *args, **kwargs)

    def _fix_color_tuple(self, color):
        """
        Puts the color tuple (r,g,b) into (b,g,r) order.

        Parameters
        ----------
        color:  tuple (r,g,b)

        Returns
        -------
        tuple (b,g,r)
        """
        (r, g, b) = color
        return (b, g, r)

    def copy(self):
        """
        Returns
        -------
        Returns a pyvision3 image which is a 'deep copy' of this image which can be
        freely modified without affecting the source of the copy.
        """
        new_data = self.data.copy()
        new_img = Image(new_data)
        new_img.annotation_data = self.annotation_data.copy()
        new_img.metadata = self.metadata.copy()
        return new_img

    def crop(self, rect):
        """
        Crops a rectangular region from this image and returns as
        a new (copied) pyvision3 image

        Parameters
        ----------
        rect:   shapely rectangle (polygon)

        Returns
        -------
        A pyvision3 image with only the contents of the rectangular area. The .metadata of the
        returned crop image will have a key added "crop_bounds" with value being a tuple of
        the integer crop coordinates used to generate it (minx, miny, maxx, maxy).

        Raises an OutOfBounds exception if the rectangle being cropped is
        partially or fully outside the bounds of the image.
        """
        if not in_bounds(rect, self):
            raise OutOfBoundsError("Cropping rectangle {} is out of bounds.".format(rect.bounds))
        (minx, miny, maxx, maxy) = integer_bounds(rect)
        cropped = self.data[miny:(maxy+1), minx:(maxx+1)].copy()
        crop_image = Image(cropped)
        crop_image.metadata = self.metadata.copy()
        crop_image.metadata["crop_bounds"] = (minx, miny, maxx, maxy)
        return crop_image

    def resize(self, new_size, keep_aspect=False, as_type="PV"):
        """
        Returns a copy of the image after resizing to a new size.

        Parameters
        ----------
        new_size: tuple (width, height)
        keep_aspect: boolean
            If True, then the resizing will preserve the original image's aspect
            ratio, which may require borders "letterboxing" to be introduced.
            Default is False.
        as_type: str in ("CV","PV")
            If as_type is "CV", then the returned image is an opencv
            format ndarray. If "PV" (default), then a pyvision3 image is returned.

        Returns
        -------
        An opencv or pyvision3 image representing the resized image depending on as_type.
        If the return is a pyvision3 image, there will also be a .metadata field "original_size"
        that stores the original size of the image.
        """
        if keep_aspect:
            # Find the scale
            w, h = self.size  # current size

            scale = min(new_size[0] / w, new_size[1] / h)
            w = int(scale * w)
            h = int(scale * h)

            # Create new image with resized tmp image centered
            tmp = cv2.resize(self.data, (w, h))
            new = np.zeros((new_size[1], new_size[0], self.nchannels), dtype=tmp.dtype)
            x = (new_size[0] - w) // 2
            y = (new_size[1] - h) // 2
            new[y:(y+h), x:(x+w), :] = tmp
        else:
            new = cv2.resize(self.data, new_size)

        if as_type == "PV":
            new_image = Image(new)
            new_image.metadata = self.metadata.copy()
            new_image.metadata["original_size"] = self.size
            return new_image
        else:
            return new

    def imshow(self, **kwargs):
        """
        Displays this image in a matplotlib figure. The same as calling img.show() method
        with highgui=False.

        Parameters
        ----------
        kwargs: key word arguments are the same as used in the show method.
        """
        if kwargs is None:
            kwargs = {}
        kwargs.update({"highgui": False})
        self.show(**kwargs)

    def show(self, window_title=None, highgui=True, annotations=True, annotations_opacity=0.5,
             delay=0, pos=None):
        """
        Displays this image in a highgui or matplotlib window.

        Parameters
        ----------
        window_title: string or None
            If None (default) then the image title will be self.desc, otherwise
            specify a string to be used for this purpose.
        highgui: boolean
            If True (default) then opencv highgui library is used to display the image.
            Otherwise the current figure of matplotlib will be used.
        annotations: boolean
            If True (default) then the annotations will be shown over the base image,
            otherwise only the base image will be shown.
        annotations_opacity: float (0.0 - 1.0)
            Controls the opacity of the annotations layer, assuming annotations=True.
            0.0 means that the annotations will be invisible and 1.0 means completely opaque.
            Default is 0.5.
        delay: int
            The delay in milliseconds to wait after showing the image. This is passed on
            to cv2.waitKey if highgui is specified as the display. This is useful if
            showing a sequence of images as if they were a video. If None, then there
            is no call to cv2.waitkey
        pos: tuple (x,y)
            Used if showing via highgui, this is the window's upper left position on
            the user's screen. Default is None, which means the window will be placed
            automatically.
        """
        if not window_title:
            window_title = self.desc

        img_array = self.as_annotated(alpha=annotations_opacity, as_type="CV") if annotations \
            else self.data.copy()
        # optional resize logic here?

        if highgui:
            # display via  an opencv window
            # Create the window
            cv2.namedWindow(window_title)

            # Set the location
            if pos is not None:
                cv2.moveWindow(window_title, pos[0], pos[1])

            # Display the result
            cv2.imshow(window_title, img_array)
            key = -1 if delay is None else cv2.waitKey(delay=delay)
            del img_array
            return key
        else:
            # display in a matplotlib figure
            if img_array.shape[-1] == 3:
                # Note cv2 image arrays are BGR order, but matplotlib expects
                # RGB order. So we're swapping channel 0 with channel 2
                tmp = img_array[:, :, 0].copy()
                img_array[:, :, 0] = img_array[:, :, 2]
                img_array[:, :, 2] = tmp
                del tmp
            if window_title is not None:
                plot.figure()
            plot.imshow(img_array)
            plot.title(window_title)
            plot.draw()
            plot.show(block=False)

    def show_annotation(self, window_title=None, highgui=True, delay=0, pos=None):
        """
        Display only the annotations layer of this image.
        """
        tmp = Image(self.annotation_data)
        tmp.show(window_title=window_title, highgui=highgui, delay=delay, pos=pos,
                 annotations=False)

    def save(self, filename, *args, as_annotated=True, **kwargs):
        """
        Saves the image data (or the annotated image data) to a file.
        This wraps cv2.imwrite function.

        Parameters
        ----------
        filename: String
            The filename, including extension, for the saved image
        as_annotated: Boolean
            If True (default) then the annotated version of the image will be saved.
        All other args and kwargs are passed on to cv2.imwrite
        """
        img_array = self.as_annotated(as_type="CV") if as_annotated else self.data
        cv2.imwrite(filename, img_array, *args, **kwargs)


def matplot_fig_to_image(fig):
    """
    Converts a matplotlib figure into a pyvision3 image

    Parameters
    ----------
    fig:    a matplotlib figure

    Returns
    -------
    a pyvision3 image, the width/height of the image are determined
    by the matplotlib figure size (in inches) times the specified dpi (dots per inch),
    which can be set upon figure creation.
    """
    buff = io.BytesIO()
    fig.savefig(buff, format='png', transparent=False, facecolor='white')
    arr = np.asarray(buff.getbuffer())
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plot.close(fig)
    return Image(img)
