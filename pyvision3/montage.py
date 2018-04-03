"""
Created on Feb 23, 2016
@author: Stephen O'Hara

This module defines the pyvision montage classes,
which are UIs that show a montage of images (or videos)
in a single window. This is useful for visualizing
a set of results.
"""
import cv2
import weakref
import numpy as np

from .image import Image
from .geometry import Rect, Point, integer_bounds
from .video import VideoInterface


class ImageMontage(object):
    """
    Displays thumbnails of a list of input images as a single
    'montage' image. Supports scrolling if there are more images
    than "viewports" in the layout.
    """

    def __init__(self, image_list, layout=(2, 4), tile_size=(64, 48), gutter=2, by_row=True, labels='index',
                 keep_aspect=True, highlight_selected=False):
        """
        Constructor

        Parameters
        ----------
        image_list: list
            A list of pyvision3 images that you wish to display as a montage.
        layout: tuple (int, int)
            (rows,cols) that indicates the number of tiles to show in a single montage page, oriented in a grid.
        tile_size: tuple (int, int)
            The size (width, height) of each thumbnail image to display in the montage.
        gutter: int
            The width in pixels of the gutter between thumbnails.
        by_row: boolean
            If true, the image tiles are placed in row-major order, that is, one row of the montage is filled
            before moving to the next. If false, then column order is used instead.
        labels: list<str>, tuple<str>, or 'index'
            Used to show a label at the lower left corner of each image in the montage. If this parameter is a list,
            then it should be the same length as len(image_list) and contain the label to be used for the
            corresponding image. If labels == 'index', then the image montage will simply display the index of the
            image in image_list. Set labels to None to suppress labels.
        keep_aspect: boolean
            If true the original image aspect ratio will be preserved.
        highlight_selected: boolean
            If true, any image tile in the montage which has been clicked will be drawn with a rectangular
            highlight. This will toggle, such that if an image is clicked a second time, the highlighting
            will be removed. The methods get_highlighted and set_highlighted can be used to set/retrieve the
            images in the montage that are highlighted.
        """
        self._tileSize = tile_size
        self._rows = layout[0]
        self._cols = layout[1]
        self._images = image_list
        self._gutter = gutter
        self._by_row = by_row
        self._txtcolor = (255, 255, 255)
        self._imgPtr = 0
        self._labels = tuple(labels) if isinstance(labels, list) else labels
        self._clickHandler = ClickHandler(self)
        self._keep_aspect = keep_aspect
        self._image_positions = []
        self._select_handler = None
        self._highlighted = highlight_selected
        self._selected_tiles = []  # which images have been selected (or clicked) by user

        # check if we need to allow for scroll-arrow padding
        if self._rows * self._cols < len(image_list):
            if by_row:
                self._xpad = 0
                self._ypad = 25
            else:
                self._ypad = 0
                self._xpad = 25
        else:
            # there will be no scrolling required
            self._xpad = 0
            self._ypad = 0

        img_width = self._cols * (tile_size[0] + gutter) + gutter + 2 * self._xpad
        img_height = self._rows * (tile_size[1] + gutter) + gutter + 2 * self._ypad
        self._size = (img_width, img_height)

        self._cvMontageImage = np.zeros((img_height, img_width, 3), dtype='uint8')

        self._init_decrement_arrow()  # build the polygon for the decrement arrow
        self._init_increment_arrow()  # build the polygon for the increment arrow
        self.draw()  # compute the initial montage image

    def draw(self):
        """
        Computes the image montage from the source images based on the current
        image pointer (position in list of images), etc. This internally constructs
        the montage, but show() is required for display and mouse-click handling.
        """
        self._cvMontageImage[:] = 0  # erase everything to start

        img_ptr = self._imgPtr
        if img_ptr > 0:
            # we are not showing the first few images in imageList
            # so display the decrement arrow
            cv2.fillConvexPoly(self._cvMontageImage, self._decrArrow, (125, 125, 125))

        if img_ptr + (self._rows * self._cols) < len(self._images):
            # we are not showing the last images in imageList
            # so display increment arrow
            cv2.fillConvexPoly(self._cvMontageImage, self._incrArrow, (125, 125, 125))

        self._image_positions = []
        if self._by_row:
            for row in range(self._rows):
                for col in range(self._cols):
                    if img_ptr > len(self._images) - 1:
                        break
                    tile = self._images[img_ptr].as_annotated(as_type="PV")
                    self._composite(tile, (row, col), img_ptr)
                    img_ptr += 1
        else:
            for col in range(self._cols):
                for row in range(self._rows):
                    if img_ptr > len(self._images) - 1:
                        break
                    tile = self._images[img_ptr].as_annotated(as_type="PV")
                    self._composite(tile, (row, col), img_ptr)
                    img_ptr += 1

    def as_image(self):
        """
        If you don't want to use the montage's built-in mouse-click handling by calling
        the ImageMontage.show() method, then this method will return the montage image
        computed from the last call to draw().
        """
        return Image(self._cvMontageImage)

    def show(self, window_title="Image Montage", pos=None, delay=0):
        """
        Will display the montage image, as well as register the mouse handling callback
        function so that the user can scroll the montage by clicking the increment/decrement
        arrows.

        Parameters
        ----------
        window_title: str
            The window title to use
        pos: tuple
            The window position on the user's screen (x,y)
        delay: int
            The delay (waitKey timeout) to display the montage, default is zero (indefinite),
            otherwise specify milliseconds.

        Returns
        -------
        The key code of the key pressed, if any, that dismissed the window.
        """
        img = self.as_image()
        cv2.namedWindow(window_title)
        cv2.setMouseCallback(window_title, self._clickHandler.onClick, window_title)
        key = img.show(window_title=window_title, highgui=True, pos=pos, delay=delay)
        return key

    def set_select_handler(self, handler):
        """
        Add a function that will be called when an image is selected.
        The handler function should take an image, the image index,
        a list of labels, and a dictionary of other info.
        """
        self._select_handler = handler

    def set_highlighted(self, idxs):
        """
        If the montage was created with highlight_selected option enabled,
        then this function will cause a set of tiles in the montage to be
        highlighted.
        @note: Calling this method will erase any previous selections made
        by the user.
        """
        self._selected_tiles = idxs
        self.draw()

    def get_highlighted(self):
        """
        Returns the index list of the tiles which were selected/highlighted
        by the users
        """
        return sorted(self._selected_tiles)

    def _check_click_region(self, x, y):
        """
        internal method to determine the clicked region of the montage.
        @return: -1 for decrement region, 1 for increment region, and 0 otherwise.
        If a select handler function was defined (via set_select_handler), then
        this function will be called when the user clicks within the region
        of one of the tiles of the montage. Signature of selectHandler function
        is f(img, imgNum, dict). As of now, the only key/value pair passed
        into the dict is "imgLabel":<label>.
        """
        if self._by_row:
            # scroll up/down to expose next/prev row
            decr_rect = Rect(0, 0, self._size[0], self._ypad)
            incr_rect = Rect(0, self._size[1] - self._ypad, self._size[0], self._ypad)
        else:
            # scroll left/right to expose next/prev col
            decr_rect = Rect(0, 0, self._xpad, self._size[1])
            incr_rect = Rect(self._size[0] - self._xpad, 0, self._xpad, self._size[1])

        pt = Point(x, y)
        if incr_rect.contains(pt):
            # print "DEBUG: Increment Region"
            return 1
        elif decr_rect.contains(pt):
            # print "DEBUG: Decrement Region"
            return -1
        else:
            # print "DEBUG: Neither Region"
            for img, imgNum, rect in self._image_positions:
                if rect.contains(pt):
                    if imgNum in self._selected_tiles:
                        self._selected_tiles.remove(imgNum)
                    else:
                        self._selected_tiles.append(imgNum)
                    if self._select_handler is not None:
                        imgLabel = self._labels[imgNum] if isinstance(self._labels, tuple) else str(imgNum)
                        self._select_handler(img, imgNum, {"imgLabel":imgLabel})
            return 0

    def _init_decrement_arrow(self):
        """
        internal method to compute the list of points that represents
        the appropriate decrement arrow (leftwards or upwards) depending
        on the image montage layout.
        """
        if self._by_row:
            # decrement upwards
            x1 = self._size[0] / 2
            y1 = 2
            halfpad = self._ypad / 2
            self._decrArrow = np.array([(x1, y1),
                                        (x1 + halfpad, self._ypad - 2),
                                        (x1 - halfpad, self._ypad - 2)],
                                       dtype="int32")
        else:
            # decrement leftwards
            x1 = 2
            y1 = self._size[1] / 2
            halfpad = self._xpad / 2
            self._decrArrow = np.array([(x1, y1),
                                        (x1 + self._xpad - 3, y1 - halfpad),
                                        (x1 + self._xpad - 3, y1 + halfpad)],
                                       dtype="int32")

    def _init_increment_arrow(self):
        """
        internal method to compute the list of points that represents
        the appropriate increment arrow (rightwards or downwards) depending
        on the image montage layout.
        """
        if self._by_row:
            # increment downwards
            x1 = self._size[0] / 2
            y1 = self._size[1] - 3
            halfpad = self._ypad / 2
            self._incrArrow = np.array([(x1, y1),
                                        (x1 + halfpad, y1 - self._ypad + 3),
                                        (x1 - halfpad, y1 - self._ypad + 3)],
                                       dtype="int32")
        else:
            # increment rightwards
            x1 = self._size[0] - 2
            y1 = self._size[1] / 2
            halfpad = self._xpad / 2
            self._incrArrow = np.array([(x1, y1),
                                        (x1 - self._xpad + 2, y1 - halfpad),
                                        (x1 - self._xpad + 2, y1 + halfpad)],
                                       dtype="int32")

    def _decr(self):
        """
        internal method used by _onClick to compute the new imgPtr location after a decrement
        """
        tmp_ptr = self._imgPtr
        if self._by_row:
            tmp_ptr -= self._cols
        else:
            tmp_ptr -= self._rows
        if tmp_ptr < 0:
            self._imgPtr = 0
        else:
            self._imgPtr = tmp_ptr

    def _incr(self):
        """
        internal method used by _onClick to compute the new imgPtr location after an increment
        """
        tmp_ptr = self._imgPtr
        if self._by_row:
            tmp_ptr += self._cols
        else:
            tmp_ptr += self._rows

        self._imgPtr = tmp_ptr

    def _composite(self, img, pos, img_num):
        """
        Internal method to composite the thumbnail of a given image into the
        correct position, given by (row,col).

        Parameters
        ----------
        img: pyvision3 Image
            The image from which a thumbnail will be composited onto the montage
        pos: tuple
            A tuple (row,col) for the position in the montage layout
        img_num: int
            The image index of the tile being drawn, this helps us display the
            appropriate label in the lower left corner if self._labels is not None.
        """
        (row, col) = pos

        tile = img.resize(self._tileSize, keep_aspect=self._keep_aspect)

        pos_x = col * (self._tileSize[0] + self._gutter) + self._gutter + self._xpad
        pos_y = row * (self._tileSize[1] + self._gutter) + self._gutter + self._ypad

        cvImg = self._cvMontageImage
        cvTile = tile.data
        roi = Rect(pos_x, pos_y, self._tileSize[0], self._tileSize[1])

        # Save the position of this image
        self._image_positions.append([self._images[img_num], img_num, roi])

        depth = cvTile.shape[-1] if len(cvTile.shape) == 3 else 1

        if depth == 1:
            cvTileBGR = cv2.cvtColor(cvTile, cv2.COLOR_GRAY2BGR)
        else:
            cvTileBGR = cvTile

        # copy pixels of tile onto appropriate location in montage image
        (minx, miny, maxx, maxy) = integer_bounds(roi)
        cvImg[miny:(maxy+1), minx:(maxx+1), :] = cvTileBGR

        if self._labels == 'index':
            # draw image number in lower left corner, respective to ROI
            lbltext = "%d" % img_num
        elif isinstance(self._labels, tuple):
            lbltext = str(self._labels[img_num])
        else:
            lbltext = None

        if lbltext is not None:
            ((tw, th), _) = cv2.getTextSize(lbltext, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if tw > 0 and th > 0:
                cv2.rectangle(cvImg,
                              (pos_x, pos_y + self._tileSize[1] - 1),
                              (pos_x + tw + 1, pos_y + self._tileSize[1] - (th + 1) - self._gutter),
                              color=(0, 0, 0),
                              thickness=cv2.FILLED)
                color = self._txtcolor
                cv2.putText(cvImg,
                            lbltext,
                            (pos_x + 1, pos_y + self._tileSize[1] - self._gutter - 2),  # location
                            cv2.FONT_HERSHEY_SIMPLEX,  # font face
                            0.5,  # font scale
                            color)

        if self._highlighted and (img_num in self._selected_tiles):
            # draw a highlight around this image
            cv2.rectangle(cvImg,
                          (int(minx), int(miny)),
                          (int(maxx), int(maxy)),
                          (0, 255, 255),
                          thickness=4)


class ClickHandler(object):
    """
    A class for objects designed to handle click events on ImageMontage objects.
    We separate this out from the ImageMontage object to address a memory leak
    when using cv.SetMouseCallback(window, self._onClick, window), because we
    don't want the image data associated with the click handler
    """

    def __init__(self, montage_object):

        self.montage_ptr = weakref.ref(montage_object)

    def onClick(self, event, x, y, flags, window):
        """
        Handle the mouse click for an image montage object.
        Increment or Decrement the set of images shown in the montage
        if appropriate.
        """
        montage = self.montage_ptr()  # use weak reference to image montage to prevent mem leak
        if montage is None:
            # if the reference was deleted already...
            return

        # print "event",event
        if event == cv2.EVENT_LBUTTONDOWN:
            rc = montage._check_click_region(x, y)
            if rc == -1 and montage._imgPtr > 0:
                # user clicked in the decrement region
                montage._decr()
            elif rc == 1 and montage._imgPtr < (len(montage._images) - (montage._rows * montage._cols)):
                montage._incr()
            else:
                pass # do nothing

            montage.draw()
            cv2.imshow(window, montage._cvMontageImage)


class VideoMontage(VideoInterface):
    """
    Provides a visualization of several videos playing back in
    a single window. This can be very handy, for example, to
    show tracking results of multiple objects from a single video,
    or for minimizing screen real-estate when showing multiple
    video sources.

    A video montage object is an iterator, so you "play" the
    montage by iterating through all the frames, just as with
    a standard video object.
    """

    def __init__(self, video_dict, layout=(2, 4), tile_size=(64, 48)):
        """
        Parameters
        ----------
        video_dict: A dictionary of videos to display in the montage. The keys are the video labels, and
            the values are objects adhering to the pyvision3 video interface. (pv.Video, pv.VideoFromImages, etc.)
        layout: A tuple of (rows,cols) to indicate the layout of the montage. Videos will be separated by
            a one-pixel gutter. Videos will be drawn to the montage such that a row is filled up prior to moving
            to the next. The videos are drawn to the montage in the sorted order of the video keys in the dictionary.
        tile_size: The window size to display each video in the montage. If the video frame sizes are larger than
            this size, it will be cropped. If you wish to resize, use the size option in the pv.Video class to have
            the output size of the video resized appropriately.
        """
        super().__init__(size=None)
        if len(video_dict) < 1:
            raise ValueError("You must provide at least one video in the video_dict variable.")
        self.vids = video_dict
        self.layout = layout
        self.vid_size = tile_size
        self.imgs = {}
        self.stopped = []

    def reset(self):
        for key in self.vids:
            v = self.vids[key]
            v.reset()
        self.current_frame = None
        self.current_frame_num = 0
        self.stopped = []
        self.imgs = {}

    def __next__(self):
        if len(self.stopped) == len(self.vids.keys()):
            print("All Videos in the Video Montage Have Completed.")
            raise StopIteration

        # get next image from each video and put on montage
        # if video has ended, continue to display last image
        # stop when all videos are done.
        for key in self.vids:
            if key in self.stopped:
                continue  # this video has already reached its end.
            v = self.vids[key]
            try:
                tmp = next(v)
                self.imgs[key] = tmp
            except StopIteration:
                # print "End of a Video %s Reached"%key
                self.stopped.append(key)

        keys = sorted(self.imgs.keys())
        image_list = []
        for k in keys:
            image_list.append(self.imgs[k])

        # create an image montage from the current video frames and advance the frame counter
        im = ImageMontage(image_list, self.layout, self.vid_size, gutter=2, by_row=True, labels=keys)
        self.current_frame = im.as_image()
        self.current_frame_num += 1

        return self._get_resized()

