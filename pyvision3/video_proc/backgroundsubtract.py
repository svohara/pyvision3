"""
Created: Oct 22, 2010
Author: Stephen O'Hara
Modified: Mar 11, 2016
    For pyvision3 compatibility.
"""
import numpy as np
import pyvision3 as pv3
import math
# import cv2

# Constants used to identify a background subtraction method,
# useful, for example, for specifying which method to use in the
# MotionDetector class.
BG_SUBTRACT_STATIC = "BG_SUBTRACT_STATIC"       # static bg model image
BG_SUBTRACT_FRAME_DIFF = "BG_SUBTRACT_FD"       # frame difference
# BG_SUBTRACT_MCFD = "BG_SUBTRACT_MCFD"         # motion compensated frame difference
BG_SUBTRACT_MEDIAN = "BG_SUBTRACT_MM"           # median model
BG_SUBTRACT_APPROX_MEDIAN = "BG_SUBTRACT_AM"    # approx median

# TODO: Port the motion compensated frame differencer from old pyvision3 + OpticFlow


class AbstractBGModel:
    def __init__(self, image_buffer, thresh=80, soft_thresh=False):
        """
        Parameters
        ----------
        image_buffer: ImageBuffer
            An ImageBuffer object that has already been filled
            with the appropriate number of images. (Provide a full buffer...so a few
            frames of initialization will be required in most cases to fill up a
            newly created buffer.)
        thresh: int
            A noise threshold to remove very small differences.
        soft_thresh: boolean
            Selects whether soft thresholding is used
        """
        self._image_buffer = image_buffer
        self._threshold = thresh
        self._softThreshold = soft_thresh
        
    def _compute_bg_diff(self):
        """
        This private method should be overridden by a concrete background subtraction
        class to yield a difference image from the background model.

        Returns
        -------
        A numpy single channel ndarray representing the difference image
        """
        raise NotImplementedError
    
    def foreground_mask(self):
        """
        Returns
        -------
        A mask image indicating which pixels are considered foreground. Returns an ndarray
        of type uint8. For some methods, the mask will be binary (consisting only of the values 0 or 255),
        or if soft threshold is used, then the full range of intensities will be returned.

        Note
        ----
        One may wish to perform additional morphological operations
            on the foreground mask prior to use.
        """
        diff = self._compute_bg_diff()
        if self._softThreshold:
            mask = 1 - math.e**(-(1.0*diff)/self._threshold)  # element-wise exp weighting
        else:
            mask = (np.absolute(diff) > self._threshold)
            # mu = np.mean(diff)
            # sigma = np.std(diff)
            # mask = np.absolute((diff-mu)/sigma) > self._threshold
        mask = (mask * 255).astype('uint8')
        return pv3.Image(mask)
        

class StaticModel(AbstractBGModel):
    """
    Uses a single static image as the fixed background model
    """
    def __init__(self, image_buffer, bg_image=None, thresh=80, soft_thresh=False):
        """
        Parameters
        ----------
        bg_image: pyvision3 Image
            The image that will serve as the background model
        """
        if bg_image is None:
            raise ValueError("You must supply a background image for use with the StaticModel")
        AbstractBGModel.__init__(self, image_buffer, thresh=thresh, soft_thresh=soft_thresh)
        self._bg_array = bg_image.as_grayscale(as_type="CV")

    def _compute_bg_diff(self):
        cur_img_array = self._image_buffer.last().as_grayscale(as_type="CV")
        delta = np.absolute(cur_img_array - self._bg_array)
        return delta


class FrameDifferenceModel(AbstractBGModel):
    """
    This class is useful for simple N-frame differencing method of
    background subtraction. If you have a stationary camera, this can
    be a simple and effective way to isolate people/moving objects
    from the background scene.
    
    FrameDifferencer uses ImageBuffer for operation. Assume the buffer
    size is 5. The output of the frame differencing operation will
    be based on the middle image, the 3rd in the buffer. The output
    is the intersection of the following two absolute differences:
    abs(Middle-First) AND abs(Last-Middle).
    """
    def _compute_bg_diff(self):
        prev_img = self._image_buffer.first().as_grayscale(as_type="CV")
        cur_img = self._image_buffer.middle().as_grayscale(as_type="CV")
        next_img = self._image_buffer.last().as_grayscale(as_type="CV")
        
        delta1 = np.absolute(cur_img - prev_img)   # frame diff 1
        delta2 = np.absolute(next_img - cur_img)   # frame diff 2
        
        # use element-wise minimum of the two difference images, which is what
        # gets compared to threshold to yield foreground mask
        return np.minimum(delta1, delta2)


class MedianModel(AbstractBGModel):
    """
    Uses median pixel values of the images in a buffer to
    approximate a background model.
    """
    def _get_median_vals(self):
        """
        Returns
        -------
        A numpy ndarray representing the gray-scale median values of the image stack.
        If you want a pyvision3 image, just wrap the result in pv3.Image(result).
        """
        self._imageStack = self._image_buffer.as_image_stack_BW()
        medians = np.median(self._imageStack, axis=0)  # median of each pixel jet in stack
        return medians
    
    def _compute_bg_diff(self):
        img_gray = self._image_buffer.last().as_grayscale(as_type="CV")
        img_BG = self._get_median_vals()
        return img_gray - img_BG
            
            
class ApproximateMedianModel(MedianModel):
    """
    Approximates the median pixels via an efficient incremental algorithm that
    would converge to the true median in a perfect world. It initializes a
    median image based on the images in the initial image buffer, but
    then only updates the median image using the last (newest) image in the
    buffer.
    """
    def __init__(self, image_buffer, thresh=80, soft_thresh=False):
        if not image_buffer.is_full():
            raise ValueError("Image Buffer must be full before initializing Approx. Median Filter.")
        MedianModel.__init__(self, image_buffer, thresh, soft_thresh)
        self._medians = self._get_median_vals()
        
    def _update_median(self):
        cur_img = self._image_buffer.last()
        cur_mat = cur_img.as_grayscale(as_type="CV")
        median = self._medians
        up = (cur_mat > median)*1.0
        down = (cur_mat < median)*1.0
        self._medians = self._medians + up - down
        
    def _compute_bg_diff(self):
        self._update_median()
        img_gray = self._image_buffer.last().as_grayscale(as_type="CV")
        img_BG = self._medians
        return img_gray - img_BG

