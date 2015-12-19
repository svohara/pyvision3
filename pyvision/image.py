'''
Created on Dec 18, 2015
@author: Stephen O'Hara

This module defines the pyvision image class, which wraps
the image data (array) in a class with convenient methods.

One very useful aspect of the image class is that annotations
(such as bounding boxes that might be drawn around detection regions)
are kept on a separate layer, as opposed to drawing directly
onto the image array itself.
'''

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plot
except ImportError:
    print("Error importing matplotlib.")
    print("Matplotlib integration will not work")


class Image(object):
    '''
    A pyvision3 Image object contains the image data, an
    annotations layer, and many convenient methods.
    Supports 1 channel and 3 channel images.
    '''

    def __init__(self, *args, desc="Pyvision Image", **kwargs):
        '''
        The constructor wraps a cv2.imread(...) function,
        passing in the args and kwargs appropriately, and
        then allocating data for the annotations layer.
        
        Parameters
        ----------
        desc: string
            Provide a short description of this image, that will be used
            by default in window titles and other functions
        args: variable
            Other args will be passed through to cv2.imread, the first arg
            should be the image source, like a file name. See the cv2 docs
            on imread for more details.
        kwargs: variable
            Keyword args will be passed through to cv2.imread
            
        Examples
        --------
        img = pv3.Image('mypic.jpg')
        img2 = pv3.Image('mypic.jpg', cv2.IMREAD_GRAYSCALE)
        '''
        self.desc = desc
        self.data = cv2.imread(*args, **kwargs)
        self.height, self.width = self.data.shape[0:2]
        self.nchannels = self.data.shape[2] if len(self.data.shape) == 3 else 1
        self.annotation_data = np.zeros((self.height,self.width, 3), dtype='uint8')  #3-channel color annotations
        
    def __str__(self):
        s = "Pyvision3 Image: {}".format(self.desc)
        s += "\nWidth: {}, Height: {}, Channels: {}, Depth: {}".format(self.width, self.height, 
                                                                    self.nchannels, self.data.dtype)
        return s
    
    def __repr__(self):
        return str(self)
    
    def __getitem__(self, slc):
        return self.data[slc]
        
    def as_annotated(self):
        '''
        Provides an array which represents the merging of the image data
        with the annotations layer.
        
        Returns
        -------
        A copy of the image array (possibly expanded to 3 channels if the
        original image is single channel) with the color annotations baked in by
        replacing those pixels in the image with the corresponding non-zero pixels
        in the annotation data.
        '''
        #TODO: What if self.data is a floating point image and the annotations
        # are uint8 BGR? We should probably call a normalizing routine of some
        # sort that copies/converts self.data into a 3-channel BGR 8-bit image
        mask = self.annotation_data.nonzero()
        if self.nchannels == 1:
            tmp_img = cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR)
        else:
            tmp_img = self.data.copy()
        tmp_img[mask] = self.annotation_data[mask]
        return tmp_img
        
    def show(self, window_title=None, highgui=False, annotations=True, delay=0, pos=None):
        '''
        Displays this image in a highgui or matplotlib window.
        
        Parameters
        ----------
        window_title: string or None
            If None (default) then the image title will be self.desc, otherwise
            specify a string to be used for this purpose.
        highgui: boolean
            If True then opencv highgui library is used to display the image.
            Otherwise (default) the current figure of matplotlib will be used.
        annotations: boolean
            If True (default) then the annotations will be shown over the base image,
            otherwise only the base image will be shown.
        delay: int
            The delay in milliseconds to wait after showing the image. This is passed on to cv2.waitKey
            if highgui is specified as the display. This is useful if showing a sequence
            of images as if they were a video.
        pos: tuple (x,y)
            Used if showing via highgui, this is the window's upper left position on the user's screen.
            Default is None, which means the window will be placed automatically.
        '''
        if not window_title:
            window_title = self.desc

        img_array = self.as_annotated() if annotations else self.data.copy()
        #optional resize logic here?
        
        if highgui:
            # display via  an opencv window
            # Create the window
            cv2.namedWindow(window_title)
            
            # Set the location
            if pos is not None:
                cv2.moveWindow(window_title, pos[0], pos[1])
            
            # Display the result
            cv2.imshow(window_title, img_array)
            key = cv2.waitKey(delay=delay)
            del img_array
            return key
        else:
            #display in a matplotlib figure
            if img_array.shape[-1] == 3:
                #Note cv2 image arrays are BGR order, but matplotlib expects
                # RGB order. So we're swapping channel 0 with channel 2
                tmp = img_array[:,:,0].copy()
                img_array[:,:,0] = img_array[:,:,2]
                img_array[:,:,2] = tmp
                del(tmp)
            plot.imshow(img_array)
            plot.title(window_title)
            plot.draw()
            plot.show(block=False)
            
    def save(self, filename, *args, as_annotated=True, **kwargs):
        '''
        Saves the image data (or the annotated image data) to a file.
        This wraps cv2.imwrite function.
        
        Parameters
        ----------
        filename: String
            The filename, including extension, for the saved image
        as_annotated: Boolean
            If True (default) then the annotated version of the image will be saved.
        All other args and kwargs are passed on to cv2.imwrite
        '''
        img_array = self.as_annotated() if as_annotated else self.data
        cv2.imwrite(filename, img_array, *args, **kwargs)
       
    
                