"""
Created: Oct 22, 2010
Modified: Mar 10, 2016 
    Modified for Pyvision 3
Author: Stephen O'Hara
"""
import numpy as np
import pyvision3 as pv3


class ImageBuffer:
    """
    Stores a limited number of images from a video (or any other source)
    Makes it easy to do N-frame-differencing, for example, by easily being
    able to get the current (middle) frame, plus the first and last frames of the
    buffer. With an ImageBuffer of size N, as images are added, eventually the
    buffer fills, and older items are dropped off the end. This is convenient
    for streaming input sources, as the user can simply keep adding images
    to this buffer, and internally, the most recent N will be kept available.
    """

    def __init__(self, N=5):
        """
        @param N: how many image frames to buffer
        """
        self._data = [None for _ in range(N)]
        self._count = 0
        self._max = N
            
    def __getitem__(self, key):
        return self._data[key]
        
    def __len__(self):
        """
        This is a fixed-sized ring buffer, so length is always the number
        of images that can be stored in the buffer (as initialized with Nframes)
        """
        return self._max
    
    def is_full(self):
        if self._count == self._max:
            return True
        else:
            return False
            
    def clear(self):
        self._data = [None for _ in range(self._max)]
        self._count = 0
            
    def get_count(self):
        """
        Note that get_count() differs from __len__() in that this method returns the number of
        image actually stored in the ImageBuffer, while __len__() returns the size of the buffer,
        defined as the number of images the buffer is allowed to store.
        """
        return self._count
    
    def get_data(self):
        return self._data

    def first(self):
        return self._data[0]

    def last(self):
        return self._data[-1]
    
    def middle(self):
        mid = int(self._count/2)
        return self._data[mid]
            
    def add(self, image):
        """
        add an image to the buffer, will kick out the oldest of the buffer is full
        @param  image: image to add to buffer
        """
        self._data.pop(0)  # remove last, if just beginning, this will be None
        self._data.append(image)
        self._count += 1
        if self._count > self._max:
            self._count = self._max
            
    def fill(self, source):
        """
        If buffer is empty, you can use this function to spool off the first
        N frames of a video or list of images to initialize/fill the buffer.
        @param source: Either a list of pv.Images, an instance of pv.Video or its
        subclass, or any other iterable that yields a pv.Image() when
        next() is called.
        @note: Will cause an assertion exception if buffer is already full.
        """
        assert not self.is_full()

        vid_flag = True if isinstance(source, pv3.VideoInterface) else False
        cur_pos = 0

        while not self.is_full():
            if vid_flag:
                im = source.next()
            else:
                im = source[cur_pos]
                cur_pos += 1
                cur_pos %= len(source)
            self.add(im)

        return

    def as_image_stack_BW(self, size=None):
        """
        Outputs an image buffer as a 3D numpy array ("stack") of grayscale images.
        @param size: A tuple (w,h) indicating the output size of each frame.
        If None, then the size of the first image in the buffer will be used.
        @return: a 3D array (stack) of the gray scale version of the images
        in the buffer. The dimensions of the stack are (N,w,h), where N is
        the number of images (buffer size), w and h are the width and height
        of each image.        
        """
        if size is None:
            img0 = self[0]        
            (w, h) = img0.size
        else:
            (w, h) = size
            
        f = self.get_count()
        stack = np.zeros((f, h, w), dtype='uint8')
        for i, img in enumerate(self._data):
            # if img is not (w,h) in size, then resize first
            sz = img.size
            if (w, h) != sz:
                img2 = img.resize((w, h))
                mat = img2.as_grayscale(as_type="CV")
            else:
                mat = img.as_grayscale(as_type="CV")
            stack[i, :, :] = mat
            
        return stack
    
    def as_montage(self, layout, tile_size=None, **kwargs):
        (w, h) = self[0].size
        if tile_size is None:
            tw = w//5
            th = h//5
            tw = 32 if tw < 32 else tw
            th = 24 if th < 24 else th
            tile_size = (tw, th)
            
        im = pv3.ImageMontage(self._data, layout=layout, tile_size=tile_size, **kwargs)
        return im
    
    def show(self, N=10, window_title="Image Buffer", pos=None, delay=0):
        """
        @param N: The number of images in the buffer to display at once
        @param window_title: The window name
        @param pos: The window position
        @param delay: The window display duration 
        """
        if self[0] is None:
            return
        
        if N <= self._count:
            im = self.as_montage(layout=(1, N))
        else:
            im = self.as_montage(layout=(1, self._count))
        im.show(window_title=window_title, pos=pos, delay=delay)
