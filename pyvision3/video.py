"""
Created on Dec 21, 2015
@author: Stephen O'Hara

This module defines the pyvision video class, which wraps
the opencv (cv2) functions in a class with convenient methods.

A few useful aspects of the video class:

1. Pause-and-play interface for playing back a video, pausing,
stepping frame-by-frame, resuming, quitting, etc., from a streamlined
interface wrapped around highgui

2. The same video interface is used to interact with actual video
files, rtsp streams, directories of images (played as a video)

3. Callback function support vid.on_new_frame() allows the
user to specify functions that operate on each new frame of the
video stream, which can be used to easily build real-time video
analysis pipelines.
"""
# The following prevents a bunch of pylint no-member errors
# with the cv2 module.
# pylint: disable=E1101

import cv2
import pyvision3 as pv3
import sys
import os
import glob


class VideoInterface(object):
    """
    The common functions required by any object implementing
    the pyvision3 Video interface, and implements the common pause-and-play
    feature of Videos.
    """
    def __init__(self, size=None):
        self.current_frame_num = 0
        self.current_frame = None
        self.size = size

        # Set the following to true when creating a subclass if it
        # supports random access to the frames without seeking. In which
        # case, you must also implement the __getitem__ magic method.
        self._random_access = False

    def __iter__(self):
        return self

    def seek_to(self, frame_num):
        """
        Set the video to the desired frame number and return the frame. Subsequent
        calls to next() will start from this position. If the Video class supports
        random access, this is very fast. Otherwise, a video may have to seek to
        the desired position, which is likely slow.

        Parameters
        ----------
        frame_num: int
            The frame number to seek/go to and return.

        Returns
        -------
        The pyvision3 image at the desired frame number.
        """
        if self._random_access:
            self.current_frame_num = frame_num
            self.current_frame = self[frame_num]
        else:
            if self.current_frame_num > frame_num:
                self.reset()

            if frame_num > 0:
                print("Seeking video to desired position...")
                sys.stdout.flush()
            while self.current_frame_num < frame_num:
                _ = next(self)

        return self._get_resized()

    def next(self):  # python 2 compatibility
        return self.__next__()

    def __next__(self):
        raise NotImplementedError

    def _get_resized(self):
        if self.size is None:
            return self.current_frame
        else:
            return self.current_frame.resize(self.size, keep_aspect=False, as_type="PV")

    def reset(self):
        """
        Reset the video to the start / reinitialize as required so that it can
        be iterated over again.
        """
        self.current_frame_num = 0
        self.current_frame = None

    def save(self, out_file, out_size=(640, 480), four_cc="MP4V", fps=15):
        """
        Saves the entire video to a movie file.
        Parameters
        ----------
        out_file: str
            The output video file name to save the video
        out_size: tuple (width, height)
            The output frame size for the video, which in many cases
            might be smaller than the input frames. Each frame will
            be resized appropriately, with letterboxing as required.
        four_cc: str
            The four_cc code used to encode the video, default
            is "MP4V"
        fps:    int
            Desired frames per second
        """
        self.reset()
        print("Saving video to: {}".format(out_file))
        vid_out = cv2.VideoWriter(filename=out_file,
                                  fourcc=cv2.VideoWriter_fourcc(*four_cc),
                                  fps=fps,
                                  frameSize=out_size)

        for frame in self:
            if frame.size != out_size:
                out_frame = frame.resize(out_size, as_type="PV", keep_aspect=True)
            else:
                out_frame = frame
            vid_out.write(out_frame.data)

        self.reset()
        print("Completed.")

    def play(self, window_title="Pyvision Video", pos=None, delay=20,
             annotate=True, image_buffer=None, start_frame=None, end_frame=None,
             on_new_frame=None, **kwargs):
        """
        Plays the video, calling the on_new_frame function after loading each
         frame from the video. The user may interrupt video playback by
         hitting (sometimes repeatedly) the spacebar, upon which they are
         given a text menu in the console to abort program, quit playback,
         continue playback, or step to the next frame.

        Parameters
        ----------
        window_title: string
            The window name used to display the video. If None,
            then the video won't be shown, but on_new_frame will be called at
            each frame.
        pos: A tuple (int: x, int: y)
            Defines where the output window should be located
            on the user's screen. None indicates default openCV positioning algorithm
            will be used.
        delay: int
            The delay in ms between window updates. This allows the user
            to control the playback frame rate. A value of 0 indicates that the video
            will wait for keyboard input prior to advancing to the next frame. This
            delay is used by the pauseAndPlay interface, so it will affect the rate
            at which on_new_frame is called as well.
        annotate: boolean
            If True, the image will be annotated with the frame number
            in the upper left corner. Set False for no frame number annotation.
        image_buffer: ImageBuffer
            An optional pyvision3 ImageBuffer object to contain the
            most recent frames. This is useful if a buffer is required for background
            subtraction, for example. The buffer contents is directly modified each
            time a new image is captured from the video, and a reference to the buffer
            is passed to the on_new_frame function (defined below).
        start_frame: int or None
             If >= 0, then the video will cue itself by seeking to the desired start frame before
             any images are shown. During the cueing process, any on_new_frame function callbacks
             will NOT be activated. If None, the video will play from whatever it's current
             position/state is.
        end_frame: int or None
             If not None, then the playback will end after this frame has
            been processed.
        on_new_frame: function or callable object
            A python callable object (function) with a signature of
            foo( pvImage, frame_num, key=None, image_buffer=None ), where key is
            the key pressed by the user (if any) during the pause_and_play interface, and
            buffer is a reference to the optional image buffer provided to the play method.
        kwargs:
            Optional keyword arguments that should be passed onto the on_new_frame function.

        Returns
        -------
        The final frame number of the video, or the frame number at which the user terminated
        playback using the 'q'uit option.
        """
        if delay == 0:
            delay_obj = {'wait_time': 20, 'current_state': 'PAUSED'}
        else:
            delay_obj = {'wait_time': delay, 'current_state': 'PLAYING'}
        key = ''

        if start_frame is not None:
            self.seek_to(start_frame)

        for img in self:

            if end_frame is not None and self.current_frame_num > end_frame:
                break

            if image_buffer is not None:
                image_buffer.add(img)

            if annotate:
                txt = "Frame: {}".format(self.current_frame_num)
                img.annotate_text(txt, (10, 10), color=(255, 255, 255), bg_color=(0, 0, 0),
                                  font_face=cv2.FONT_HERSHEY_PLAIN, font_scale=1)

            if window_title is not None:
                img.show(window_title=window_title, highgui=True, pos=pos, delay=None, annotations_opacity=1.0)

            if on_new_frame is not None:
                on_new_frame(img, self.current_frame_num, key=key,
                             image_buffer=image_buffer, **kwargs)

            key = self._pause_and_play(delay_obj)
            if key == 'q':
                break  # user selected quit playback

        if window_title:
            cv2.destroyWindow(window_title)

        return self.current_frame_num

    def _pause_and_play(self, delay_obj={'wait_time': 20, 'current_state': 'PLAYING'}):
        """
        This function is intended to be used in the play back loop of a video.
        It allows the user to interrupt the play back to pause the video, to
        step through it one frame at a time, and to register other keys/commands
        that the user may select.
        @param delay_obj: The "delay object", which is just a dictionary that
        specifies the wait_time (the delay in ms between frames), and
        the current_state of either 'PLAYING' or 'PAUSED'

        NOTE: We are intentionally using a mutable default argument in this
        function.
        """
        state = delay_obj['current_state']
        wait = delay_obj['wait_time']
        # print state, wait

        menu_str = "Select <a>bort program, <q>uit playback, <c>ontinue playback,"
        menu_str += " or <s>tep to next frame."

        if state == "PAUSED":
            print("PAUSED: {}".format(menu_str))
            wait = 0

        c = cv2.waitKey(wait)
        c &= 127  # bit mask to get only lower 8 bits

        # sometimes a person has to hold down the spacebar to get the input
        # recognized by the cv.WaitKey() within the short time limit. So
        # we need to 'soak up' these extra inputs when the user is still
        # holding the space bar, but we've gotten into the pause state.
        while c == ord(' '):
            print("PAUSED: {}".format(menu_str))
            c = cv2.waitKey(0)
            c &= 127  # bit mask to get only lower 8 bits

        # At this point, we have a non-spacebar input, so process it.
        if c == ord('a'):  # abort
            print("User Aborted Program.")
            raise SystemExit
        elif c == ord('q'):  # quit video playback
            return 'q'
        elif c == ord('c'):  # continue video playback
            delay_obj['current_state'] = "PLAYING"
            return 'c'
        elif c == ord('s'):  # step to next frame, keep in paused state
            delay_obj['current_state'] = "PAUSED"
            return 's'
        else:  # any other keyboard input is just returned
            # delay_obj['current_state'] = "PAUSED"
            return chr(c)


class Video(VideoInterface):
    """
    A Pyvision Video object makes using and interacting with video
    streams easier than 'raw' opencv code. A pyvision3 video object
    is an interable (i.e., for img in vid: ....), and also provides
    a high level "play" method, which plays-back the video, displays
    the frame number as an annotation, and supports a callback function
    to perform per-frame tasks.
    """

    def __init__(self, video_source, size=None):
        """
        Constructor.
        Input is the video source, which is anything that cv2.VideoCapture
        can take, such as a video file, a webcam number, or an rtsp stream
        URI.

        Parameters
        ----------
        video_source:   variable
            The video source may be the path of a video file, a webcam
            number, an rtsp stream URI, or anything else accepted by the
            cv2.VideoCapture object.
        size: tuple (w,h)
            Optional. Used to specify the size of the output. This will
            force each frame of the video source to be resized appropriately.
            Specify None to return the native size of the video source.
        """
        super().__init__(size=size)
        self.source = video_source
        self.cap = cv2.VideoCapture(video_source)

    def __del__(self):
        if self.cap is not None:
            self.cap.release()

    def reset(self):
        VideoInterface.reset(self)
        self.cap.release()
        self.cap = cv2.VideoCapture(self.source)

    def __next__(self):
        """
        We wrap the read method of the video capture object for a few reasons.
        1. Adhere to python iterator interface
        2. Encapsulate some helpful error handling
        3. Maintain state variables: self.current_frame and self.current_frame_num
        
        Example usage
        -------------
        vid = pv3.Video("path/to/some/video.avi")
        for img in vid:
            print(vid.current_frame_num)
            img.show(highgui=True, delay=25)        
        """
        if self.cap.isOpened():
            (ok_flag, img) = self.cap.read()
        else:
            raise ValueError("Error: VideoCapture object has been closed.")

        if ok_flag:
            self.current_frame_num += 1
            self.current_frame = pv3.Image(img)
        else:
            if self.current_frame_num == 0:
                # something is wrong with the video source
                raise ValueError("Error: Video source can't be read. VideoCapture retrieve failed.")
            else:
                raise StopIteration

        return self._get_resized()


class VideoFromFileList(VideoInterface):
    """
    Given a sorted list of filenames (including full path), this will
    treat the list as a video sequence.
    """
    def __init__(self, filelist, size=None):
        """
        Parameters
        ----------
        filelist: list[str]
            a list of full file paths to the images that comprise the video.
            They must be files capable of being loaded into a pv.Image() object, and should
            be in sorted order for playback.
        size: tuple (w,h)
            Optional tuple to indicate the desired playback window size.
        """
        super().__init__(size=size)
        self.filelist = filelist
        self.num_frames = len(filelist)
        self._random_access = True

    def __getitem__(self, frame_num):
        frame = self.filelist[frame_num]
        try:
            img = pv3.Image(frame)
        except AttributeError:
            raise pv3.InvalidImageFile("Image file is not valid: {}".format(frame))
        return img

    def __next__(self):
        """
        For iterating the frames in the video sequence
        """
        if self.current_frame_num >= self.num_frames:
            raise StopIteration

        frame = self[self.current_frame_num]
        self.current_frame = frame
        self.current_frame_num += 1

        return self._get_resized()


class VideoFromDir(VideoFromFileList):
    """
    Consider the sorted files in a given directory as a video.
    This is a convenience class that uses VideoFromFileList
    """
    def __init__(self, directory, pattern="*", size=None):
        """

        Parameters
        ----------
        directory:  the directory where the image files are stored
        pattern:    the file pattern to include, default is "*", meaning all files
                    in the directory. If your directory contains non-image files
                    as well as the images, use this to filter the input files
                    for example, "*.jpg". This pattern will be given to the glob function
                    as sorted(glob.glob(os.path.join(directory, pattern)))
        size:       The output video frame size
        """
        assert os.path.isdir(directory)
        self.directory = directory
        self.pattern = pattern
        file_list = sorted(glob.glob(os.path.join(directory, pattern)))
        super().__init__(filelist=file_list, size=size)


class VideoFromImageStack(VideoInterface):
    """
    This class allows the user to treat a stack of grayscale images in a 3D numpy array as a video.
    We assume that the dimensions of the array are ordered as (frame #, width, height)
    """
    def __init__(self, image_stack, size=None):
        """
        Parameters
        ----------
        image_stack: numpy ndarray (frames, width, height)
            A numpy 3D ndarray that represents the image stack. Should be of dimensions (frames,width,height).
            Each image in the stack is single-channel (gray), so slicing stack[idx, :, :] gets the gray-scale
            image ndarray at position idx.
        size: tuple (w,h)
            the optional width,height to resize the input frames
        """
        super().__init__(size=size)
        self.image_stack = image_stack
        self.num_frames = image_stack.shape[0]
        self._random_access = True

    def __getitem__(self, frame_num):
        frame = self.image_stack[frame_num, :, :]
        return pv3.Image(frame)

    def __next__(self):
        """
        For iterating the frames in the video sequence
        """
        if self.current_frame_num >= self.num_frames:
            raise StopIteration

        frame = self.image_stack[self.current_frame_num, :, :]
        self.current_frame = pv3.Image(frame)
        self.current_frame_num += 1

        return self._get_resized()
