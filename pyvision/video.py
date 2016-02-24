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

import pyvision as pv3


class Video(object):
    """
    A Pyvision Video object makes using and interacting with video
    streams easier than 'raw' opencv code. A pyvision video object
    is an interable (i.e., for img in vid: ....), and also provides
    a high level "play" method, which plays-back the video, displays
    the frame number as an annotation, and supports a callback function
    to perform per-frame tasks.
    """

    def __init__(self, video_source):
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
        """
        self.source = video_source
        self.cap = cv2.VideoCapture(video_source)
        self.current_frame_num = 0
        self.current_frame = None

    def __del__(self):
        self.cap.release()

    def __iter__(self):
        return self

    def next(self):  # python 2 compatiblity
        return self.__next__()

    def __next__(self):
        """
        We wrap the read method of the video capture object for a few reasons.
        1. Adhere to python iterator interface
        2. Encapsulate some helpful error handling
        3. Maintain state variables: self.current_frame and self.current_fram_num
        
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
            return self.current_frame
        else:
            if self.current_frame_num == 0:
                # something is wrong with the video source
                raise ValueError("Error: Video source can't be read. VideoCapture retrieve failed.")
            else:
                raise StopIteration

    def play(self, window="Pyvision Video", pos=None, delay=20,
             annotate=True, image_buffer=None, start_frame=0, end_frame=None,
             on_new_frame=None, **kwargs):
        """
        Plays the video, calling the on_new_frame function after loading each
         frame from the video. The user may interrupt video playback by
         hitting (sometimes repeatedly) the spacebar, upon which they are
         given a text menu in the console to abort program, quit playback,
         continue playback, or step to the next frame.

        Parameters
        ----------
        window: string
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
            An optional pyvision ImageBuffer object to contain the
            most recent frames. This is useful if a buffer is required for background
            subtraction, for example. The buffer contents is directly modified each
            time a new image is captured from the video, and a reference to the buffer
            is passed to the on_new_frame function (defined below).
        start_frame: int
             If > 0, then the video will cue itself by quickly fast-forwarding
            to the desired start frame before any images are shown. During the cueing process,
            any on_new_frame function callbacks will NOT be activated.
        end_frame: int
             If not None, then the playback will end after this frame has
            been processed.
        on_new_frame: function or callable object
            A python callable object (function) with a signature of
            foo( pvImage, frame_num, key=None, buffer=None ), where key is
            the key pressed by the user (if any) during the pause_and_play interface, and
            buffer is a reference to the optional image buffer provided to the play method.
        kwargs:
            Optional keyword arguments that should be passed onto the on_new_frame function.

        Returns
        -------
        The final frame number of the video, or the frame number at which the user terminated
        playback using the 'q'uit option.
        """
        vid = self
        if delay == 0:
            delay_obj = {'wait_time': 20, 'current_state': 'PAUSED'}
        else:
            delay_obj = {'wait_time': delay, 'current_state': 'PLAYING'}
        key = ''
        for img in vid:
            if self.current_frame_num == 0 and start_frame > 0:
                print("Cueing video to start at {}".format(start_frame))

            if self.current_frame_num < start_frame:
                continue
            if end_frame is not None and self.current_frame_num > end_frame:
                break

            if image_buffer is not None:
                image_buffer.add(img)

            if annotate:
                txt = "Frame: {}".format(self.current_frame_num)
                img.annotate_text(txt, (10, 10), color=(255, 255, 255), bg_color=(0, 0, 0),
                      font_face=cv2.FONT_HERSHEY_PLAIN, font_scale=1)

            if window != None:
                img.show(window_title=window, highgui=True, pos=pos, delay=1, annotations_opacity=1.0)

            if on_new_frame != None:
                on_new_frame(img, self.current_frame_num, key=key,
                             image_buffer=image_buffer, **kwargs)

            key = self._pause_and_play(delay_obj)
            if key == 'q':
                break  # user selected quit playback

        if window:
            cv2.destroyWindow(window)

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

