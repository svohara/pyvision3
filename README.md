# pyvision3
A python3 computer vision library that complements OpenCV 3.x to add many useful features for developers and researchers alike.
Pyvision provides utilities that help in these core areas: training data preparation, displaying results, and applying vision
algorithms to video streams.

Training data preparation:
* Apply affine transformations
* Label points and polygons in images
* Generate defined or random crops in images
* Select good vs. bad tiles in a montage display

Displaying results:
* Easy and powerful image annotations of shapes, lines, text, and overlay masks
* Compatible with shapely -- annotate shapely polygons to images
* Annotations are kept on a separate layer from the underlying image
* Display the image and save images with or without annotations
* Show results in an "image montage" -- multiple images or tiles shown in a grid
* Use image montage to select results (or to provide manual QA on the results)
* Play back sequences of images like a video, with pause-and-play interface

Video processing:
* Pyvision Video class makes it easy to view and process common video files (avi, mpg, etc.), directories of images, or
live streams from USB or IP cameras.
* Video objects are iterators. ```for frame in vid: ...```
* Create videos of your results, save a sequence of results as an avi file, for example. Simple and pythonic!
* Apply background subtraction and motion detection to your video streams. Several common algorithms included or invent your own.
* Video interface provides ability to buffer the video, to pause playback, to step through frame-by-frame, to resume playing
with a certain per-frame delay, etc.
* Video montage allows you to display multiple videos side-by-side in lockstep! And save the result as another video!
* Register a call-back with a video object to allow you to process each frame with your own encapsulated code.
* Create an image buffer from images or video
* Treat image buffer as a 3D array of grayscale images, or display as a montage

Pyvision3 is intended to be a successor to Pyvision that takes advantage of OpenCV 3.x and Python 3.x features. 
Huge thanks to David Bolme for being the originator of Pyvision. 
Many of the ease-of-use and interface ideas from the original Pyvision are carried forward, albeit with new implementations for Pyvision3.

## Installation
Install prerequisites that are not pip-installable:
* python >= 3.4
* opencv >= 3.0 with bindings for python

Then install using pip:

```pip install pyvision```

You may also clone the repo and install in development mode. After cloning, change to the project directory and install via:

```pip install -e .```

## Examples
Please view the "examples" directory of the repository for several helpful examples.

The following are some simple code samples to give you the flavor of the library.


```python

import pyvision3 as pv3

#load and display an image
img = pv3.Image("path/to/somefile.jpg")
img.show()

#annotate the image
img.annotate_text("Waldo was here", (254, 246), color=(0, 0, 255), bg_color=(255, 255, 255))
img.show()  # by default uses an opencv highgui window for display
img.imshow()  # will use a matplotlib figure, optional arguments are the same as img.show()

#annotations are on a separate layer, original data is still there
img.show(annotations=False)

#load and play a video or webcam
vid = pv3.Video("path/to/somevideo.mov")
vid.play()  #supports pausing and single-frame advance

#We can control the position of playback in the video
vid.play(start_frame=150, end_frame=400)

#or you can treat a video as an iterator
for img in vid:
	print(vid.current_frame_number)
	img.show(delay=25) #25ms delay between frames


#motion detection
md = pv3.MotionDetector(method=pv3.BG_SUBTRACT_FRAME_DIFF, buff_size=5, thresh=80)
for img in vid:
    md.detect(img)
    out = md.annotate_frame()
    if out is not None:
        out.show(delay=25)
```

## Main Contributors to Pyvision3
Stephen O'Hara

## Main Contributors to the original Pyvision
1. David Bolme
2. Stephen O'Hara

## Backwards Compatibility
Pyvision3 is not (necessarily) designed to be backwards compatible to earlier versions of Python and OpenCV. For those developing with Python 2.7, I suggest continuing to use the original Pyvision code, which can be found on github.
