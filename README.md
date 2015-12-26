# pyvision3
A computer vision library that complements OpenCV 3.x to add many useful features for developers and scientists alike. Pyvision3 is intended to be a successor to Pyvision that takes advantage of OpenCV 3.x and Python 3.x features. Huge thanks to David Bolme for being the originator of Pyvision. Many of the ease-of-use and interface ideas from the original Pyvision are carried forward, albeit with new implementations for Pyvision3.

## Examples

```python
import pyvision as pv3

#load and display an image
img = pv3.Image("path/to/somefile.jpg")
img.show()

#annotate the image
img.annotate_text("Waldo was here", (254, 246), color=(0, 0, 255), bg_color=(255, 255, 255))
img.show()

#annotations are on a separate layer, original data is still there
img.show(annotations=False)

#load and play a video or webcam
vid = pv3.Video("path/to/somevideo.mov")
vid.play()  #supports pausing and single-frame advance

#or you can treat a video as an iterator
for img in vid:
	print(vid.current_frame_number)
	img.show()

```

## Main Contributors to Pyvision3
Stephen O'Hara

## Main Contributors to the original Pyvision
David Bolme
Stephen O'Hara

## Backwards Compatibility
Pyvision3 is not (necessarily) designed to be backwards compatible to earlier versions of Python and OpenCV. For those developing with Python 2.7, I suggest continuing to use the original Pyvision code, which can be found on github.
