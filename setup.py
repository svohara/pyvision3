from setuptools import setup, find_packages
import os
import json

with open(os.path.join('pyvision3', 'data', 'project_info.json'), 'r') as infile:
    _info = json.load(infile)

_version = ".".join([str(x) for x in _info["version_tuple"]])

setup(
    name="pyvision3",
    version=_version,
    packages=find_packages(),
    url=_info["url"],
    license=_info["license"],
    author=_info["author"],
    author_email=_info["email"],
    description=_info["description"],
    package_data={
        # include all files found in the 'data' subdirectory
        # of the 'pyvision3' package
        'pyvision3': ['data/*']
    },
    test_suite="tests",
    download_url="{}/archive/{}.tar.gz".format(_info["url"], _version),
    keywords="images video vision",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        "matplotlib>=1.3.1",
        "numpy>=1.8.2",
        "pillow>=2.3.0",
        "shapely>=1.3.0"
    ],
    python_requires='>=3.4',
    long_description="""
    A python 3 computer vision library that complements OpenCV 3.x to add many useful features for developers
    and researchers alike. Pyvision provides utilities that help in these core areas: training data preparation,
    displaying results, and applying vision algorithms to video streams.

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

    * Pyvision Video class makes it easy to view and process common video files (avi, mpg, etc.), directories of images,
      or live streams from USB or IP cameras.
    * Video objects are iterators. ```for frame in vid: ...```
    * Create videos of your results, save a sequence of results as an avi file, for example. Simple and pythonic!
    * Apply background subtraction and motion detection to your video streams. Several common algorithms included
      or invent your own.
    * Video interface provides ability to buffer the video, to pause playback, to step through frame-by-frame,
      to resume playing with a certain per-frame delay, etc.
    * Video montage allows you to display multiple videos side-by-side in lockstep! And save the result
      as another video!
    * Register a call-back with a video object to allow you to process each frame with your own encapsulated code.
    * Create an image buffer from images or video
    * Treat image buffer as a 3D array of grayscale images, or display as a montage

    Pyvision3 is intended to be a successor to Pyvision that takes advantage of OpenCV 3.x and Python 3.x features.
    Huge thanks to David Bolme for being the originator of Pyvision. Many of the ease-of-use and interface ideas from
    the original Pyvision are carried forward, albeit with new implementations for Pyvision3.
    """,
)
