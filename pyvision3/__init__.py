"""
This is the top-level namespace for the pyvision 3 library
"""
from .constants import *
from .pv_exceptions import *
from .geometry import Point, Rect, CenteredRect, in_bounds, integer_bounds, integer_coords_array
from .image import Image, matplot_fig_to_image
from .affine import AffineTransformer, AffineRotation, AffineTranslate
from .imagebuffer import ImageBuffer
from .video import VideoInterface, Video, VideoFromFileList, VideoFromDir, VideoFromImageStack
from .montage import ImageMontage, VideoMontage

from pyvision3.video_proc.backgroundsubtract import \
    FrameDifferenceModel, MedianModel, ApproximateMedianModel, AbstractBGModel, StaticModel, \
    BG_SUBTRACT_STATIC, BG_SUBTRACT_FRAME_DIFF, BG_SUBTRACT_MEDIAN, BG_SUBTRACT_APPROX_MEDIAN
from pyvision3.video_proc.motiondetection import \
    MotionDetector, MD_BOUNDING_RECTS, MD_STANDARDIZED_RECTS

from pyvision3.dataset_tools.crops import crop_regions, crop_negative_regions, random_rect_gen
from pyvision3.dataset_tools.tile_selection import TileSelector, \
    tiles_from_dir, tiles_from_files, tiles_from_vid
from pyvision3.dataset_tools.capture_clicks import CaptureClicks
from pyvision3.dataset_tools.capture_polygons import CapturePolygons

__version__ = ".".join([str(x) for x in VERSION_TUPLE])

