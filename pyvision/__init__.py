"""
This is the top-level namespace for the pyvision 3 library
"""
from .constants import *
from .pv_exceptions import *
from .geometry import Point, Rect, CenteredRect, in_bounds, integer_bounds, integer_coords_array
from .image import Image
from .imagebuffer import ImageBuffer
from .montage import ImageMontage

from .video import VideoInterface, Video, VideoFromFileList, VideoFromImageStack
from pyvision.video_proc.backgroundsubtract import \
    FrameDifferenceModel, MedianModel, ApproximateMedianModel, AbstractBGModel, StaticModel, \
    BG_SUBTRACT_STATIC, BG_SUBTRACT_FRAME_DIFF, BG_SUBTRACT_MEDIAN, BG_SUBTRACT_APPROX_MEDIAN
from pyvision.video_proc.motiondetection import \
    MotionDetector, MD_BOUNDING_RECTS, MD_STANDARDIZED_RECTS

from pyvision.dataset_tools.crops import crop_regions, crop_negative_regions, random_rect_gen
from pyvision.dataset_tools.tile_selection import TileSelector, \
    tiles_from_dir, tiles_from_files, tiles_from_vid

