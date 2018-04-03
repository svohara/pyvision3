"""
Demonstration of a video montage
"""
import pyvision3 as pv3

vid1 = pv3.Video(pv3.VID_PRIUS)
vid2 = pv3.Video(pv3.VID_PRIUS)

vid_dict = {"vid_1": vid1, "vid_2": vid2}

vid_montage = pv3.VideoMontage(vid_dict, layout=(1, 2), tile_size=(320, 240))
vid_montage.seek_to(200)
vid_montage.play(window_title="Video Montage", delay=20)
