"""
Created on Dec 26, 2015
@author: sohara
"""

import pyvision as pv3

def play_webcam(cam_num=0):
    vid = pv3.Video(cam_num)
    vid.play(delay=0)  # start in a paused state

def play_sample_mov():
    vid = pv3.Video(pv3.VID_PRIUS)
    vid.play()

if __name__ == '__main__':
    print("=================================================================")
    print("Webcam video will start paused.")
    print("With focus on video window, use the keyboard to control playback.")
    print("During playback, with focus on window, use SPACEBAR to pause.")
    print("=================================================================")
    play_webcam()
