#! /usr/bin/python
from vision.Tracker import get_marker_object, Tracker, get_markers
import vision.marker

import time

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "anfv"
__date__ = "$16-Nov-2016 19:16:33$"

if __name__ == "__main__":
    mid = 5
    transform = None
    mid_aux = 0
    video_source = 1
    capture = False
    show = True
    debug = False
    thread1 = Tracker(mid, transform, mid_aux,
                      video_source, capture, show, debug)
    # Start new Threads
    thread1.start()

    for i in range(100):
        time.sleep(1)
        marker = get_marker_object(9)
        if marker is not None:
            print(marker.orientation())
