#! /usr/bin/python
from Tracker import get_marker_object, Tracker, get_markers
import marker

import time

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "anfv"
__date__ = "$16-Nov-2016 19:16:33$"

if __name__ == "__main__":
    print("Hello World")
    mid= 5
    transform=None
    mid_aux = 0
    video_source = 0
    capture=False
    show=True
    debug = True
    thread1 = Tracker(mid, transform, mid_aux , video_source, capture, show, debug)
    # Start new Threads
    thread1.start()

    while(True):
        continue
    # for i in range(100):
    #     time.sleep(1)
    #     marker = get_marker_object(5)
    #     if marker is not None:
    #         print(marker.orientation())
