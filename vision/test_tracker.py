#! /usr/bin/python
from tracker import Tracker, get_marker_object, get_markers
import time

# 1 (0.06, 0.08)
# 2 (1.08, 0.009)
# 3 ()
# 4 ()

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
    # Start Vision Thread
    thread1.start()

    for i in range(1000):
        # mid 9 marker on Thymio
        marker = get_marker_object(9)
        if marker is not None:
            print(marker.realxy())
