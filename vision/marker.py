import cv2
import numpy as np
from math import pi, cos, sin, atan2, degrees, fabs, sqrt, radians
import calibration as cal
import inspect


class Marker:

    def __init__(self, mid, root, l1, l2, transform=None, height = -10):
        self.mid = mid
        self.root = root
        self.black_leaves = l1
        self.white_leaves = l2
        self.transform = transform
        self.height = height
        if(l1 is not None or l2 is not None):
            self.no_bl = len(l1)
            self.no_al = len(l1) + len(l2)
        self.cx, self.cy = None, None
        self.acx, self.acy = None, None
        self.rxy = None
        self.angle = None
        self.isMoving = False

    def center(self):
        if self.cx is None or self.cy is None:
            M = cv2.moments(self.root)
            self.cx = int(M['m10'] / M['m00'])
            self.cy = int(M['m01'] / M['m00'])

        return (self.cx, self.cy)

    def orientation(self):
        if self.angle == None:
            sum_all, sum_black = (0, 0), (0, 0)

            for l in self.black_leaves:
                ((cx, cy), size, angle) = cv2.minAreaRect(l)
                sum_all = (sum_all[0] + cx, sum_all[1] + cy)
                sum_black = (sum_black[0] + cx, sum_black[1] + cy)

            for l in self.white_leaves:
                ((cx, cy), size, angle) = cv2.minAreaRect(l)
                sum_all = (sum_all[0] + cx, sum_all[1] + cy)

            avg_all = (int(sum_all[0] / self.no_al),
                        int(sum_all[1] / self.no_al))

            avg_black = (int(sum_black[0] / self.no_bl),
                            int(sum_black[1] / self.no_bl))

            self.angle = Marker.angle_between_points(
                avg_all[0], avg_all[1], avg_black[0], avg_black[1])

            # Save alternative averaged centroid
            self.acx = avg_all[0]
            self.acy = avg_all[1]
            self.acx_black = avg_black[0]
            self.acy_black = avg_black[1]

        return self.angle

    def realxy(self):
        if self.rxy is None:
            if self.transform is not None:
                (cxs, cys) = self.center()
                self.rxy = np.dot(self.transform, [cxs, cys, 1.0])
                self.rxy /= self.rxy[2]
                self.rxy[2] = self.height

        return self.rxy

    def updateMarker(self, cx, cy, angle):
        self.cx = cx
        self.cy = cy
        self.angle = angle

    def getAlternativeAngle(self):
        return self.acx, self.acy, self.acx_black, self.acy_black

    @staticmethod
    def angle_trunc(a):
        while a < 0.0:
            a += pi * 2
        return a

    @staticmethod
    def angle_between_points(x_start, y_start, x_end, y_end):
        deltaY = y_end - y_start
        deltaX = x_end - x_start
        return Marker.angle_trunc(atan2(deltaY, deltaX))
