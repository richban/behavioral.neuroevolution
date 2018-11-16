#! /usr/bin/python

# TrackerFai module using multi threading
import threading
from copy import deepcopy
from marker import Marker
import calibration
import cv2
import numpy as np
from math import pi, cos, sin, atan2, degrees, fabs, sqrt, radians
import calibration as cal
import inspect
import time
import os
from collections import deque
import math


lock = threading.Lock()
safeMarkers = []
currentMarkers = []
performCalibration = False

# Default marker trees
def_markers = {
    '0122222212212121111': 1,
    '0122212221222121111': 2,
    '0122212221221221111': 3,
    '0122212221212121111': 4,
    '0122222212221211111': 5,
    '0122221221221221111': 6,
    '0122212212212211111': 7,
    '0122221222122121111': 8,
    '0122222222212111111': 9,
    '0122221221212121111': 10,
    '0122222122221211111': 11,
    '0122221222122211111': 12
}

# Small marker trees
small_markers = {
    '0122222211111': 1,
    '0122222221111': 2,
    '0122222121111': 3,
    '0122212211111': 4,
    '0122212121111': 5,
    '0122221211111': 6,
    '0122122121111': 7,
    '0122122111111': 8,
    '0122121211111': 9,
    '0122221221111': 10,
    '0121212121111': 11,
    '0122212221111': 12
}

marker_trees = small_markers

# Camera intrinsics @1280x720 Evobliss

#RMS=0.35
cmat = np.array([[1069.393562444551, 0.0, 594.979364987246],
                [0.0, 1069.2071530333799, 427.54396736819604],
                [0.0, 0.0, 1.0]])

cdist = np.array( [[-0.008223905370418273],
                    [-0.5684443990011243],
                    [0.009909930203826769],
                    [0.00044666252383880796],
                    [1.4732231718761766]])


def calibrate():
    global performCalibration
    performCalibration = True
    while(performCalibration == True):
        time.sleep(0.1)
    print("Calibration finished!")
    time.sleep(1)


def get_markers():
    lock.acquire()
    markers = safeMarkers
    lock.release()
    return markers

def get_marker_object(mid, ur5 = None):
    idx = -1
    marker = None
    markers = get_markers()
    while marker is None:
        for i, m in enumerate(markers):
            if mid == m.mid:
                return m
        if(ur5 is not None):
            if(ur5.at_home == False):
                ur5.home()
        time.sleep(0.1)
        print("Marker " + str(mid) + " not found. Waiting...")
        markers = get_markers()

def get_marker_object_fast(mid, ur5 = None):
    global currentMarkers
    idx = -1
    marker = None
    #lock.acquire()
    markers = currentMarkers
    #lock.release()
    while marker is None:
        for i, m in enumerate(markers):
            if mid == m.mid:
                return m
        if(ur5 is not None):
            if(ur5.at_home == False):
                ur5.home()
        time.sleep(0.1)
        print("Marker " + str(mid) + " not found in fast markers. Waiting...")
        #lock.acquire()
        markers = currentMarkers
        #lock.release()



def euclidian_distance(p1, p2):
    dist = sqrt(pow(fabs(p1[0]-p2[0]),2)+pow(fabs(p1[1]-p2[1]),2))
    return dist

def inscribed_circle(p1, p2, p3):
    a = euclidian_distance(p1, p2)
    b = euclidian_distance(p1, p3)
    c = euclidian_distance(p2, p3)
    p = a+b+c
    k = 0.5*p
    r = (sqrt(k*(k-a)*(k-b)*(k-c)))/k
    ox = (a*p3[0]+b*p2[0]+c*p1[0])/p
    oy = (a*p3[1]+b*p2[1]+c*p1[1])/p
    return (r, (int(ox), int(oy)))

def get_vacant_position(markers, radius, pri_moves=None, future_pos=None):
    # TODO: the ordering is hardwired to the current corner marker layout
    # and should somehow be made general or configurable
    cnt_order = [markers[0], markers[2], markers[3], markers[1]]
    corners = np.array([[int(m.center()[0]), int(m.center()[1])] for m in cnt_order])
    print(corners)
    rect = cv2.boundingRect(corners)
    subdiv = cv2.Subdiv2D(rect)
    for m in markers:
        if pri_moves is not None and future_pos is not None and m.mid in pri_moves:
            real_xy = future_pos[m.mid][0]
            screen_xy = np.dot( np.linalg.inv(m.transform), [real_xy[0], real_xy[1], 1])
            #TODO: Normalize screen_xy
            subdiv.insert((int(screen_xy[0]), int(screen_xy[1])))
        else:
            subdiv.insert(m.center())
    triangleList = subdiv.getTriangleList()
    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if cv2.pointPolygonTest(corners, pt1, False) >= 0 and cv2.pointPolygonTest(
                corners, pt2, False) >= 0 and cv2.pointPolygonTest(corners, pt3, False) >= 0:
            r, incenter = inscribed_circle((t[0], t[1]), (t[2], t[3]), (t[4], t[5]))
            if r >= radius:
                # return pixel and robot coordinates of vacant position
                return (incenter, np.dot(markers[0].transform, [incenter[0], incenter[1], 1.0]))
    return None

_counter = 0

class Tracker(threading.Thread):


    def __init__(self, mid, transform, mid_aux = 0, video_source=1, capture=True, show=False, debug=False):
        threading.Thread.__init__(self)
        self.mid = mid
        self.mid_aux = mid_aux
        self.source = video_source
        self.transform = transform
        self.capture = capture
        self.show = show
        self.fps = 0
        self.lastMarkers=deque([])
        self.filterLen=8
        self.debug=debug
        self.height = -10
        self.originCalibrationMarkers = []

        cv2.setUseOptimized(True)


        print("starting Tracker, video source: ", self.source)
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(3, 1920)#1920
        self.cap.set(4, 1080)#1080
        self.cap.set(37, 1) # turn the autofocus off, save and comment
        self.cap.set(cv2.CAP_PROP_FPS, 15)
#        for i in range(30):
        self.cap.read()

    def stop(self):
        self.cap.release()

    def _get_marker_object(self, mid, markers):
        idx = -1
        marker = None
        for i, m in enumerate(markers):
            if mid == m.mid:
                idx = i
                marker = m
                break # return first, similar to index()
        return (idx, marker)

    def _getMarkers(self):
        import pdb; pdb.set_trace()
        global safeMarkers
        global image
        global currentMarkers
        ret, frame = self.cap.read()
        if self.debug == True:
            global _counter
            photos=0
            for file in os.listdir("../images/images_cal/markers/"):
                if file.endswith(".jpg"):
                    #print(file)
                    photos+=1
            name = '../images/images_cal/markers/' + str( _counter%photos) + '.jpg'
            print("processing image ", name)
            _counter += 1
            time.sleep(1)
            cv2.waitKey(0)
            frame = cv2.imread(name,cv2.IMREAD_COLOR)


        #undist = cv2.remap(frame, self.mapx, self.mapy, cv2.INTER_LINEAR)
        undist = cv2.undistort(frame, self.cmat2, cdist)
        #undist = cv2.pyrDown(undist)
        thresh = self.preprocess_image(undist)
        image =  undist#cv2.cvtColor(thresh, cv2.COLOR_RGB2GRAY)
        if self.capture:
            cv2.imwrite('raw.jpg', frame)
            cv2.imwrite('undist.jpg', undist)
            cv2.imwrite('thresh.jpg', thresh)

        cv2.imshow('blackandwhite', thresh)
        img2, contours, hierarchy = self.segmentation(thresh)
        markers = self.find_markers(contours, hierarchy, 13, self.transform)
#        for m in markers:
#            print "Marker ", m.mid, " found"

        markers.sort(key=lambda m: m.mid)

        currentMarkersTmp = deepcopy(markers)
        self.handleDoubleMarkers(currentMarkersTmp)
        lock.acquire()
        currentMarkers = currentMarkersTmp
        lock.release()

        if self.show:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for m in markers:
                cv2.circle(image, m.center(), 50, (255, 255, 255), 3)
                cx, cy = m.center()
                cv2.putText(image, str(m.mid), (cx + 25, cy + 25), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, str(round(self.fps,2)), (10, 25), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

            for m in safeMarkers:
                cx, cy = m.center()
                center = (int(round(cx)), int(round(cy)))
                if m.isMoving == False:
                    cv2.circle(image, center, 60, (255, 0, 0), 3)
                else:
                    cv2.circle(image, center, 60, (255, 0, 0), 7)
                angle = m.orientation()
                l = 40
                pt2 = (int(round(cx+l*math.cos(angle))), int(round(cy + l*math.sin(angle))))
                cv2.line(image, center, pt2, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(m.mid), (int(round(cx)) + 45, int(round(cy)) + 45), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

            idx, m = self._get_marker_object(5, currentMarkersTmp)
            if m is not None:
                cx, cy = m.center()
                center = (int(round(cx)), int(round(cy)))
                cv2.circle(image, center, 60, (0, 0, 255), 3)
                angle = m.orientation()
                l = 40
                pt2 = (int(round(cx+l*math.cos(angle))), int(round(cy + l*math.sin(angle))))
                cv2.line(image, center, pt2, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, str(m.mid), (int(round(cx)) + 45, int(round(cy)) + 45), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            for m in self.originCalibrationMarkers:
                x, y = m.center()
                cv2.circle(image, (int(x), int(y)), 4, (255, 0, 255), -1)

            cv2.imshow('capture', image)
            #cv2.imshow('thresh', thresh)
            cv2.waitKey(1)
        if self.capture:
            cv2.imwrite('foundMarkers.jpg', thresh)
        return markers
    def run(self):
        global lock
        global safeMarkers
        global performCalibration


        #Find the undistorsion parameters
        self.cmat2, self.roi = cv2.getOptimalNewCameraMatrix(cmat, cdist, (1920,1080), 0, (1920,1080))
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.cmat2, cdist, None, self.cmat2, (1920,1080), 5)

        #Fill the markers array
        for i in range(self.filterLen):
            m = self._getMarkers()
            self.lastMarkers.append(m)
        print("Last markers len: " + str(len(self.lastMarkers)))


        while(True):

            start = time.time()
            m = self._getMarkers()
            #Add the last markers and remove the first one
            self.lastMarkers.append(m)
            self.lastMarkers.popleft()

            #filter the markers
            filteredMarkers = self.filterMarkers()

            lock.acquire()
            safeMarkers = filteredMarkers
            lock.release()
            end = time.time()
            self.fps = self.fps * 9/10 + 1/(10*(end-start))

            if (self.transform is None and self.areCornersDetected(filteredMarkers) == True):
                calibration.redo_transform(filteredMarkers)
                self.transform, _, self.height = calibration.restore()
                for i in range (4):
                    _, mTmp = self._get_marker_object(i+1, filteredMarkers)
                    self.originCalibrationMarkers.append(mTmp)

            if(performCalibration == True):
                print("Trying to calibrate")
                if self.areCornerMarkersWellPlaced(filteredMarkers) == True:
                    calibration.redo_transform(filteredMarkers)
                    self.transform, _, self.height = calibration.restore()
                    self.originCalibrationMarkers=[]
                    for i in range (4):
                        _, mTmp = self._get_marker_object(i+1, filteredMarkers)
                        self.originCalibrationMarkers.append(mTmp)
                    performCalibration = False


    def filterMarkers(self):

        tempM=[None]*13
        filteredMarkers=[]
        for i in range(len(self.lastMarkers)):
            for m in self.lastMarkers[i]:
                if m.mid < 13:
                    if tempM[m.mid] == None:
                        tempM[m.mid]=[]
                    tempM[m.mid].append(m)

        for i in range(13):
            if tempM[i] is not None:
                cxA = np.zeros(len(tempM[i]))
                cyA = np.zeros(len(tempM[i]))
                angleA1 = np.zeros(len(tempM[i]))
                angleA2 = np.zeros(len(tempM[i]))
                angleA3 = np.zeros(len(tempM[i]))
                angleA4 = np.zeros(len(tempM[i]))

                #Get the coordinates and the angle and store them in np arrays
                for j in range(len(tempM[i])):
                    cx, cy = tempM[i][j].center()
                    cxA[j] = cx
                    cyA[j] = cy
                    orient = tempM[i][j].orientation()
                    a1, a2, a3, a4 = tempM[i][j].getAlternativeAngle()
                    angleA1[j] = a1
                    angleA2[j] = a2
                    angleA3[j] = a3
                    angleA4[j] = a4
#                cxM = np.mean(cxA)
#                cyM = np.mean(cyA)
#                angle1M = np.mean(angleA1)
#                angle2M = np.mean(angleA2)
#                angle3M = np.mean(angleA3)
#                angle4M = np.mean(angleA4)

                #Detect outliers
                data = [cxA, cyA, angleA1, angleA2, angleA3, angleA4]
                outliers = self.calculateOutliers(data)

                #Remove outliers
                cxA = np.delete(cxA, outliers)
                cyA = np.delete(cyA, outliers)
                angleA1 = np.delete(angleA1, outliers)
                angleA2 = np.delete(angleA2, outliers)
                angleA3 = np.delete(angleA3, outliers)
                angleA4 = np.delete(angleA4, outliers)

                #Create the new Markers
                if(len(cxA) > 0):
                    m = Marker(i, None, None, None, self.transform, self.height)
                    a1M = np.mean(angleA1)
                    a2M = np.mean(angleA2)
                    a3M = np.mean(angleA3)
                    a4M = np.mean(angleA4)
                    newAngle = Marker.angle_between_points(a1M, a2M, a3M, a4M)
                    m.updateMarker(np.mean(cxA), np.mean(cyA), newAngle)

                    #detect movement
                    z = len(angleA1)-1
                    if((abs(angleA1[z] - angleA1[0]) > 3)  or (abs(angleA2[z] - angleA2[0]) > 3) or (abs(angleA3[z] - angleA3[0]) > 3)  or (abs(angleA4[z] - angleA4[0]) > 3)):
                        m.isMoving = True

                    filteredMarkers.append(m)



#                if i==1:
#                    print str(len(cxA)) + " " + str(len(outliers))
#                    print str(round(cxM,2)) + "  " +  str(round(cyM,2)) + "  " +  str(round(angleM,2))+ " == " + str(round(np.mean(cxA),2)) + " " + str(round(np.mean(cyA),2)) + " " + str(round(np.mean(angleA),2)) + " == " + str(len(outliers)/float(len(tempM[i]))*100 )

        self.handleDoubleMarkers(filteredMarkers)
        return filteredMarkers


    def handleDoubleMarkers(self, markers):
        ###Hack to get the correct pos of the robot
        index5, m5 = self._get_marker_object(5, markers)
        if(m5 is not None):
            index6, m6 = self._get_marker_object(6, markers)
            if(m6 is not None):
                cx5, cy5 = m5.center()
                cx6, cy6 = m6.center()
                angle5 = m5.orientation()
                angle6 = m6.orientation()
                newAngle = Marker.angle_between_points(cx5, cy5, cx6, cy6)
                m5.updateMarker((cx5+cx6)/2, (cy5+cy6)/2, newAngle)
            else:
                #If we find only the marker 5 we cannot calcualte the position of the robot
                markers.pop(index5)

    def calculateOutliers(self, data):
        median = np.median(data, axis=1)
        iL = np.percentile(data, 25, axis=1)
        iH = np.percentile(data, 75, axis=1)

        outlierA=[]
        for i in range(len(data[0])):
            outlier=False
            for j in range(3):
                err = abs(data[j][i]-median[j])
                if(err > 1.5*(iH[j]-iL[j])):
                    outlier=True
            if(outlier):
                outlierA.append(i)
        return outlierA



    def preprocess_image(self, img, blur=False):
    #    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #    if blur:
    #        blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #        ret, th = cv2.threshold(blur, 0, 255,
    #                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #    else:
    #        ret, th = cv2.threshold(gray, 0, 255,
    #                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #    img = cv2.medianBlur(img,3)
        img = cv2.bilateralFilter(img,5,100,100)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        #th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #        cv2.THRESH_BINARY,13,3)
        th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY ,23,3)
        kernel = np.ones((3,3),np.uint8)
        th = cv2.erode(th,kernel,iterations = 1)
        kernel = np.ones((2,2),np.uint8)
        th = cv2.dilate(th,kernel,iterations = 1)
        return th


    def segmentation(self, img):
        return cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    def find_markers(self, contours, hierarchy, size, transform=None):
        """Find and id markers in image contours

        Keyword arguments:
        contours  -- contours found in image using cv2.findContours
        hierarchy -- the contour hierarchy. Requires mode CV_RETR_TREE
        size      -- the fixed no. of tree nodes in a marker tree
        """
        tree = hierarchy[0]
        tsize = len(tree)
        matches = []
        for n in range(0, tsize):
            nxt, pre, chd, par = tree[n]
            if chd != -1:
                if n + size <= tsize:
                    (found, m) = self.id_marker(contours, tree, n, n + size)
                    if found:
                        m.transform = transform
                        matches.append(m)
        return matches

    def id_marker(self, contours, tree, start, end):
        """Return a Marker object if found in subtree

        Keyword arguments:
        contours -- contours found in image using cv2.findContours
        tree     -- hierarchy array (ie. hierarchy[0])
        start    -- start index
        end      -- end index
        """
        global image
        leafMaxRadius = 8
        markerMaxRadius = 45
        markerMinRadius = 25
        seq = '0'
        sub, t = [], []
        l1, l2 = [], []
        depth = 1
        parent = [start]
        for n in range(start + 1, end):
            if len(contours[n]) <= 2:
                return (False, None)
            nxt, pre, chd, par = tree[n]
            t.append(depth)
            if depth == 2:
                l2.append(contours[n])
            # Parent since child is next node - increment depth and put parent on stack
            if chd == n + 1:
                depth = depth + 1
                if depth > 2:
                    return (False, None)  # Immediately return on invalid depth
                parent.append(n)
            # Lonely node on depth one - singleton leave
            elif depth == 1 and chd == -1:
                sub.append(t)
                t = []
                l1.append(contours[n])
            # Last child of the last parent on stack - decrement depth
            # and start new sub-sequence
            elif nxt == -1 and par == parent[-1:][0]:
                depth = depth - 1
                sub.append(t)
                t = []
                parent.pop()
        # Process sub-sequences -
        # 1. sort by length in reverse for left-heavy ordering
        # 2. flatten sub-sequences
        # 3. join to string and append to seq
        if len(sub) > 4:
            sub.sort(key=len, reverse=True)
            join = [n for s in sub for n in s]
            seq = seq + reduce(lambda x, y: x + y, map(str, join))
            if marker_trees.has_key(seq):
                m = Marker(marker_trees[seq], contours[start], l1, l2, self.transform, self.height)

                #Check ouside countour
                font = cv2.FONT_HERSHEY_SIMPLEX
                (xC,yC),radiusC = cv2.minEnclosingCircle(contours[start])
                outRadius = radiusC
                outX = xC
                outY = yC
                center = (int(xC),int(yC))
                radiusC = int(radiusC)
                #cv2.circle(image,center,radiusC,(255,0,255),1)
                #cv2.putText(image, str(radiusC), center, font, 1, (255,0,255), 2, cv2.LINE_AA)
                #Discard small ones and large ones
                if((radiusC > markerMaxRadius) or (radiusC < markerMinRadius)):
                    return (False, None)
#                if(m.mid == 5 ):
#                    print "Radio: ", radiusC


                #Check leaves l1 and l2
                for leaf in range(len(l1)):
                    (xC,yC),radiusC = cv2.minEnclosingCircle(l1[leaf])
                    dist = math.sqrt ( (outX-xC)*(outX-xC) + (outY-yC)*(outY-yC))
                    if((radiusC > leafMaxRadius) or dist > outRadius):
                        return (False, None)
                    center = (int(xC),int(yC))
                    radiusC = int(radiusC)
                    cv2.circle(image,center,radiusC,(0,0,255),-1)
                    #cv2.putText(image, str(radiusC), center, font, 1, (0,0,255), 2, cv2.LINE_AA)

                for leaf in range(len(l2)):
                    (xC,yC),radiusC = cv2.minEnclosingCircle(l2[leaf])
                    dist = math.sqrt ( (outX-xC)*(outX-xC) + (outY-yC)*(outY-yC))
                    if((radiusC > leafMaxRadius) or dist > outRadius):
                        return (False, None)
                    center = (int(xC),int(yC))
                    radiusC = int(radiusC)
                    cv2.circle(image,center,radiusC,(0,255,255),-1)
                    #cv2.putText(image, str(radiusC), center, font, 1, (0,255,255), 2, cv2.LINE_AA)

                #Draw the outer contour
                    cv2.circle(image,(int(outX),int(outY)),int(outRadius),(0,255,0),1)
                    cv2.putText(image, str(int(outRadius)), (int(outX),int(outY)), font, 1, (0,255,0), 2, cv2.LINE_AA)
                return (True, m)
            else:
                return (False, None)
        else:
            return (False, None)


    def areCornersDetected(self, ms):
        repeat = True
        print(len(ms))

        md = {}
        mu = {}
        index=0
        for m in ms:
            if (m.mid <= 4 and m.mid > 0):
                mu[index] = m
                md[m.mid] = m
                index +=1
                #print "Marker ", m.mid, " found"

        repeat = False
        if not md.has_key(1):
            #print("ERR: Marker 1 missing.")
            repeat = True
        if not md.has_key(2):
            #print("ERR: Marker 2 missing.")
            repeat = True
        if not md.has_key(3):
            #print("ERR: Marker 3 missing.")
            repeat = True
        if not md.has_key(4):
            #print("ERR: Marker 4 missing.")
            repeat = True
        if len(mu) is not 4:
            repeat = True

        return not repeat

    def areCornerMarkersWellPlaced(self, markers):

        if self.areCornersDetected(markers) == False:
            return False

        idx1, m1 = self._get_marker_object(1, markers)
        idx2, m2 = self._get_marker_object(2, markers)
        idx3, m3 = self._get_marker_object(3, markers)
        idx4, m4 = self._get_marker_object(4, markers)

        if(m1.realxy() is None):
            return False

        if(m1.realxy()[0] > 0.3 and m1.realxy()[0] < 0.2):
            return False
        if(m1.realxy()[1] > -0.45 and m1.realxy()[1] < -0.55):
            return False

        if(m2.realxy()[0] > 0.3 and m2.realxy()[0] < 0.2):
            return False
        if(m2.realxy()[1] < 0.45 and m2.realxy()[1] > 0.55):
            return False

        if(m3.realxy()[0] > 0.8 and m3.realxy()[0] < 0.7):
            return False
        if(m3.realxy()[1] < 0.45 and m3.realxy()[1] > 0.55):
            return False

        if(m4.realxy()[0] > 0.8 and m4.realxy()[0] < 0.7):
            return False
        if(m4.realxy()[1] > -0.45 and m4.realxy()[1] < -0.55):
            return False

        print("Corners: ")
        print(m1.realxy())
        print(m2.realxy())
        print(m3.realxy())
        print(m4.realxy())
        return True
