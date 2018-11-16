#! /usr/bin/python

# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "anfv"
__date__ = "$16-May-2016 10:43:47$"

import vision
import cv2
import time

def test_threshold():
    
    totalMarkers = 0
    maxMarkers = 0
    for i in range(20):
        
        
        name = str( i) + '.jpg'
        print "processing image ", name
        frame = cv2.imread(name,cv2.IMREAD_COLOR)
        markers = vision.get_markers(None, False, frame, True, 1, False)
        goodMarkers = {}
        superGoodMaekers = {}
        badMarkers = 0;
        for m in markers:
            if not goodMarkers.has_key(m.mid):
                goodMarkers[m.mid] = True
            else:
                badMarkers +=1
                goodMarkers[m.mid] = False
                
        for m in markers:
            if goodMarkers[(m.mid)]: 
                if m.mid == 1 or m.mid == 2 or m.mid == 3 or m.mid == 4 or m.mid == 5 or m.mid == 12 or m.mid == 6 or m.mid == 9 or m.mid == 11: 
                    totalMarkers +=1
                else:
                    badMarkers +=1
            else:
                badMarkers +=1
        
        print "Found ", len(markers), "markers"
        print ""
        
        maxMarkers += 9
        time.sleep(3)
    
    print "Found ", totalMarkers, " of ", maxMarkers, " ( ",  totalMarkers / float(maxMarkers) *100, "% )"
    print "Found ", badMarkers, " badMarkers"
    return

    
def adjust_parameter(name):
       
    
    frame = cv2.imread(name,cv2.IMREAD_COLOR)
    markers = vision.get_markers(None, False, frame, True, 1, False)
#    print "Found ", len(markers), "markers" 
    parameter =-3
    iter = 0
    repeat = True
    while (repeat and iter <5):
        markers = vision.get_markers(None, False, frame, True, 1, False, parameter)
        repeat = check(markers)
        parameter += 1
        iter +=1
#        print "Found ", len(markers), "markers" 
    print "parameter: ", parameter
    return

def check(ms):
    md = {}
    for m in ms:
        md[m.mid] = m
    repeat = True
    if md.has_key(6) and md.has_key(9) and md.has_key(11):
        repeat = False
    return repeat

def test():
    for i in range(20):
        adjust_parameter(str(i)+".jpg")
if __name__ == "__main__":
    print "Hello World";
    test_threshold()
#    test()
    
    
