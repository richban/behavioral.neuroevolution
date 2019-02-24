import vrep
import time
import array
import cv2
import numpy as np
from PIL import Image

__credits__ = 'nemilya'

OP_MODE = vrep.simx_opmode_oneshot_wait
PORT_NUM = 19997


def track_green_object(image):
    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image for only green colors
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([80, 200, 200])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Blur the mask
    bmask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Take the moments to get the centroid
    moments = cv2.moments(bmask)
    m00 = moments['m00']
    centroid_x, centroid_y = None, None
    if m00 != 0:
        centroid_x = int(moments['m10']/m00)
        centroid_y = int(moments['m01']/m00)

    # Assume no centroid
    ctr = None

    # Use centroid if it exists
    if centroid_x is not None and centroid_y is not None:
        ctr = (centroid_x, centroid_y)
    return ctr


def run():
    # Just in case, close all opened connections
    vrep.simxFinish(-1)

    client_id = vrep.simxStart(
        '127.0.0.1',
        PORT_NUM,
        True,
        True,
        5000,
        5)  # Connect to V-REP

    if client_id == -1:
        print('Failed connecting to remote API server')
        print('Program ended')
        return

    # get vision sensor objects
    res, v0 = vrep.simxGetObjectHandle(
        client_id, 'v0', vrep.simx_opmode_oneshot_wait)
    res, v1 = vrep.simxGetObjectHandle(
        client_id, 'v1', vrep.simx_opmode_oneshot_wait)

    err, resolution, image = vrep.simxGetVisionSensorImage(
        client_id, v0, 0, vrep.simx_opmode_streaming)
    time.sleep(1)

    if (vrep.simxStartSimulation(client_id, OP_MODE) == -1):
        print('Failed to start the simulation\n')
        print('Program ended\n')
        return

    while (vrep.simxGetConnectionId(client_id) != -1):
        # get image from vision sensor 'v0'
        err, resolution, image = vrep.simxGetVisionSensorImage(
            client_id, v0, 0, vrep.simx_opmode_buffer)
        if err == vrep.simx_return_ok:
            image_byte_array = bytes(array.array('b', image))
            image_buffer = Image.frombuffer(
                "RGB", (resolution[0], resolution[1]), image_byte_array, "raw", "RGB", 0, 1)
            img2 = np.asarray(image_buffer)

            # try to find something green
            ret = track_green_object(img2)

            # overlay rectangle marker if something is found by OpenCV
            if ret:
                cv2.rectangle(img2, (ret[0] - 15, ret[1] - 15),
                              (ret[0] + 15, ret[1] + 15), (0xff, 0xf4, 0x0d), 1)

            # return image to sensor 'v1'
            img2 = img2.ravel()
            vrep.simxSetVisionSensorImage(
                client_id, v1, img2, 0, vrep.simx_opmode_oneshot)

        elif err == vrep.simx_return_novalue_flag:
            print("Object not received")
            pass
        else:
            print(err)


if __name__ == '__main__':
    run()
