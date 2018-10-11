import sys
import numpy as np
import serial
import time
import math
from datetime import datetime, timedelta

PORT = '/dev/cu.usbmodem14101'
BAUDRATE = 9600
RUNTIME = 60
MIN_DIST = 3


class Robot(object):
    def __init__(self, port=PORT, baudrate=BAUDRATE):
        try:
            self.ser = serial.Serial(port, baudrate)
        except serial.serialexception as e:
            sys.exit(e)

    def set_motors(self, left, right):
        self.ser.write(str.encode('W, %d, %d\n' % (left, right)))

    def get_sensors(self):
        packet = self.ser.readline().decode('utf-8')
        if packet.startswith('S'):
            inputs = list(map(int, packet.strip().split(' ')[2:8]))
            return inputs

    def reset(self):
        self.ser.write(str.encode('R\n'))

    def close(self):
        self.ser.close()

    def collison_avoidance(self):
        PI = math.pi
        sensors_loc = np.array([-30 / 180.0 * PI, 30/180.0 * PI,
                                -PI / 2, -PI / 2, PI / 2, PI / 2])
        now = datetime.now()
        while datetime.now() - now < timedelta(seconds=RUNTIME):
            sensor_val = np.array(self.get_sensors())
            # controller specific
            # take only the front sensors
            sensor_sq = sensor_val[0:2]
            # find the minimum sensor value
            min_ind = np.where(sensor_sq == np.min(sensor_sq))
            min_ind = min_ind[0][0]
            if sensor_sq[min_ind] < 7:
                steer = -1 / sensors_loc[min_ind]
            else:
                steer = 0
            velocity = 100
            steering_gain = 0.5
            left_motor = velocity + steering_gain * steer
            right_motor = velocity - steering_gain * steer
            self.set_motors(left_motor, right_motor)

            time.sleep(0.1)


if __name__ == "__main__":
    args = sys.argv[1:]
    robot = Robot()
    robot.collison_avoidance()
    robot.reset()
    robot.close()
