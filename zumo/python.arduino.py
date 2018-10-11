import sys
import time
from datetime import datetime, timedelta

import serial

PORT = '/dev/cu.usbmodem14101'
BAUDRATE = 9600
RUNTIME = 10


class Com(object):
    def __init__(self, port=PORT, baudrate=BAUDRATE):
        try:
            self.ser = serial.Serial(PORT, BAUDRATE)
        except serial.SerialException as e:
            sys.exit(e)

    def __del__(self):
        self.ser.close()

    def test_motor(self):
        self.ser.write(str.encode('W, 100, 100\n'))
        time.sleep(2)
        self.ser.write(str.encode('B\n'))
        time.sleep(2)
        self.ser.write(str.encode('F\n'))
        time.sleep(2)
        self.ser.write(str.encode('R\n'))

    def test_sensors(self):
        now = datetime.now()
        while datetime.now() - now < timedelta(seconds=RUNTIME):
            self.ser.write(str.encode('S\n'))
            time.sleep(0.1)
            packet = self.ser.readline()
            print("packets received: ", packet)

    def stream_sensors(self):
        while True:
            packet = self.ser.readline()
            time.slep(0.1)
            print(packet)


if __name__ == "__main__":
    args = sys.argv[1:]
    arduino = Com(*args)
    # arduino.test_motor()
    arduino.test_sensors()
    # arduino.stream_sensors()
