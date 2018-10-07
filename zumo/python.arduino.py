import serial
import sys
import time
from datetime import datetime, timedelta

PORT = '/dev/cu.usbmodem14201'
BAUDRATE = 9600
RUNTIME = 10

try:
    ser = serial.Serial(PORT, BAUDRATE)
    print('Opening Port name', ser.name)
except serial.SerialException as e:
    sys.exit(e)

now = datetime.now()

while datetime.now() - now < timedelta(seconds=RUNTIME):
    # packet = ser.readline()
    # print("packets received: ", packet)
    ser.write(str.encode('W, 100, 100\n'))
    time.sleep(0.1)

ser.write(str.encode('R\n'))
ser.close()
