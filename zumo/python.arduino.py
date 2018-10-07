import serial
import sys
from datetime import datetime, timedelta

PORT = '/dev/cu.usbmodem14101'
BAUDRATE = 9600
RUNTIME = 30

try:
    ser = serial.Serial(PORT, BAUDRATE)
    print('Opening Port name', ser.name)
except serial.SerialException as e:
    sys.exit(e)

now = datetime.now()

while datetime.now() - now < timedelta(seconds=RUNTIME):
    packet = ser.readline()
    print("packets received: ", packet)

ser.close()
