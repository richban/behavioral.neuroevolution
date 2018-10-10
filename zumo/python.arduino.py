import sys
import time
from datetime import datetime, timedelta

import serial

PORT = '/dev/cu.usbmodem14101'
BAUDRATE = 9600
RUNTIME = 10

try:
    ser = serial.Serial(PORT, BAUDRATE)
    print('Opening Port name', ser.name)
except serial.SerialException as e:
    sys.exit(e)

now = datetime.now()

print('Program Starting...')
time.sleep(1)

while datetime.now() - now < timedelta(seconds=RUNTIME):
    packet = ser.readline()
    print("packets received: ", packet)
    ser.write(str.encode('W, 100, 100\n'))
    time.sleep(0.1)
    ser.write(str.encode('B\n'))


print('Shuting down the agent.')

ser.write(str.encode('R\n'))
time.sleep(0.5)
ser.close()

print('Agent shut down.')
