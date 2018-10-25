import vrep
import math
import numpy as np
import pickle
import logging
# from helpers import sensors_offset, normalize

PI = math.pi
NUM_SENSORS = 16
PORT_NUM = 19997
RUNTIME = 20
OP_MODE = vrep.simx_opmode_oneshot_wait
X_MIN = 0
X_MAX = 48
DEBUG = False


class Robot:

    def __init__(self, client_id, id, op_mode, noDetection=1.0, minDetection=0.05, initSpeed=2):
        self.id = id
        self.client_id = client_id
        self.op_mode = op_mode

        # Specific props
        self.noDetection = noDetection
        self.minDetection = minDetection
        self.initSpeed = initSpeed
        self.wheel_speeds = np.array([])
        self.sensor_activation = np.array([])
        self.norm_wheel_speeds = np.array([])

        # Custom Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        c_handler = logging.StreamHandler()
        self.logger.setLevel(logging.INFO)
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        self.logger.addHandler(c_handler)

        res, self.body = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx%s" %
            self.suffix, self.op_mode)

        # Initialize Motors
        res, self.left_motor = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx_leftMotor%s" %
            self.suffix, self.op_mode)
        res, self.right_motor = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx_rightMotor%s" %
            self.suffix, self.op_mode)
        self.wheels = [self.left_motor, self.right_motor]

        # Initialize Proximity Sensors
        self.prox_sensors = []
        self.prox_sensors_val = np.array([])
        for i in range(1, NUM_SENSORS + 1):
            res, sensor = vrep.simxGetObjectHandle(
                self.client_id, 'Pioneer_p3dx_ultrasonicSensor%d%s' %
                (i, self.suffix), self.op_mode)
            self.prox_sensors.append(sensor)
            errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
                self.client_id, sensor, vrep.simx_opmode_streaming)
            np.append(self.prox_sensors_val, np.linalg.norm(detectedPoint))

        # Orientation of all the sensors:
        self.sensors_loc = np.array([-PI / 2, -50 / 180.0 * PI, -30 / 180.0 * PI,
                                    -10 / 180.0 * PI, 10 / 180.0 * PI, 30 / 180.0 * PI,
                                     50 / 180.0 * PI, PI / 2, PI / 2, 130 / 180.0 * PI,
                                     150 / 180.0 * PI, 170 / 180.0 * PI, -170 / 180.0 * PI,
                                     -150 / 180.0 * PI, -130 / 180.0 * PI, -PI / 2])

    @property
    def suffix(self):
        if self.id is not None:
            return '#%d' % self.id
        return ''

    def move_forward(self, speed=2.0):
        self.set_motors(speed, speed)

    def move_backward(self, speed=2.0):
        self.set_motors(-speed, -speed)

    def set_motors(self, left, right):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.left_motor,
            left,
            vrep.simx_opmode_streaming)
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.right_motor,
            right,
            vrep.simx_opmode_streaming)

    def set_left_motor(self, left):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.left_motor,
            left,
            vrep.simx_opmode_streaming)

    def set_right_motor(self, right):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.right_motor,
            right,
            vrep.simx_opmode_streaming)

    def get_sensor_state(self, sensor):
        errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
            self.client_id, sensor, vrep.simx_opmode_streaming)
        return detectionState

    def get_sensor_distance(self, sensor):
        errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
            self.client_id, sensor, vrep.simx_opmode_buffer)
        return np.linalg.norm(detectedPoint)

    @property
    def position(self):
        returnCode, (x, y, z) = vrep.simxGetObjectPosition(
            self.client_id, self.body, -1, self.op_mode)
        return x, y

    def save_robot(self, filename):
        with open(filename, 'wb') as robot:
            pickle.dump(self, robot)


if __name__ == '__main__':
    print('Program started')
    vrep.simxFinish(-1)  # just in case, close all opened connections
    client_id = vrep.simxStart(
        '127.0.0.1',
        PORT_NUM,
        True,
        True,
        5000,
        5)  # Connect to V-REP
    if client_id != -1:
        print('Connected to remote API server')
        op_mode = vrep.simx_opmode_oneshot_wait
        robot = Robot(client_id=client_id, id=None, op_mode=OP_MODE)
        vrep.simxStopSimulation(client_id, op_mode)
        vrep.simxStartSimulation(client_id, op_mode)
        vrep.simxStopSimulation(client_id, op_mode)
        vrep.simxFinish(client_id)

        print('Program ended')
    else:
        print('Failed connecting to remote API server')
