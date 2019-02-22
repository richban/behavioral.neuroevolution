import vrep
import time
import math
from datetime import datetime, timedelta
import numpy as np
import pickle
import logging
from utility.helpers import sensors_offset, normalize
import uuid

PORT_NUM = 19997
OP_MODE = vrep.simx_opmode_oneshot_wait


class VrepRobot(object):

    def __init__(self, client_id, id, op_mode, **kw):
        # super(VrepRobot, self).__init__(**kw)

        self.id = id
        self.client_id = client_id
        self.op_mode = op_mode

        # Robot Specific Attributes
        self.v_no_detection = 1.0
        self.v_minDetection = 0.05
        self.v_initSpeed = 2.0
        self.v_wheel_speeds = np.array([])
        self.v_sensor_activation = np.array([])
        self.v_norm_wheel_speeds = np.array([])
        self.v_position = (0, 0, 0)
        self.v_num_sensors = 16

        # Initialize Robot Body
        res, self.v_body = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx%s" %
            self.suffix, self.op_mode)

        # Initialize Left Motor
        res, self.v_left_motor = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx_leftMotor%s" %
            self.suffix, self.op_mode)

        # Initialize Right Motor
        res, self.v_right_motor = vrep.simxGetObjectHandle(
            self.client_id, "Pioneer_p3dx_rightMotor%s" %
            self.suffix, self.op_mode)

        # Initialize Wheels
        self.v_wheels = [self.v_left_motor, self.v_right_motor]

        # Initialize Proximity Sensors
        self.v_prox_sensors = []
        self.v_prox_sensors_val = np.array([])
        for i in range(1, self.v_num_sensors + 1):
            res, sensor = vrep.simxGetObjectHandle(
                self.client_id, 'Pioneer_p3dx_ultrasonicSensor%d%s' %
                (i, self.suffix), self.op_mode)
            self.v_prox_sensors.append(sensor)
            errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
                self.client_id, sensor, vrep.simx_opmode_streaming)
            np.append(self.v_prox_sensors_val, np.linalg.norm(detectedPoint))

        # Custom Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        c_handler = logging.StreamHandler()
        self.logger.setLevel(logging.INFO)
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        self.logger.addHandler(c_handler)

    @property
    def suffix(self):
        if self.id is not None:
            return '#%d' % self.id
        return ''

    def get_position(self):
        returnCode, self.v_position = vrep.simxGetObjectPosition(
            self.client_id, self.v_body, -1, self.op_mode)
        return self.v_position

    def set_position(self, position):
        returnCode = vrep.simxSetObjectPosition(
            self.client_id, self.v_body, -1, self.op_mode)
        if returnCode == vrep.simx_return_ok:
            print('Robot:', self.v_body, ' position: ', position)
        else:
            print('setPose remote function call failed.')
            return -1

    def v_move_forward(self, speed=2.0):
        self.v_set_motors(speed, speed)

    def v_move_backward(self, speed=2.0):
        self.v_set_motors(-speed, -speed)

    def v_set_motors(self, left, right):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.v_left_motor,
            left,
            vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.v_right_motor,
            right,
            vrep.simx_opmode_streaming)

    def v_set_left_motor(self, left):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.v_left_motor,
            left,
            vrep.simx_opmode_oneshot)

    def v_set_right_motor(self, right):
        vrep.simxSetJointTargetVelocity(
            self.client_id,
            self.v_right_motor,
            right,
            vrep.simx_opmode_oneshot)

    def v_get_sensor_state(self, sensor):
        errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
            self.client_id, sensor, vrep.simx_opmode_buffer)
        return detectionState

    def v_get_sensor_distance(self, sensor):
        errorCode, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(
            self.client_id, sensor, vrep.simx_opmode_buffer)
        return np.linalg.norm(detectedPoint)

    def v_test_sensors(self):
        while True:
            self.v_sensor_activation = np.array([])
            for s in self.v_prox_sensors:
                if self.v_get_sensor_state(s):
                    # offset
                    activation = sensors_offset(self.v_get_sensor_distance(s),
                                                self.v_min_detection, self.v_no_detection)
                    self.v_sensor_activation = np.append(
                        self.v_sensor_activation, activation)
                else:
                    self.v_sensor_activation = np.append(
                        self.v_sensor_activation, 0)

            self.logger.info('Sensors Activation {}'.format(
                self.v_sensor_activation))

    def v_read_prox(self):
        self.v_sensor_activation = np.array([])
        for i, sensor in enumerate(self.v_prox_sensors):
            if self.v_get_sensor_state(sensor):
                activation = self.v_get_sensor_distance(sensor)
                self.v_sensor_activation = np.append(
                    self.v_sensor_activation, activation)
            else:
                self.v_sensor_activation = np.append(
                    self.v_sensor_activation, 0)
        return self.v_sensor_activation


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
        robot = VrepRobot(client_id=client_id, id=None, op_mode=OP_MODE)
        vrep.simxStopSimulation(client_id, op_mode)
        vrep.simxStartSimulation(client_id, op_mode)
        robot.v_set_motors(2.0, 2.0)
        time.sleep(100)
        vrep.simxStopSimulation(client_id, op_mode)
        vrep.simxFinish(client_id)

        print('Program ended')
    else:
        print('Failed connecting to remote API server')
