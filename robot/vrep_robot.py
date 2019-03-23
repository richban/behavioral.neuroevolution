import vrep.vrep as vrep
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

    def __init__(self, client_id, id, op_mode, robot_type, **kw):
        # super(VrepRobot, self).__init__(**kw)

        self.id = id
        self.client_id = client_id
        self.op_mode = op_mode

        # Robot Specific Attributes
        self.v_chromosome = None
        self.v_no_detection = 1.0
        self.v_min_detection = 0.05
        self.v_initSpeed = 0.0
        self.v_wheel_speeds = np.array([])
        self.v_sensor_activation = np.array([])
        self.v_norm_wheel_speeds = np.array([])
        self.v_position = (0, 0, 0)
        self.v_num_sensors = robot_type['num_sensors']
        self.v_min_detection = 0.05
        self.v_robot_type = robot_type

        # Initialize Robot Body
        _, self.v_body = vrep.simxGetObjectHandle(
            self.client_id, "{}{}".format(self.v_robot_type['body'], self.suffix), self.op_mode)

        # Initialize Left Motor
        _, self.v_left_motor = vrep.simxGetObjectHandle(
            self.client_id, "{}{}".format(self.v_robot_type['left_motor'], self.suffix), self.op_mode)

        # Initialize Right Motor
        _, self.v_right_motor = vrep.simxGetObjectHandle(
            self.client_id, "{}{}".format(self.v_robot_type['right_motor'], self.suffix), self.op_mode)

        # Initialize Wheels
        self.v_wheels = [self.v_left_motor, self.v_right_motor]

        # Initialize Proximity Sensors
        self.v_prox_sensors = []
        self.v_prox_sensors_val = np.array([])
        for i in range(1, self.v_num_sensors + 1):
            _, sensor = vrep.simxGetObjectHandle(
                self.client_id, "{}{}{}".format(self.v_robot_type['sensor'], i, self.suffix), self.op_mode)
            self.v_prox_sensors.append(sensor)
            _, _, detectedPoint, _, _ = vrep.simxReadProximitySensor(
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
        if self.id is not None and self.v_robot_type['name'] == 'pd3x':
            return '#%d' % self.id
        elif self.id is not None and self.v_robot_type['name'] == 'thymio':
            return '%d' % self.id
        return ''

    def v_reset_init(self):
        self.v_chromosome = None
        self.v_wheel_speeds = np.array([])
        self.v_sensor_activation = np.array([])
        self.v_norm_wheel_speeds = np.array([])

    def v_get_position(self):
        _, self.v_position = vrep.simxGetObjectPosition(
            self.client_id, self.v_body, -1, self.op_mode)
        return np.array(self.v_position)

    def v_set_position(self, position):
        returnCode = vrep.simxSetObjectPosition(
            self.client_id, self.v_body, -1, position, self.op_mode)
        if returnCode == vrep.simx_return_ok:
            pass
            # print('Robot:', self.v_body, ' position: ', position)
        else:
            print('setPose remote function call failed.')
            return -1

    def v_set_orientation(self, orientation):
        """Set the orientation of an object in the simulation
        Euler angles (alpha, beta and gamma)
        parent_handle: -1 is the world frame, any other int should be a vrep object handle
        """
        res = vrep.simxSetObjectOrientation(
            self.client_id,
            self.v_body,
            -1,
            orientation,
            OP_MODE)
        if res == vrep.simx_return_ok:
            pass
            # print('SetOrientation object:', self.v_body,
            #      ' orientation: ', orientation)
        else:
            print('setOrientatinon remote function call failed.')
            return -1
        return orientation

    def v_set_pos_angle(self, position, orientation):
        """Set the orientation and position of the robot in the simulation"""
        self.v_set_position(position)
        self.v_set_orientation(orientation)

    def v_get_orientation(self, op_mode):
        """get the orientation of an object in the simulation
        Euler angles (alpha, beta and gamma)
        parent_handle: -1 is the world frame, any other int should be a vrep object handle
        """
        res, angles = vrep.simxGetObjectOrientation(
            self.client_id,
            self.v_body,
            -1,
            op_mode)
        if res == vrep.simx_return_ok:
            print('SetOrientation object:', self.v_body)
        else:
            print('get object orientation function call failed.')
            return -1
        return angles

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
            vrep.simx_opmode_oneshot)

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
        _, detectionState, _, _, _ = vrep.simxReadProximitySensor(
            self.client_id, sensor, vrep.simx_opmode_buffer)
        return detectionState

    def v_get_sensor_distance(self, sensor):
        _, _, detectedPoint, _, _ = vrep.simxReadProximitySensor(
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
            print(self.v_sensor_activation)
            time.sleep(10)

    def v_read_prox(self):
        self.v_sensor_activation = np.array([])
        for _, sensor in enumerate(self.v_prox_sensors):
            if self.v_get_sensor_state(sensor):
                activation = self.v_get_sensor_distance(sensor)
                self.v_sensor_activation = np.append(
                    self.v_sensor_activation, activation)
            else:
                self.v_sensor_activation = np.append(
                    self.v_sensor_activation, 0)
        return self.v_sensor_activation

    def v_neuro_loop(self, offset=False):
        self.v_sensor_activation = np.array([])
        for _, sensor in enumerate(self.v_prox_sensors):
            if self.v_get_sensor_state(sensor):
                if offset:
                    activation = sensors_offset(self.v_get_sensor_distance(
                        sensor), self.v_min_detection, self.v_no_detection)
                else:
                    activation = self.v_get_sensor_distance(sensor)
                self.v_sensor_activation = np.append(
                    self.v_sensor_activation, activation)
            else:
                self.v_sensor_activation = np.append(
                    self.v_sensor_activation, 0)

    def v_stop(self):
        self.v_set_motors(0, 0)


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
        vrep.simxStopSimulation(client_id, op_mode)
        vrep.simxStartSimulation(client_id, op_mode)
        time.sleep(10)
        vrep.simxStopSimulation(client_id, op_mode)
        vrep.simxFinish(client_id)

        print('Program ended')
    else:
        print('Failed connecting to remote API server')
