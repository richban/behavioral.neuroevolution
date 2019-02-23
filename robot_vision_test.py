from vision.tracker import Tracker, get_marker_object, get_markers, euclidian_distance
from robot.thymio_robot import ThymioII
from datetime import datetime, timedelta
from vision.calibration import restore
from math import fabs
import time


def thymio(name='thymio-II'):

    robot = ThymioII(name)

    now = datetime.now()

    robot.t_set_motors(-80, -80)

    while datetime.now() - now < timedelta(seconds=10):
        # print(robot.t_read_prox())
        continue

    robot.t_stop()


def braitenberg(robot):

    # Read Proximity sensors
    prox_sensors_val = robot.t_read_prox()[:5]

    # Parameters of the Braitenberg, to give weight to each wheels
    left_wheel = [-0.01, -0.005, -0.0001, 0.006, 0.015]
    right_wheel = [0.012, +0.007, -0.0002, -0.0055, -0.011]

    # Braitenberg algorithm
    total_left = 0
    total_right = 0

    for i in range(5):
        total_left = total_left + (prox_sensors_val[i] * left_wheel[i])
        total_right = total_right + (prox_sensors_val[i] * right_wheel[i])

    # add a constant speed to each wheels so the robot moves always forward
    total_right = total_right + 50
    total_left = total_left + 50

    robot.t_set_motors(total_left, total_right)


if __name__ == "__main__":

    goal_position = [0.59646187, 0.38297419, 0.0]
    goal_orientation = 1.7359450042095235
    transform, _, _ = restore()
    robot = ThymioII('Thymio-II')
    robot_home = False
    relax_dist = 0.01
    relax_angle = 0.15
    vision_thread = Tracker(mid=5,
                            transform=None,
                            mid_aux=0,
                            video_source=-1,
                            capture=False,
                            show=True,
                            debug=True
                            )

    vision_thread.start()

    while robot_home is not True:
        braitenberg(robot)

        robot_m = get_marker_object(9)

        if robot_m is not None:
            dist = euclidian_distance(robot_m.realxy()[:2], goal_position)
            rot = fabs(robot_m.orientation() - goal_orientation)

            if dist > relax_dist or rot > relax_angle:
                robot_home = True
                robot.t_stop()
