from thymio_robot import ThymioII
from evolved_robot import EvolvedRobot
from datetime import datetime, timedelta

import time
import sys
import vrep

OP_MODE = vrep.simx_opmode_oneshot_wait


def main(name='thymio-II'):
    print('Neuroevolutionary program started!')
    # Just in case, close all opened connections
    vrep.simxFinish(-1)

    CLIENT_ID = vrep.simxStart(
        '127.0.0.1',
        19997,
        True,
        True,
        5000,
        5)  # Connect to V-REP

    if CLIENT_ID == -1:
        print('Failed connecting to remote API server')
        print('Program ended')
        return

    if (vrep.simxStartSimulation(CLIENT_ID, vrep.simx_opmode_oneshot) == -1):
        print('Failed to start the simulation\n')
        print('Program ended\n')
        return

    robot = EvolvedRobot(name, CLIENT_ID, None, OP_MODE, None)
    now = datetime.now()

    robot.v_set_motors(1.0, 1.0)

    while datetime.now() - now < timedelta(seconds=10):
        print(robot.t_read_prox())
        print(robot.v_loop())

    robot.t_set_motors(500, 500)
    time.sleep(10)
    robot.t_stop()

    if (vrep.simxStopSimulation(CLIENT_ID, OP_MODE) == -1):
        print('Failed to stop the simulation\n')
        print('Program ended\n')
        return


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except IndexError:
        main()
