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


    robot = EvolvedRobot(name, CLIENT_ID, None, OP_MODE, None)
    now = datetime.now()

    while datetime.now() - now < timedelta(seconds=10):
        print(robot.check_prox())

    robot.set_motor(500, 500)
    time.sleep(10)
    robot.stop()


if __name__ == '__main__':
    try:
        main(sys.argv[1])
    except IndexError:
        main()
