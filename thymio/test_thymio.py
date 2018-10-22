from thymioII import ThymioII
from evolved_thymio import EvolvedThymio
from datetime import datetime, timedelta
import time
import sys


def main(name='thymio-II'):

    robot = EvolvedThymio(name)
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
