from robot.thymio_robot import ThymioII
from robot.vrep_robot import VrepRobot
from robot.aseba import Aseba

import numpy as np


class EvolvedRobot(VrepRobot, ThymioII):

    def __init__(self, name, client_id, id, op_mode, chromosome):
        VrepRobot.__init__(self, client_id, id, op_mode)
        ThymioII.__init__(self, name)

        self.chromosome = chromosome