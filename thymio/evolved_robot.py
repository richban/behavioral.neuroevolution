from thymio_robot import ThymioII
from vrep_robot import VrepRobot
from aseba import Aseba

import numpy as np


class EvolvedRobot(ThymioII, VrepRobot):

    def __init__(self, t_name, client_id, id, op_mode, chromosome):
        super(EvolvedRobot, self).__init__(t_name, client_id, id, op_mode)
        self.chromosome = chromosome
