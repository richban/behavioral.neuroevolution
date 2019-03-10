from robot.thymio_robot import ThymioII
from robot.vrep_robot import VrepRobot
from aseba.aseba import Aseba
from utility.helpers import normalize
import numpy as np

T_SEN_MIN = 0
T_SEN_MAX = 4000


class EvolvedRobot(VrepRobot, ThymioII):

    def __init__(self, name, client_id, id, op_mode, chromosome):
        VrepRobot.__init__(self, client_id, id, op_mode)
        ThymioII.__init__(self, name)

        self.chromosome = chromosome
        self.n_t_sensor_activation = np.array([])

    def t_read_prox(self):
        sensor_activation = super(EvolvedRobot, self).t_read_prox()
        self.n_t_sensor_activation = np.array(
            [normalize(xi, T_SEN_MIN, T_SEN_MAX, 0.0, 1.0) for xi in sensor_activation])
        return self.n_t_sensor_activation
