from thymioII import ThymioII
from aseba import Aseba

import numpy as np


class EvolvedThymio(ThymioII, Aseba):

    def __init__(self, name, chromosome):
        super(EvolvedThymio, self).__init__(name)
        self.chromosome = chromosome
        self.wheel_speed = np.array([])
        self.norm_wheel_speed = np.array([])
        self.sensor_activation = np.array([])
        self.num_sensors = 7

    def __str__(self):
        return "Chromosome: %s\n WheelSpeed: %s\n Normalized Speed: %s\n Sensor Activation: %s\n Max Sensor Activation: %s\n" % (
            self.chromosome, self.wheel_speed, self.norm_wheel_speed, self.sensor_activation, self.sensor_activation)
