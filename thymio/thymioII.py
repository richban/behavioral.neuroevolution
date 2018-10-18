import dbus
import glib
import math
import time

from aseba import Aseba, AsebaException

__author__ = "Davide Laezza"

class ThymioII(Aseba):
    wheel_distance = 9 # cm
    
    def __init__(self, name):
        super(ThymioII, self).__init__()

        nodes = self.network.GetNodesList()
        if name not in nodes:
            nodes = map(str, list(nodes))
            raise AsebaException("Cannot find node {nodeName}! "
                                 "These are the available nodes: {nodes}" \
                                 .format(nodeName=name, nodes=list(nodes)))
        self.name = name
        self.desired_speed = 0

    def __enter__():
        pass

    def _turn(self, direction, deg):
        radians = math.pi * deg / 180
        speed = self.network.GetVariable(self.name,
                'motor.{dir}.speed'.format(dir=direction))

        cms_speed = speed[0] * 20 / 500 * 0.75
        if cms_speed <= 0:
            return
        
        time_stop = ThymioII.wheel_distance * radians / cms_speed
        self.network.SetVariable(self.name,
                'motor.{dir}.target'.format(dir=direction),
                [0])
        time.sleep(time_stop)
        self.move_forward(self.desired_speed)

    def get(self, *args, **kwargs):
        return super(ThymioII, self).get(self.name, *args, **kwargs)

    def set(self, *args, **kwargs):
        return super(ThymioII, self).set(self.name, *args, **kwargs)

    def move_forward(self, speed):
        self.desired_speed = speed
        self.network.SetVariable(self.name, 'motor.left.target', [speed])
        self.network.SetVariable(self.name, 'motor.right.target', [speed])

    def stop(self):
        self.move_forward(0)

    def turn_left(self, deg):
        self._turn('left', deg)

    def turn_right(self, deg):
        self._turn('right', deg)

    def u_turn(self):
        self._turn('right', 180)
