from aseba.aseba import Aseba, AsebaException
from dbus import DBusException
import math
import time

__credits__ = "Davide Laezza"


class ThymioII(Aseba):

    def __init__(self, name, **kw):
        super(ThymioII, self).__init__(**kw)

        nodes = self.network.GetNodesList()
        if name not in nodes:
            nodes = map(str, list(nodes))
            raise AsebaException("Cannot find node {nodeName}! "
                                 "These are the available nodes: {nodes}"
                                 .format(nodeName=name, nodes=list(nodes)))
        self.t_name = name
        self.t_speed = 0
        self.t_num_sensors = 7

    def get(self, *args, **kwargs):
        return super(ThymioII, self).get(self.t_name, *args, **kwargs)

    def set(self, *args, **kwargs):
        return super(ThymioII, self).set(self.t_name, *args, **kwargs)

    def t_move_forward(self, speed):
        self.desired_speed = speed
        self.network.SetVariable(self.t_name, 'motor.left.target', [speed])
        self.network.SetVariable(self.t_name, 'motor.right.target', [speed])

    def t_set_motors(self, left, right):
        self.network.SetVariable(self.t_name, 'motor.left.target', [left])
        self.network.SetVariable(self.t_name, 'motor.right.target', [right])

    def t_stop(self):
        self.network.SetVariable(self.t_name, 'motor.left.target', [0])
        self.network.SetVariable(self.t_name, 'motor.right.target', [0])

    def t_read_prox(self):
        try:
            return self.get('prox.horizontal')
        except DBusException:
            return [0, 0, 0, 0, 0, 0, 0]
