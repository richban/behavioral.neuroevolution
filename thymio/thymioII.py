from aseba import Aseba, AsebaException

__credits__ = "Davide Laezza"


class ThymioII(Aseba):
    
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

    def get(self, *args, **kwargs):
        return super(ThymioII, self).get(self.name, *args, **kwargs)

    def set(self, *args, **kwargs):
        return super(ThymioII, self).set(self.name, *args, **kwargs)

    def move_forward(self, speed):
        self.desired_speed = speed
        self.network.SetVariable(self.name, 'motor.left.target', [speed])
        self.network.SetVariable(self.name, 'motor.right.target', [speed])

    def set_motor(self, left, right):
        self.network.SetVariable(self.name, 'motor.left.target', [left])
        self.network.SetVariable(self.name, 'motor.right.target', [right])

    def stop(self):
        self.move_forward(0)

    def check_prox(self):
        return self.get('prox.horizontal')


