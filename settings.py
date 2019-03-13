from datetime import datetime
import vrep.vrep as vrep
import os


class Settings:
    def __init__(self):
        self.client_id = -1
        self.op_mode = vrep.simx_opmode_oneshot_wait
        self.path = './data/neat/' + datetime.now().strftime('%Y-%m-%d') + '/'
        self.run_time = 60
        self.port_num = 19997
        self.address = '127.0.0.1'

        if not os.path.exists(self.path):
            os.makedirs(self.path)
