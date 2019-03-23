from datetime import datetime
import vrep.vrep as vrep
import os


class Settings:
    def __init__(self, robot_type, save_data=False, n_gen=2):
        self.client_id = -1
        self.op_mode = vrep.simx_opmode_oneshot_wait
        self.path = './data/neat/' + datetime.now().strftime('%Y-%m-%d') + '/'
        self.run_time = 30
        self.port_num = 19997
        self.address = '127.0.0.1'
        self.robot_type = robot_type
        self.save_data = save_data
        self.n_gen = n_gen
        self.exec_time = True

        if not os.path.exists(self.path):
            os.makedirs(self.path)
