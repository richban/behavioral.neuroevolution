from datetime import datetime
import vrep.vrep as vrep
import platform
import os


class Settings:
    def __init__(self, robot_type, save_data=True, debug=False, n_gen=20, conf_scene=False):
        self.client_id = -1
        self.op_mode = vrep.simx_opmode_oneshot_wait
        self.path = './data/neat/' + datetime.now().strftime('%Y-%m-%d') + '/'
        self.run_time = 60
        self.port_num = 19997
        self.address = '127.0.0.1'
        self.robot_type = robot_type
        self.save_data = save_data
        self.n_gen = n_gen
        self.exec_time = False
        self.debug = debug
        self.base_path = './data/neat/'
        self.vrep_abspath = '~/Developer/vrep-edu/vrep.app/Contents/MacOS/vrep'
        self.logtime_data = {}
        # Obstacle Markers IDs and dimensions in mm
        self.obstacle_markers = [
            dict([(9, dict(dimensios=[400, 80]))]),
            dict([(10, dict(dimensios=[200, 70]))]),
            dict([(11, dict(dimensios=[40, 100]))]),
        ]
        self.config_scene = conf_scene
        self.grid = None

        if platform.system() == 'Linux':
            self.vrep_abspath = '~/Developer/vrep-edu/vrep.sh'

        if self.save_data:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
