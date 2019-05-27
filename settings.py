from datetime import datetime
import vrep.vrep as vrep
import platform
import os


class Settings:
    def __init__(self, robot_type, save_data=True, debug=False, n_gen=20, multiobjective, conf_scene=True):
        self.client_id = -1
        self.op_mode = vrep.simx_opmode_oneshot_wait
        self.path = './data/neat/' + datetime.now().strftime('%Y-%m-%d') + '/'
        self.run_time = 10
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
        self.pop = 20
        self.CXPB = 0.3
        self.STR = 0.5
        self.multiobjective = multiobjective
        # Obstacle Markers IDs and dimensions in mm
        self.obstacle_markers = [
            dict([(9, dict(dimension=[80, 400]))]),
            dict([(10, dict(dimension=[40, 250]))]),
            dict([(11, dict(dimension=[260, 60]))]),
        ]
        self.config_scene = conf_scene
        self.grid = None

        if platform.system() == 'Linux':
            self.vrep_abspath = '~/Developer/vrep-edu/vrep.sh'

        if self.save_data:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

            if self.multiobjective:
                if not os.path.exists(self.path+'deap_inds/'):
                    os.makedirs(self.path+'deap_inds/')

                if not os.path.exists(self.path+'keras_models/'):
                    os.makedirs(self.path+'keras_models/')
