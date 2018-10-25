import vrep
import random
from datetime import timedelta, datetime
import numpy as np
from robot import Robot

OP_MODE = vrep.simx_opmode_oneshot_wait
PORT_NUM = 19997


def run():
    # Just in case, close all opened connections
    vrep.simxFinish(-1)

    client_id = vrep.simxStart(
        '127.0.0.1',
        PORT_NUM,
        True,
        True,
        5000,
        5)  # Connect to V-REP

    if client_id == -1:
        print('Failed connecting to remote API server')
        print('Program ended')
        return

    if (vrep.simxStartSimulation(client_id, OP_MODE) == -1):
        print('Failed to start the simulation\n')
        print('Program ended\n')
        return

    robot = Robot(client_id=client_id, id=0, op_mode=OP_MODE)
    
    mock_data = np.load('mock_data.npy')
    now = datetime.now()
    loc_stream = np.array([0, 0])

    while(vrep.simxGetConnectionId(client_id) != -1 and datetime.now() - now < timedelta(seconds=30)):
        err, loc_handler = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx', OP_MODE)
        err, loc = vrep.simxGetObjectPosition(client_id, loc_handler, -1, OP_MODE)
        err, orientation = vrep.simxGetObjectOrientation(client_id, loc_handler, -1, OP_MODE)

        for pos in mock_data:
            vrep.simxSetObjectPosition(client_id, loc_handler, -1, pos, vrep.simx_opmode_oneshot)
        
        (x, y) = robot.position
        loc_stream = np.vstack((loc_stream, np.array([x, y])))
        
    # np.save('mock_data', loc_stream)

    if (vrep.simxStopSimulation(client_id, OP_MODE) == -1):
        print('Failed to stop the simulation\n')
        print('Program ended\n')
        return


def generate_cor():
    random.seed(datetime.now())
    while True:
        yield (random.uniform(0.0, 4.0), random.uniform(0.0, 4.0))



if __name__ == '__main__':
    run()
