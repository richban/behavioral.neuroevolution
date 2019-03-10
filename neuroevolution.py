import os
import neat
import time
import numpy as np
from datetime import datetime, timedelta
from utility.helpers import scale, euclidean_distance
from utility.evolution import log_statistics, visualize_results
from utility.path_tracking import search, \
    smooth, transform_points_from_image2real, \
    transform2robot_frame, is_near, \
    send_path_4_drawing, get_distance, \
    transform_pos_angle, pid, \
    pioneer_robot_model, \
    follow_path
from vision.tracker import Tracker, \
    get_marker_object, \
    get_markers, \
    CAM_MAT, CAM_DIST
from robot.evolved_robot import EvolvedRobot
import vrep.vrep as vrep
from functools import partial

PATH_NE = './data/neat/' + datetime.now().strftime('%Y-%m-%d') + '/'
OP_MODE = vrep.simx_opmode_oneshot_wait
PORT_NUM = 19997
CLIENT_ID = -1
RUNTIME = 10


def eval_genomes(robot, genomes, config):

    robot_m = get_marker_object(7)
    while robot_m.realxy() is None:
        # obtain goal marker postion
        robot_m = get_marker_object(7)
    init_position = robot_m.realxy()[:2]

    for _, genome in genomes:
        if (vrep.simxStartSimulation(CLIENT_ID, vrep.simx_opmode_oneshot) == -1):
            print('Failed to start the simulation\n')
            print('Program ended\n')
            return

        robot.chromosome = genome
        individual = robot

        now = datetime.now()
        scaled_output = np.array([])
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # get robot marker
        robot_m = get_marker_object(7)
        if robot_m.realxy() is not None:
            start_position = robot_m.realxy()[:2]

        while datetime.now() - now < timedelta(seconds=RUNTIME):
            # read proximity sensors data
            individual.t_read_prox()
            # input data to the neural network
            net_output = net.activate(
                list(map(lambda x: x if x != 0.0 else 1.0, individual.n_t_sensor_activation)))
            # normalize motor wheel wheel_speeds [0.0, 2.0] - robot
            scaled_output = np.array([scale(xi, 0.0, 300.0)
                                      for xi in net_output])
            # set thymio wheel speeds
            individual.t_set_motors(*list(scaled_output))

        # get robot marker
        robot_m = get_marker_object(7)
        if robot_m.realxy() is not None:
            end_position = robot_m.realxy()[:2]

        # calculate the euclidean distance
        fitness = euclidean_distance(end_position, start_position)
        genome.fitness = fitness[0]

        follow_path(individual, init_position,
                    get_marker_object, vrep, CLIENT_ID)


def run(config_file):
    print('Neuroevolutionary program started!')
    # Just in case, close all opened connections
    vrep.simxFinish(-1)
    global CLIENT_ID
    CLIENT_ID = vrep.simxStart(
        '127.0.0.1',
        PORT_NUM,
        True,
        True,
        5000,
        5)  # Connect to V-REP

    if CLIENT_ID == -1:
        print('Failed connecting to remote API server')
        print('Program ended')
        return

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)

    robot = EvolvedRobot(
        'thymio-II',
        client_id=CLIENT_ID,
        id=None,
        op_mode=OP_MODE,
        chromosome=None
    )

    # Run for up to N_GENERATIONS generations.
    winner = p.run(partial(eval_genomes, robot), 2)

    print('\nBest genome:\n{!s}'.format(winner))
    # Statistics and Visualizarion
    log_statistics(stats, winner, PATH_NE)
    visualize_results(config, stats, winner, PATH_NE)


if __name__ == '__main__':
    # Determine path to configuration file.
    local_dir = os.path.abspath('evolution')
    config_path = os.path.join(local_dir, 'config.ini')

    if not os.path.exists(PATH_NE):
        os.makedirs(PATH_NE)

    vision_thread = Tracker(mid=5,
                            transform=None,
                            mid_aux=0,
                            video_source=-1,
                            capture=False,
                            show=True,
                            debug=False,
                            )
    vision_thread.start()

    while vision_thread.cornersDetected is not True:
        print('Locating markers...')
        time.sleep(2)

    run(config_path)
