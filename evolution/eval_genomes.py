from utility.helpers import scale, euclidean_distance, f_wheel_center, f_straight_movements, f_pain, scale, scale_thymio_sensors
from utility.path_tracking import follow_path
from vision.tracker import get_marker_object
from datetime import datetime, timedelta
import vrep.vrep as vrep
import numpy as np
import neat


def eval_genomes_hardware(individual, settings, genomes, config):

    robot_m = get_marker_object(7)
    while robot_m.realxy() is None:
        # obtain goal marker postion
        robot_m = get_marker_object(7)
    init_position = robot_m.realxy()[:2]

    for _, genome in genomes:
        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            print('Failed to start the simulation\n')
            print('Program ended\n')
            return

        individual.chromosome = genome

        now = datetime.now()
        scaled_output = np.array([])
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # get robot marker
        robot_m = get_marker_object(7)
        if robot_m.realxy() is not None:
            start_position = robot_m.realxy()[:2]

        while datetime.now() - now < timedelta(seconds=settings.run_time):
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
                    get_marker_object, vrep, settings.client_id)


def eval_genomes_simulation(individual, settings, genomes, config):

    for _, genome in genomes:
        # Enable the synchronous mode
        vrep.simxSynchronous(settings.client_id, True)

        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            return

        individual.v_reset_init()
        individual.chromosome = genome
        now = datetime.now()
        collision = False
        scaled_output = np.array([])
        fitness_agg = np.array([])
        network = neat.nn.FeedForwardNetwork.create(genome, config)

        # collistion detection initialization
        _, collision_handle = vrep.simxGetCollisionHandle(
            settings.client_id, 'wall_collision', vrep.simx_opmode_blocking)
        _, collision = vrep.simxReadCollision(
            settings.client_id, collision_handle, vrep.simx_opmode_streaming)

        start_position = individual.v_get_position()

        while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):

            # The first simulation step waits for a trigger before being executed
            vrep.simxSynchronousTrigger(settings.client_id)
            _, collision = vrep.simxReadCollision(
                settings.client_id, collision_handle, vrep.simx_opmode_buffer)

            individual.v_neuro_loop()

            # Net output [0, 1]
            output = network.activate(individual.v_sensor_activation)

            # scale motor wheel wheel_speeds [0.0, 2.0] - robot
            scaled_output = np.array(
                [scale(xi, -2.0, 2.0) for xi in output])

            individual.v_set_motors(*list(scaled_output))

            # After this call, the first simulation step is finished
            vrep.simxGetPingTime(settings.client_id)

            # Fitness function; each feature;
            # V - wheel center
            V = f_wheel_center(output[0], output[1])
            # pleasure - straight movements
            pleasure = f_straight_movements(output[0], output[1])
            # pain - closer to an obstacle more pain
            pain = f_pain(np.array([scale_thymio_sensors(xi, 0.0, 1.0)
                                    for xi in individual.sensor_activation]))
            #  fitness_t at time stamp
            fitness_t = V * pleasure * pain
            fitness_agg = np.append(fitness_agg, fitness_t)

        # calculate the fitnesss
        fitness = np.sum(fitness_agg)

        # Now send some data to V-REP in a non-blocking fashion:
        vrep.simxAddStatusbarMessage(
            settings.client_id, 'fitness: {}'.format(fitness), vrep.simx_opmode_oneshot)

        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(settings.client_id)

        if (vrep.simxStopSimulation(settings.client_id, settings.op_mode) == -1):
            return

        genome.fitness = fitness
