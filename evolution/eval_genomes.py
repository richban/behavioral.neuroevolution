import time
import uuid
import neat
import numpy as np
import vrep.vrep as vrep
from datetime import datetime, timedelta
from vision.tracker import get_marker_object
from utility.path_tracking import follow_path, transform_pos_angle
from utility.helpers import scale, euclidean_distance, \
    f_wheel_center, f_straight_movements, \
    f_obstacle_dist, scale, scale_thymio_sensors, normalize_0_1


def eval_genomes_hardware(individual, settings, genomes, config):

    robot_m = get_marker_object(7)
    while robot_m.realxy() is None:
        # obtain goal marker postion
        robot_m = get_marker_object(7)
    init_position = robot_m.realxy()[:2]

    for genome_id, genome in genomes:
        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            print('Failed to start the simulation\n')
            print('Program ended\n')
            return
        
        individual.n_t_sensor_activation = np.array([])
        individual.chromosome = genome
        now = datetime.now()
        id = genome_id
        collision = False
        scaled_output = np.array([])
        fitness_agg = np.array([])
        net = neat.nn.FeedForwardNetwork.create(genome, config)
 
        # get robot marker
        robot_m = get_marker_object(7)
        if robot_m.realxy() is not None:
            # update current position of the robot
            robot_current_position = robot_m.realxy()[:2]

        # calculate robot orientation
        # update position and orientation of the robot in vrep
        position, orientation = transform_pos_angle(
            robot_current_position, robot_m.orientation())
        individual.v_set_pos_angle(position, orientation)
        
        # collistion detection initialization
        _, collision_handle = vrep.simxGetCollisionHandle(
            settings.client_id, 'wall_collision', vrep.simx_opmode_blocking)
        _, collision = vrep.simxReadCollision(
            settings.client_id, collision_handle, vrep.simx_opmode_streaming)

        while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):
            step_start = time.time()

            ts = time.time()
            # get robot marker
            robot_m = get_marker_object(7)
            if robot_m.realxy() is not None:
                # update current position of the robot
                robot_current_position = robot_m.realxy()[:2]

            # update position and orientation of the robot in vrep
            position, orientation = transform_pos_angle(
                robot_current_position, robot_m.orientation())
            individual.v_set_pos_angle(position, orientation)
            te = time.time()
            if settings.exec_time:
                time_marker = (te - ts) * 1000
                # print('%s  %2.2f ms' % ('get/set position', time_marker))

            _, collision = vrep.simxReadCollision(
                settings.client_id, collision_handle, vrep.simx_opmode_buffer)
            # read proximity sensors data
            ts = time.time()
            individual.t_read_prox()
            te = time.time()
            if settings.exec_time:
                time_sensors = (te - ts) * 1000
                # print('%s  %2.2f ms' % ('sensory readings', time_sensors))

            # input data to the neural network
            ts = time.time()
            net_output = net.activate(individual.n_t_sensor_activation)
                # list(map(lambda x: x if x != 0.0 else 1.0, individual.n_t_sensor_activation)))
            te = time.time()
            if settings.exec_time:
                time_network = (te - ts) * 1000
                # print('%s  %2.2f ms' % ('network output', time_network))

            ts = time.time()
            # normalize motor wheel wheel_speeds [0.0, 2.0] - robot
            scaled_output = np.array([scale(xi, -200, 200)
                                      for xi in net_output])
            # set thymio wheel speeds
            individual.t_set_motors(*list(scaled_output))

            # Fitness function; each feature;
            # V - wheel center
            wheel_center = f_wheel_center(scaled_output, -200, 200)
            # pleasure - straight movements
            straight_movements = f_straight_movements(scaled_output, 0, 400)
            # pain - closer to an obstacle more pain
            obstacles_distance = f_obstacle_dist(
                individual.n_t_sensor_activation)
            #  fitness_t at time stamp
            fitness_t = wheel_center * straight_movements * obstacles_distance
            fitness_agg = np.append(fitness_agg, fitness_t)

            te = time.time()
            if settings.exec_time:
                time_calculation = (te - ts) * 1000
                # print('%s  %2.2f ms' %
                #      ('fitness calculation', (te - ts) * 1000))

            step_end = time.time()
            if settings.exec_time:
                time_simulation_step = (step_end - step_start) * 1000
                # print('%s  %2.2f ms' %
                #      ('simulation_step', (step_end - step_start) * 1000))

            # dump individuals data
            if settings.save_data:
                with open(settings.path + str(id) + '_hw_simulation.txt', 'a') as f:
                    f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16}\n'.format(
                        id, net_output[0], net_output[1], scaled_output[0], scaled_output[1],
                        np.array2string(
                            individual.t_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        np.array2string(
                            individual.n_t_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        wheel_center, straight_movements, obstacles_distance, np.max(
                            individual.n_t_sensor_activation), fitness_t,
                        time_marker, time_sensors, time_network, time_calculation, time_simulation_step))

        individual.t_stop()
        # calculate the fitnesss
        fitness = np.sum(fitness_agg)/fitness_agg.size
        print('genome_id: %s fitness: %f' % (str(id), fitness))
        genome.fitness = fitness

        follow_path(individual, init_position,
                    get_marker_object, vrep, settings.client_id)

        if (vrep.simxStopSimulation(settings.client_id, settings.op_mode) == -1):
            print('Failed to stop the simulation')
            print('Program ended')
            return

        time.sleep(1)


def eval_genomes_simulation(individual, settings, genomes, config):

    for genome_id, genome in genomes:
        # Enable the synchronous mode
        vrep.simxSynchronous(settings.client_id, True)

        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            return

        individual.v_reset_init()
        individual.chromosome = genome
        id = genome_id
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

        while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):
            step_start = time.time()
            # The first simulation step waits for a trigger before being executed
            vrep.simxSynchronousTrigger(settings.client_id)
            _, collision = vrep.simxReadCollision(
                settings.client_id, collision_handle, vrep.simx_opmode_buffer)

            ts = time.time()
            individual.v_neuro_loop()
            te = time.time()
            if settings.exec_time:
                time_sensors = (te - ts) * 1000
                # print('%s  %2.2f ms' % ('sensory readings', (ts - te) * 1000))
            # print(individual.v_sensor_activation)
            # Net output [0, 1]
            ts = time.time()
            output = network.activate(individual.v_norm_sensor_activation)
            te = time.time()
            if settings.exec_time:
                time_network = (te - ts) * 1000
                # print('%s  %2.2f ms' % ('network output', (te - ts) * 1000))

            # Scalling and normalization
            ts = time.time()
            # [-2, 2] wheel speed thymio
            scaled_output = np.array(
                [scale(xi, -2.0, 2.0) for xi in output])
            # set motor wheel speeds
            individual.v_set_motors(*list(scaled_output))

            # After this call, the first simulation step is finished
            vrep.simxGetPingTime(settings.client_id)

            # Fitness function; each feature;
            # V - wheel center
            wheel_center = f_wheel_center(scaled_output, -2.0, 2.0)
            # pleasure - straight movements
            straight_movements = f_straight_movements(scaled_output, 0.0, 4.0)
            # pain - closer to an obstacle more pain
            obstacles_distance = f_obstacle_dist(
                individual.v_norm_sensor_activation)
            #  fitness_t at time stamp
            fitness_t = wheel_center * straight_movements * obstacles_distance
            fitness_agg = np.append(fitness_agg, fitness_t)

            te = time.time()
            if settings.exec_time:
                time_calculation = (te - ts) * 1000
                # print('%s  %2.2f ms' % ('fitness calculation', (te - ts) * 1000))

            step_end = time.time()
            if settings.exec_time:
                time_simulation_step = (step_end - step_start) * 1000
                # print('%s  %2.2f ms' % ('simulation_step', (step_end - step_start) * 1000))

            # dump individuals data
            if settings.save_data:
                with open(settings.path + str(id) + '_simulation.txt', 'a') as f:
                    f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}\n'.format(
                        id, output[0], output[1], scaled_output[0], scaled_output[1],
                        np.array2string(
                            individual.v_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        np.array2string(
                            individual.v_norm_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        wheel_center, straight_movements, obstacles_distance, np.amax(
                            individual.v_norm_sensor_activation), fitness_t,
                        time_sensors, time_network, time_calculation, time_simulation_step))

        # calculate the fitnesss
        fitness = np.sum(fitness_agg)/fitness_agg.size

        # Now send some data to V-REP in a non-blocking fashion:
        vrep.simxAddStatusbarMessage(
            settings.client_id, 'fitness: {}'.format(fitness), vrep.simx_opmode_oneshot)

        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(settings.client_id)

        if (vrep.simxStopSimulation(settings.client_id, settings.op_mode) == -1):
            return

        print('genome_id: %s fitness: %f' % (str(id), fitness))

        time.sleep(1)
        genome.fitness = fitness
