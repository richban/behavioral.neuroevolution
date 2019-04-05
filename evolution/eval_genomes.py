import time
import uuid
import neat
import numpy as np
import vrep.vrep as vrep
import threading
from datetime import datetime, timedelta
from vision.tracker import get_marker_object
from robot.vrep_robot import VrepRobot
from utility.path_tracking import follow_path, transform_pos_angle
from utility.util_functions import scale, euclidean_distance, \
    f_wheel_center, f_straight_movements, \
    f_obstacle_dist, scale, scale_thymio_sensors, \
    normalize_0_1, f_t_obstacle_avoidance, thymio_position
try:
    from robot.evolved_robot import EvolvedRobot
except ImportError as error:
    print(error.__class__.__name__ + ": " + 'DBus works only on linux!')


def eval_genomes_hardware(individual, settings, genomes, config):

    robot_m = get_marker_object(7)
    while robot_m.realxy() is None:
        # obtain goal marker postion
        robot_m = get_marker_object(7)
    init_position = robot_m.realxy()[:2]

    for genome_id, genome in genomes:
        # individual reset
        individual.n_t_sensor_activation = np.array([])
        individual.chromosome = genome
        individual.id = genome_id
        # simulation specific props
        collision = False
        scaled_output = np.array([])
        fitness_agg = np.array([])
        # neural network initialization
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # get robot marker
        robot_m = get_marker_object(7)
        if robot_m.realxy() is not None:
            # update current position of the robot
            robot_current_position = robot_m.realxy()[:2]

        # update position and orientation of the robot in vrep
        position, orientation = transform_pos_angle(
            robot_current_position, robot_m.orientation())
        individual.v_set_pos_angle(position, orientation)

        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            print('Failed to start the simulation\n')
            return

        # collistion detection initialization
        _, collision_handle = vrep.simxGetCollisionHandle(
            settings.client_id, 'wall_collision', vrep.simx_opmode_blocking)
        _, collision = vrep.simxReadCollision(
            settings.client_id, collision_handle, vrep.simx_opmode_streaming)

        now = datetime.now()

        while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):

            # get robot marker
            robot_m = get_marker_object(7)
            if robot_m.realxy() is not None:
                # update current position of the robot
                robot_current_position = robot_m.realxy()[:2]

            # update position and orientation of the robot in vrep
            position, orientation = transform_pos_angle(
                robot_current_position, robot_m.orientation())
            individual.v_set_pos_angle(position, orientation)

            _, collision = vrep.simxReadCollision(
                settings.client_id, collision_handle, vrep.simx_opmode_buffer)
            # read proximity sensors data
            individual.t_read_prox()

            # input data to the neural network
            net_output = net.activate(individual.n_t_sensor_activation)
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

            # dump individuals data
            if settings.debug:
                with open(settings.path + str(individual.id) + '_hw_simulation.txt', 'a') as f:
                    f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n'.format(
                        individual.id, net_output[0], net_output[1], scaled_output[0], scaled_output[1],
                        np.array2string(
                            individual.t_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        np.array2string(
                            individual.n_t_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        wheel_center, straight_movements, obstacles_distance, np.max(
                            individual.n_t_sensor_activation), fitness_t))

        individual.t_stop()
        # calculate the fitnesss
        fitness = np.sum(fitness_agg)/fitness_agg.size
        print('genome_id: %s fitness: %f' % (str(individual.id), fitness))
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

        # reset the individual
        individual.v_reset_init()
        individual.chromosome = genome
        individual.id = genome_id

        # evaluation specific props
        collision = False
        scaled_output = np.array([])
        fitness_agg = np.array([])

        # neural network initialization
        network = neat.nn.FeedForwardNetwork.create(genome, config)

        # Enable the synchronous mode
        vrep.simxSynchronous(settings.client_id, True)

        # timetep 50 ms
        dt = 0.05
        runtime = 0

        # start the simulation
        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            return

        # collistion detection initialization
        _, collision_handle = vrep.simxGetCollisionHandle(
            settings.client_id, 'wall_collision', vrep.simx_opmode_blocking)
        _, collision = vrep.simxReadCollision(
            settings.client_id, collision_handle, vrep.simx_opmode_streaming)

        now = datetime.now()

        while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):
            # The first simulation step waits for a trigger before being executed
            vrep.simxSynchronousTrigger(settings.client_id)
            _, collision = vrep.simxReadCollision(
                settings.client_id, collision_handle, vrep.simx_opmode_buffer)

            individual.v_neuro_loop()
            output = network.activate(individual.v_norm_sensor_activation)
            scaled_output = np.array(
                [scale(xi, -2.0, 2.0) for xi in output])

            # set motor wheel speeds
            individual.v_set_motors(*list(scaled_output))

            # After this call, the first simulation step is finished
            # Now we can safely read all  values
            vrep.simxGetPingTime(settings.client_id)
            runtime += dt

            #  fitness_t at time stamp
            (
                fitness_t,
                wheel_center,
                straight_movements,
                obstacles_distance
            ) = f_t_obstacle_avoidance(
                scaled_output, individual.v_norm_sensor_activation, 'vrep')

            fitness_agg = np.append(fitness_agg, fitness_t)

            # dump individuals data
            if settings.debug:
                with open(settings.path + str(individual.id) + '_simulation.txt', 'a') as f:
                    f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n'.format(
                        str(individual.id), output[0], output[1], scaled_output[0], scaled_output[1],
                        np.array2string(
                            individual.v_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        np.array2string(
                            individual.v_norm_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        wheel_center, straight_movements, obstacles_distance, np.amax(
                            individual.v_norm_sensor_activation), fitness_t))

        # calculate the fitnesss
        fitness = np.sum(fitness_agg)/fitness_agg.size

        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive.
        vrep.simxGetPingTime(settings.client_id)

        if (vrep.simxStopSimulation(settings.client_id, settings.op_mode) == -1):
            return

        print('genome_id: %s fitness: %f runtime: %f' %
              (str(individual.id), fitness, runtime))

        time.sleep(1)
        genome.fitness = fitness


def eval_genome(client_id, settings, genome_id, genome, config):

    t = threading.currentThread()
    kw = {'v_chromosome': genome}

    individual = VrepRobot(
        client_id=client_id,
        id=genome_id,
        op_mode=settings.op_mode,
        robot_type=settings.robot_type,
        **kw
    )
    # evolution specific props
    scaled_output = np.array([])
    fitness_agg = np.array([])

    # neural network initialization
    network = neat.nn.FeedForwardNetwork.create(genome, config)

    # Enable the synchronous mode
    vrep.simxSynchronous(client_id, True)

    # timetep 50 ms
    dt = 0.05
    runtime = 0

    if (vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot) == -1):
        return

    # collistion detection initialization
    _, collision_handle = vrep.simxGetCollisionHandle(
        client_id, 'wall_collision', vrep.simx_opmode_blocking)
    _, collision = vrep.simxReadCollision(
        client_id, collision_handle, vrep.simx_opmode_streaming)

    now = datetime.now()

    while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):
        # The first simulation step waits for a trigger before being executed
        vrep.simxSynchronousTrigger(client_id)
        _, collision = vrep.simxReadCollision(
            client_id, collision_handle, vrep.simx_opmode_buffer)

        individual.v_neuro_loop()
        # Net output [0, 1]
        output = network.activate(individual.v_norm_sensor_activation)
        # [-2, 2] wheel speed thymio
        scaled_output = np.array(
            [scale(xi, -2.0, 2.0) for xi in output])
        # set motor wheel speeds
        individual.v_set_motors(*list(scaled_output))
        # After this call, the first simulation step is finished
        vrep.simxGetPingTime(client_id)
        runtime += dt

        (
            fitness_t,
            wheel_center,
            straight_movements,
            obstacles_distance
        ) = f_t_obstacle_avoidance(
            scaled_output, individual.v_norm_sensor_activation, 'vrep')

        fitness_agg = np.append(fitness_agg, fitness_t)

        # dump individuals data
        if settings.debug:
            with open(settings.path + str(individual.id) + '_simulation.txt', 'a') as f:
                f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n'.format(
                    individual.id, output[0], output[1], scaled_output[0], scaled_output[1],
                    np.array2string(
                        individual.v_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                    np.array2string(
                        individual.v_norm_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                    wheel_center, straight_movements, obstacles_distance, np.amax(
                        individual.v_norm_sensor_activation), fitness_t,
                ))

    # calculate the fitnesss
    fitness = np.sum(fitness_agg)/fitness_agg.size

    # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive.
    vrep.simxGetPingTime(client_id)

    if (vrep.simxStopSimulation(client_id, settings.op_mode) == -1):
        return

    print('%s genome_id: %s fitness: %f runtime: %f' %
          (str(t.getName()), str(individual.id), fitness, runtime))

    time.sleep(1)
    return fitness


def post_eval_genome(individual, settings, genome, config):
    
    print('Postevaluation of {0} started!'.format(type(individual).__name__))
    
    network = neat.nn.FeedForwardNetwork.create(genome, config)
    
    if type(individual).__name__ == 'VrepRobot':
        individual.v_chromosome = genome
        individual.id = genome.key
        # Enable the synchronous mode
        vrep.simxSynchronous(settings.client_id, True)
        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            return
        # collistion detection initialization
        _, collision_handle = vrep.simxGetCollisionHandle(
            settings.client_id, 'wall_collision', vrep.simx_opmode_blocking)
        _, collision = vrep.simxReadCollision(
            settings.client_id, collision_handle, vrep.simx_opmode_streaming)

        now = datetime.now()

        while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):
            # The first simulation step waits for a trigger before being executed
            vrep.simxSynchronousTrigger(settings.client_id)
            _, collision = vrep.simxReadCollision(
                settings.client_id, collision_handle, vrep.simx_opmode_buffer)

            individual.v_neuro_loop()
            # Net output [0, 1]
            output = network.activate(individual.v_norm_sensor_activation)
            # [-2, 2] wheel speed thymio
            scaled_output = np.array(
                [scale(xi, -2.0, 2.0) for xi in output])
            # set motor wheel speeds
            individual.v_set_motors(*list(scaled_output))
            # After this call, the first simulation step is finished
            vrep.simxGetPingTime(settings.client_id)

        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive.
        vrep.simxGetPingTime(settings.client_id)

        if (vrep.simxStopSimulation(settings.client_id, settings.op_mode) == -1):
            return
        return individual

    elif type(individual).__name__ == 'EvolvedRobot':
        individual.chromosome = genome
        individual.id = genome.key
        t_xy, t_angle = thymio_position()
        
        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            print('Failed to start the simulation\n')
            return
 
        # update position and orientation of the robot in vrep
        position, orientation = transform_pos_angle(
            t_xy, t_angle)
        individual.v_set_pos_angle(position, orientation)

        # collistion detection initialization
        _, collision_handle = vrep.simxGetCollisionHandle(
            settings.client_id, 'wall_collision', vrep.simx_opmode_blocking)
        _, collision = vrep.simxReadCollision(
            settings.client_id, collision_handle, vrep.simx_opmode_streaming)

        now = datetime.now()

        while datetime.now() - now < timedelta(seconds=settings.run_time):

            t_xy, t_angle = thymio_position()
            # update position and orientation of the robot in vrep
            position, orientation = transform_pos_angle(
                t_xy, t_angle)
            individual.v_set_pos_angle(position, orientation)

            _, collision = vrep.simxReadCollision(
                settings.client_id, collision_handle, vrep.simx_opmode_buffer)
            # read proximity sensors data
            individual.t_read_prox()
            
            net_output = network.activate(individual.n_t_sensor_activation)
            # normalize motor wheel wheel_speeds [0.0, 2.0] - robot
            scaled_output = np.array([scale(xi, -200, 200)
                                      for xi in net_output])
            # set thymio wheel speeds
            individual.t_set_motors(*list(scaled_output))

        individual.t_stop()

        if (vrep.simxStopSimulation(settings.client_id, settings.op_mode) == -1):
            print('Failed to stop the simulation')
            print('Program ended')
            return
        return individual
    else:
        return None
