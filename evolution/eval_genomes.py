import time
import uuid
import neat
import uuid
import numpy as np
import vrep.vrep as vrep
import threading
import pickle
from datetime import datetime, timedelta
from vision.tracker import get_marker_object
from robot.vrep_robot import VrepRobot
from utility.path_tracking import follow_path, transform_pos_angle, create_grid
from utility.util_functions import scale, euclidean_distance, \
    f_wheel_center, f_straight_movements, \
    f_obstacle_dist, scale, scale_thymio_sensors, \
    normalize_0_1, f_t_obstacle_avoidance, thymio_position, \
    flatten_dict, calc_behavioral_features, save_debug_data
try:
    from robot.evolved_robot import EvolvedRobot
except ImportError as error:
    print(error.__class__.__name__ + ": " + 'DBus works only on linux!')
from multiprocessing import current_process
from vrep.control_env import get_object_handle, get_pose, set_pose
import schedule


def eval_genomes_hardware(individual, settings, genomes, config, generation):
    """Evaluation function to evaluate NEAT genomes on Thymio robot"""

    for _, genome in genomes:
        _ = eval_genome_hardware(
            individual,
            settings,
            genome,
            model=None,
            config=config,
            generation=generation
        )


def eval_genomes_simulation(individual, settings, genomes, config, generation):
    """Evaluation function to evaluate NEAT genomes in VREP simulator"""

    for _, genome in genomes:
        _ = eval_genome_simulation(
            individual,
            settings,
            None,
            config,
            generation,
            genome
        )


def post_eval_genome(individual, settings, genome, model=None, config=None, generation=None):
    """Post evaluation of controllers using NEAT.
       Only used for testing controllers evolved using NEAT.

       :individual: `VrepRobot` or `EvolvedRobot` (Thymio)
       :genome: controller
       :config: NN configuration
    """

    print('Postevaluation of {0} started!'.format(type(individual).__name__))

    # neural network initialization
    network = init_network(genome, config, model)

    if type(individual).__name__ == 'VrepRobot':
        individual.v_chromosome = genome
        individual.id = genome.key

        # Enable the synchronous mode
        vrep.simxSynchronous(settings.client_id, True)
        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            return

        # collistion detection initialization
        collision_handle, collision = init_collision(individual.client_id)

        dt = 0.05
        runtime = 0
        steps = 0

        while not collision and settings.run_time > runtime:
            # The first simulation step waits for a trigger before being executed
            vrep.simxSynchronousTrigger(settings.client_id)
            _, collision = vrep.simxReadCollision(
                settings.client_id, collision_handle, vrep.simx_opmode_buffer)

            individual.v_neuro_loop()

            # input data to the neural network
            if type(network).__name__ == 'FeedForwardNetwork':
                net_output = network.activate(
                    individual.v_norm_sensor_activation)

            if type(network).__name__ == 'Sequential':
                net_output = network.predict(
                    (individual.v_norm_sensor_activation).reshape((1, 7)))[0]
            # [-2, 2] wheel speed thymio
            scaled_output = np.array(
                [scale(xi, -2.0, 2.0) for xi in net_output])
            # set motor wheel speeds
            individual.v_set_motors(*list(scaled_output))
            # After this call, the first simulation step is finished
            vrep.simxGetPingTime(settings.client_id)
            runtime += dt
            steps += 1

        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive.
        vrep.simxGetPingTime(settings.client_id)

        if (vrep.simxStopSimulation(settings.client_id, settings.op_mode) == -1):
            return
        return individual

    elif type(individual).__name__ == 'EvolvedRobot':
        init_position = np.array([0.19, 0.22])

        # Build Scene and get the obstacles grid
        obstacle_grid = build_scene(
            settings.config_scene, settings, individual.client_id)

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
        collision_handle, collision = init_collision(individual.client_id)

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

            # input data to the neural network
            if type(network).__name__ == 'FeedForwardNetwork':
                net_output = network.activate(individual.n_t_sensor_activation)

            if type(network).__name__ == 'Sequential':
                net_output = network.predict(
                    (individual.n_t_sensor_activation).reshape((1, 7)))[0]

            # normalize motor wheel wheel_speeds [0.0, 2.0] - robot
            scaled_output = np.array([scale(xi, -200, 200)
                                      for xi in net_output])
            # set thymio wheel speeds
            individual.t_set_motors(*list(scaled_output))

        individual.t_stop()

        follow_path(
            individual,
            init_position,
            get_marker_object,
            vrep,
            settings.client_id,
            grid=obstacle_grid,
            log_time=settings.logtime_data
        )

        if (vrep.simxStopSimulation(settings.client_id, settings.op_mode) == -1):
            print('Failed to stop the simulation')
            print('Program ended')
            return
        return individual
    else:
        return None


def eval_genome_hardware(individual, settings, genome, model=None, config=None, generation=None):
    """Evaluation function for a single genome encoded with NEAT or DEAP."""

    robot_m = get_marker_object(7)
    while robot_m.realxy() is None:
        # obtain goal marker postion
        robot_m = get_marker_object(7)
    init_position = np.array([0.19, 0.22])

    # Build Scene and get the obstacles grid
    obstacle_grid = build_scene(
        settings.config_scene, settings, individual.client_id)

    # individual reset
    individual.n_t_sensor_activation = np.array([])
    individual.chromosome = genome
    individual.id = genome.key

    # simulation specific props
    thymio_position = []
    schedule.every(2).seconds.do(thymio_get_position_every_2s, thymio_position)

    collision = False
    scaled_output = np.array([])
    fitness_agg = np.array([])

    dt = 0.05
    runtime = 0
    steps = 0
    # Behavioral Features #1 and #2
    wheel_speeds = []
    sensor_activations = []

    # neural network initialization
    network = init_network(genome, config, model)

    # get robot marker
    robot_m = get_marker_object(7)
    if robot_m.realxy() is not None:
        # update current position of the robot
        robot_current_position = robot_m.realxy()[:2]

    # update position and orientation of the robot in vrep
    pos, orientation = transform_pos_angle(
        robot_current_position, robot_m.orientation())
    individual.v_set_pos_angle(pos, orientation)

    if (vrep.simxStartSimulation(individual.client_id, vrep.simx_opmode_oneshot) == -1):
        print('Failed to start the simulation\n')
        return

    # collistion detection initialization
    collision_handle, collision = init_collision(individual.client_id)

    # areas detection initlaization
    areas_name = ('area0', 'area1', 'area2')
    areas_handle = [(area,) + vrep.simxGetCollisionHandle(
        individual.client_id, area, vrep.simx_opmode_blocking) for area in areas_name]

    _ = [(handle[0],) + vrep.simxReadCollision(
        individual.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

    areas_counter = dict([(area, dict(count=0, percentage=0.0, total=0))
                          for area in areas_name])

    now = datetime.now()

    # while datetime.now() - now < timedelta(seconds=settings.run_time):
    while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):
        schedule.run_pending()
        # get robot marker
        robot_m = get_marker_object(7)
        if robot_m.realxy() is not None:
            # update current position of the robot
            robot_current_position = robot_m.realxy()[:2]

        # update position and orientation of the robot in vrep
        pos, orientation = transform_pos_angle(
            robot_current_position, robot_m.orientation())
        individual.v_set_pos_angle(pos, orientation)

        _, collision = vrep.simxReadCollision(
            individual.client_id, collision_handle, vrep.simx_opmode_buffer)

        # Behavioral Feature #3
        areas = [(handle[0],) + vrep.simxReadCollision(
            individual.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

        for area, _, detected in areas:
            if detected:
                areas_counter.get(area).update(
                    count=areas_counter.get(area)['count']+1)

        # read proximity sensors data
        individual.t_read_prox()

        # input data to the neural network
        if type(network).__name__ == 'FeedForwardNetwork':
            net_output = network.activate(individual.n_t_sensor_activation)

        if type(network).__name__ == 'Sequential':
            net_output = network.predict(
                (individual.n_t_sensor_activation).reshape((1, 7)))[0]

        # normalize motor wheel wheel_speeds [0.0, 2.0] - robot
        scaled_output = np.array([scale(xi, -200, 200)
                                  for xi in net_output])

        # Collect behavioral feature data
        wheel_speeds.append(net_output)
        sensor_activations.append(
            list(map(lambda x: 1 if x > 0.0 else 0, individual.n_t_sensor_activation)))

        # set thymio wheel speeds
        individual.t_set_motors(*list(scaled_output))

        runtime += dt
        steps += 1

        # every 10 seconds the robot is in the same position given a threshold stop the simulation
        if round(runtime, 2) % 10.0 == 0.0:
            print(euclidean_distance(thymio_position[0], thymio_position[-1]))
            if (euclidean_distance(thymio_position[0], thymio_position[-1])) < .09:
                collision = True

        #  fitness_t at time stamp
        (
            fitness_t,
            wheel_center,
            straight_movements,
            obstacles_distance
        ) = f_t_obstacle_avoidance(
            scaled_output, individual.n_t_sensor_activation, 'thymio')

        fitness_agg = np.append(fitness_agg, fitness_t)

        # dump individual data
        if settings.debug:
            save_debug_data(
                settings.path,
                individual.id,
                individual.t_sensor_activation,
                individual.n_t_sensor_activation,
                net_output,
                scaled_output,
                wheel_center,
                straight_movements,
                obstacles_distance,
                fitness_t,
                'THYMIO',
                robot_current_position
            )

    individual.t_stop()
    # calculate the fitnesss
    fitness = np.sum(fitness_agg)/settings.run_time

    schedule.clear()

    print('Transfered to thymio genome_id: {} fitness: {:.4f} runtime: {:.2f} s steps: {}'.format(
        individual.id, fitness, runtime, steps))

    if type(genome).__name__ == 'Individual':
        generation = genome.gen

    behavioral_features = calc_behavioral_features(
        areas_counter,
        wheel_speeds,
        sensor_activations,
        settings.path,
        genome.key,
        generation,
        'THYMIO'
    )

    if settings.debug:
        print('behavioral_features: {0}\n pos_sample: {1}\n'.format(
            behavioral_features, thymio_position))

    if type(genome).__name__ == 'Individual':
        genome.features = behavioral_features
        genome.task_fitness = fitness
        genome.position = thymio_position
        genome.evaluation = 'THYMIO'
        genome.weights = network.get_weights()

    if type(genome).__name__ == 'DefaultGenome':
        genome.features = behavioral_features
        genome.fitness = fitness
        genome.position = thymio_position

    follow_path(
        individual,
        init_position,
        get_marker_object,
        vrep,
        individual.client_id,
        grid=obstacle_grid,
        log_time=settings.logtime_data
    )

    if (vrep.simxStopSimulation(individual.client_id, settings.op_mode) == -1):
        print('Failed to stop the simulation')
        print('Program ended')
        return

    time.sleep(1)

    return fitness


def eval_genome_simulation(individual, settings, model, config, generation, genome):
    """Evaluation function for multiobjective optimization NSGA-II.
       :individual: robotic controller (VREP)
       :settings: simulation specific settings
       :model: Keras model Feedforward NN
       :genome: weights of the NN encoded that are being optimized

       :return: (fitness, transferability)

       fitness - task dependent fitness value V * (1 - sqr(delta v)) * (1 - max(S_activation))
       transferability - measure the distance betweeen simulation and real behavior
    """

    # reset the individual
    individual.v_reset_init()
    individual.chromosome = genome
    individual.id = genome.key

    # Behavioral Features #1 and #2
    wheel_speeds = []
    sensor_activations = []

    position = []

    # evaluation specific props
    collision = False
    scaled_output = np.array([], ndmin=2)
    fitness_agg = np.array([], ndmin=2)

    # setting time step to 50 ms (miliseconds)
    dt = 0.05
    runtime = 0
    steps = 0

    network = init_network(genome, config, model)

    # Enable the synchronous mode
    vrep.simxSynchronous(individual.client_id, True)

    # start the simulation
    if (vrep.simxStartSimulation(individual.client_id, vrep.simx_opmode_oneshot) == -1):
        return

    # collistion detection initialization
    collision_handle, collision = init_collision(individual.client_id)

    # areas detection initlaization
    areas_name = ('area0', 'area1', 'area2')
    areas_handle = [(area,) + vrep.simxGetCollisionHandle(
        individual.client_id, area, vrep.simx_opmode_blocking) for area in areas_name]

    _ = [(handle[0],) + vrep.simxReadCollision(
        individual.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

    areas_counter = dict([(area, dict(count=0, percentage=0.0, total=0))
                          for area in areas_name])

    while not collision and settings.run_time > runtime:
        # The first simulation step waits for a trigger before being executed
        vrep.simxSynchronousTrigger(individual.client_id)

        # read the collision
        _, collision = vrep.simxReadCollision(
            individual.client_id, collision_handle, vrep.simx_opmode_buffer)

        # Behavioral Feature #3
        areas = [(handle[0],) + vrep.simxReadCollision(
            individual.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

        for area, _, detected in areas:
            if detected:
                areas_counter.get(area).update(
                    count=areas_counter.get(area)['count']+1)

        # read proximity sensors data
        individual.v_neuro_loop()

        # input data to the neural network
        if type(network).__name__ == 'FeedForwardNetwork':
            net_output = network.activate(individual.v_norm_sensor_activation)

        if type(network).__name__ == 'Sequential':
            net_output = network.predict(
                (individual.v_norm_sensor_activation).reshape((1, 7)))[0]

        scaled_output = np.array(
            [scale(xi, -2.0, 2.0) for xi in net_output])

        # Collect behavioral feature data
        wheel_speeds.append(net_output)
        sensor_activations.append(
            list(map(lambda x: 1 if x > 0.0 else 0, individual.v_norm_sensor_activation)))

        # set motor wheel speeds
        individual.v_set_motors(*list(scaled_output))

        # sample every 2 second robot position
        if round(runtime, 2) % 2.0 == 0.0:
            # get vrep robot current position
            # print(individual.v_get_position())
            position.append(individual.v_get_position())

        # After this call, the first simulation step is finished
        # Now we can safely read all  values
        vrep.simxGetPingTime(individual.client_id)

        runtime += dt
        steps += 1

        # every 10 seconds the robot is in the same position given a threshold stop the simulation
        if round(runtime, 2) % 10.0 == 0.0:
            # print(euclidean_distance(position[0], position[-1]))
            if (euclidean_distance(position[0], position[-1])) < .01:
                collision = True

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
            save_debug_data(
                settings.path,
                individual.id,
                individual.v_sensor_activation,
                individual.v_norm_sensor_activation,
                net_output,
                scaled_output,
                wheel_center,
                straight_movements,
                obstacles_distance,
                fitness_t,
                'VREP',
                position[-1]
            )

    # calculate the fitnesss
    fitness = np.sum(fitness_agg)/settings.run_time

    # Now send some data to V-REP in a non-blocking fashion:
    vrep.simxAddStatusbarMessage(
        individual.client_id, 'genome_id: {} fitness: {:.4f} runtime: {:.2f} s'.format(
            individual.id, fitness, runtime), vrep.simx_opmode_oneshot)

    # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    vrep.simxGetPingTime(individual.client_id)

    if (vrep.simxStopSimulation(individual.client_id, settings.op_mode) == -1):
        return

    if type(genome).__name__ == 'Individual':
        generation = genome.gen

    behavioral_features = calc_behavioral_features(
        areas_counter,
        wheel_speeds,
        sensor_activations,
        settings.path,
        genome.key,
        generation,
        'VREP'
    )

    print('genome_id: {} fitness: {:.4f} runtime: {:.2f} s steps: {}'.format(
        genome.key, fitness, runtime, steps))

    if settings.debug:
        print('behavioral_features: {0}\n pos_sample: {1}\n n_samples: {2}\n'.format(
            behavioral_features, position, len(position)))

    if type(genome).__name__ == 'Individual':
        genome.features = np.array(behavioral_features)
        genome.task_fitness = fitness
        genome.position = np.array(position)
        genome.evaluation = 'VREP'
        genome.weights = network.get_weights()

        # Save the neural network model
        model.save(settings.path + 'keras_models/' +
                   str(individual.id) + '_model.h5')

    if type(genome).__name__ == 'DefaultGenome':
        genome.features = behavioral_features
        genome.fitness = fitness
        genome.position = position

    time.sleep(1)

    return fitness


#####################################################
#               AUXILIARY METHODS                   #
#               AUXILIARY METHODS                   #
#               AUXILIARY METHODS                   #
#               AUXILIARY METHODS                   #
#####################################################


def thymio_get_position_every_2s(position):
    robot_m = get_marker_object(7)
    if robot_m.realxy() is not None:
        position.append(robot_m.realxy()[:2])


def init_network(genome, config=None, model=None):

    network = None

    if type(genome).__name__ == 'Individual':
        if True:
            # TODO the sizeing is hardcoded.
            weights = [
                np.array(genome[:35]).reshape(genome.weights_shape[0]),
                np.array(genome[35:40]).reshape(genome.weights_shape[1]),
                np.array(genome[40:50]).reshape(genome.weights_shape[2]),
                np.array(genome[-2:]).reshape(genome.weights_shape[3]),
            ]

            model.set_weights(weights)
        else:
            model.set_weights(genome)

        network = model

    if type(genome).__name__ == 'DefaultGenome':
        network = neat.nn.FeedForwardNetwork.create(genome, config)

    return network


def build_scene(config_scene, settings, client_id):
    if config_scene:
            # Get the position of all the obstacles in reality
        obstacles_pos = [get_marker_object(obstacle).realxy()
                         for obstacle in (9, 10, 11)]
        # Get all obstacle handlers from VREP
        obstacle_handlers = [get_object_handle(client_id, obstacle) for obstacle in (
            'obstacle', 'obstacle1', 'obstacle0')]
        # Set the position of obstacles in vrep according the obstacles from reality
        for obs, handler in zip(obstacles_pos, obstacle_handlers):
            set_pose(client_id, handler, [obs[0], obs[1], 0.099999])

        # add markers position to obstacle_markers
        for position, marker in zip(obstacles_pos, settings.obstacle_markers):
            for _, value in marker.items():
                value.update(center=(position[:2]*1000).astype(int))

        obstacle_grid = create_grid(settings.obstacle_markers)
    else:
        # add markers position to obstacle_markers
        obstacles_pos = [[620, 590, 0], [880, 100, 0], [150, 430, 0]]
        for position, marker in zip(obstacles_pos, settings.obstacle_markers):
            for _, value in marker.items():
                value.update(center=position[:2])
        obstacle_grid = create_grid(settings.obstacle_markers)

    return obstacle_grid


def init_collision(client_id):
    # collistion detection initialization
    _, collision_handle = vrep.simxGetCollisionHandle(
        client_id, 'wall_collision', vrep.simx_opmode_blocking)
    _, collision = vrep.simxReadCollision(
        client_id, collision_handle, vrep.simx_opmode_streaming)

    return collision_handle, collision
