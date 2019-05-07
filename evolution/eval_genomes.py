import time
import uuid
import neat
import numpy as np
import vrep.vrep as vrep
import threading
from datetime import datetime, timedelta
from vision.tracker import get_marker_object
from robot.vrep_robot import VrepRobot
from utility.path_tracking import follow_path, transform_pos_angle, create_grid
from utility.util_functions import scale, euclidean_distance, \
    f_wheel_center, f_straight_movements, \
    f_obstacle_dist, scale, scale_thymio_sensors, \
    normalize_0_1, f_t_obstacle_avoidance, thymio_position, \
    flatten_dict
try:
    from robot.evolved_robot import EvolvedRobot
except ImportError as error:
    print(error.__class__.__name__ + ": " + 'DBus works only on linux!')
from multiprocessing import current_process
from vrep.control_env import get_object_handle, get_pose, set_pose


def eval_genomes_hardware(individual, settings, genomes, config):

    robot_m = get_marker_object(7)
    while robot_m.realxy() is None:
        # obtain goal marker postion
        robot_m = get_marker_object(7)
    init_position = np.array([0.19, 0.22])

    if settings.config_scene:
        # Get the position of all the obstacles in reality
        obstacles_pos = [get_marker_object(obstacle).realxy()
                         for obstacle in (9, 10, 11)]
        # Get all obstacle handlers from VREP
        obstacle_handlers = [get_object_handle(settings.client_id, obstacle) for obstacle in (
            'obstacle', 'obstacle1', 'obstacle0')]
        # Set the position of obstacles in vrep according the obstacles from reality
        for obs, handler in zip(obstacles_pos, obstacle_handlers):
            set_pose(settings.client_id, handler, [obs[0], obs[1], 0.099999])

        # add markers position to obstacle_markers
        for position, marker in zip(obstacles_pos, settings.obstacle_markers):
            for _, value in marker.items():
                value.update(center=(position[:2]*1000).astype(int))

        obstacle_grid = create_grid(settings.obstacle_markers)
    else:
        obstacle_grid = None

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

        # timetep 50 ms
        dt = 0.05
        runtime = 0
        steps = 0

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

        # areas detection initlaization
        areas_name = ('area0', 'area1', 'area2')
        areas_handle = [(area,) + vrep.simxGetCollisionHandle(
            settings.client_id, area, vrep.simx_opmode_blocking) for area in areas_name]

        _ = [(handle[0],) + vrep.simxReadCollision(
            settings.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

        areas_counter = dict([(area, dict(count=0, percentage=0.0, total=0))
                              for area in areas_name])
        # Behavioral Features #1 and #2
        wheel_speeds = []
        sensor_activations = []

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

            # Behavioral Feature #3
            areas = [(handle[0],) + vrep.simxReadCollision(
                settings.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

            for area, _, detected in areas:
                if detected:
                    areas_counter.get(area).update(
                        count=areas_counter.get(area)['count']+1)

            # read proximity sensors data
            individual.t_read_prox()
            # input data to the neural network
            net_output = net.activate(individual.n_t_sensor_activation)
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

            #  fitness_t at time stamp
            (
                fitness_t,
                wheel_center,
                straight_movements,
                obstacles_distance
            ) = f_t_obstacle_avoidance(
                scaled_output, individual.n_t_sensor_activation, 'thymio')

            fitness_agg = np.append(fitness_agg, fitness_t)

            # dump individuals data
            if settings.debug:
                with open(settings.path + str(individual.id) + '_hw_simulation.txt', 'a') as f:
                    f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n'.format(
                        individual.id, net_output[0], net_output[1], scaled_output[0], scaled_output[1],
                        np.array2string(
                            individual.t_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        np.array2string(
                            individual.n_t_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        wheel_center, straight_movements, obstacles_distance, np.max(
                            individual.n_t_sensor_activation), fitness_t, robot_current_position))

        individual.t_stop()
        # calculate the fitnesss
        fitness = np.sum(fitness_agg)/settings.run_time

        print('genome_id: {} fitness: {:.4f} runtime: {:.2f} s steps: {}'.format(
            individual.id, fitness, runtime, steps))

        # Compute and store behavioral featuers
        total_steps_in_areas = sum(val['count']
                                   for _, val in areas_counter.items())
        for _, value in areas_counter.items():
            value.update(
                percentage=value['count']/total_steps_in_areas,
                total=total_steps_in_areas
            )

        avg_wheel_speeds = np.mean(np.array(wheel_speeds), axis=0)
        avg_sensors_activation = np.mean(np.array(sensor_activations), axis=0)

        behavioral_features = np.concatenate((
            [individual.id],
            avg_wheel_speeds,
            avg_sensors_activation,
            list(flatten_dict(areas_counter).values()))
        )

        with open(settings.path + str(individual.id) + '_behavioral_features.dat', 'a') as b:
            np.savetxt(b, (behavioral_features,), delimiter=',',
                       fmt='%d,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%d,%1.3f,%d,%d,%1.3f,%d,%d,%1.3f,%d')

        genome.fitness = fitness

        follow_path(individual, init_position,
                    get_marker_object, vrep, settings.client_id, grid=obstacle_grid, log_time=settings.logtime_data)

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
        scaled_output = np.array([], ndmin=2)
        fitness_agg = np.array([], ndmin=2)

        # neural network initialization
        network = neat.nn.FeedForwardNetwork.create(genome, config)

        # Enable the synchronous mode
        vrep.simxSynchronous(settings.client_id, True)

        # timetep 50 ms
        dt = 0.05
        runtime = 0
        steps = 0

        # start the simulation
        if (vrep.simxStartSimulation(settings.client_id, vrep.simx_opmode_oneshot) == -1):
            return

        # collistion detection initialization
        _, collision_handle = vrep.simxGetCollisionHandle(
            settings.client_id, 'wall_collision', vrep.simx_opmode_blocking)
        _, collision = vrep.simxReadCollision(
            settings.client_id, collision_handle, vrep.simx_opmode_streaming)

        # areas detection initlaization
        areas_name = ('area0', 'area1', 'area2')
        areas_handle = [(area,) + vrep.simxGetCollisionHandle(
            settings.client_id, area, vrep.simx_opmode_blocking) for area in areas_name]

        _ = [(handle[0],) + vrep.simxReadCollision(
            settings.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

        areas_counter = dict([(area, dict(count=0, percentage=0.0, total=0))
                              for area in areas_name])
        # Behavioral Features #1 and #2
        wheel_speeds = []
        sensor_activations = []

        now = datetime.now()

        while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):
            # The first simulation step waits for a trigger before being executed
            vrep.simxSynchronousTrigger(settings.client_id)
            _, collision = vrep.simxReadCollision(
                settings.client_id, collision_handle, vrep.simx_opmode_buffer)

            # Behavioral Feature #3
            areas = [(handle[0],) + vrep.simxReadCollision(
                settings.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

            for area, _, detected in areas:
                if detected:
                    areas_counter.get(area).update(
                        count=areas_counter.get(area)['count']+1)

            individual.v_neuro_loop()
            output = network.activate(individual.v_norm_sensor_activation)
            scaled_output = np.array(
                [scale(xi, -2.0, 2.0) for xi in output])

            # Collect behavioral feature data
            wheel_speeds.append(output)
            sensor_activations.append(
                list(map(lambda x: 1 if x > 0.0 else 0, individual.v_norm_sensor_activation)))

            # set motor wheel speeds
            individual.v_set_motors(*list(scaled_output))

            # After this call, the first simulation step is finished
            # Now we can safely read all  values
            vrep.simxGetPingTime(settings.client_id)
            runtime += dt
            steps += 1
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
                with open(settings.path + str(individual.id) + '_simulation.dat', 'a') as f:
                    f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n'.format(
                        str(
                            individual.id), output[0], output[1], scaled_output[0], scaled_output[1],
                        np.array2string(
                            individual.v_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        np.array2string(
                            individual.v_norm_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        wheel_center, straight_movements, obstacles_distance, np.amax(
                            individual.v_norm_sensor_activation), fitness_t))

        # calculate the fitnesss
        fitness = np.sum(fitness_agg)/settings.run_time

        # Now send some data to V-REP in a non-blocking fashion:
        vrep.simxAddStatusbarMessage(
            settings.client_id, 'genome_id: {} fitness: {:.4f} runtime: {:.2f} s'.format(
                individual.id, fitness, runtime), vrep.simx_opmode_oneshot)

        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(settings.client_id)

        if (vrep.simxStopSimulation(settings.client_id, settings.op_mode) == -1):
            return

        print('genome_id: {} fitness: {:.4f} runtime: {:.2f} s steps: {}'.format(
            individual.id, fitness, runtime, steps))

        # Compute and store behavioral featuers
        total_steps_in_areas = sum(val['count']
                                   for _, val in areas_counter.items())
        for _, value in areas_counter.items():
            value.update(
                percentage=value['count']/total_steps_in_areas,
                total=total_steps_in_areas
            )

        avg_wheel_speeds = np.mean(np.array(wheel_speeds), axis=0)
        avg_sensors_activation = np.mean(np.array(sensor_activations), axis=0)

        behavioral_features = np.concatenate((
            [individual.id],
            avg_wheel_speeds,
            avg_sensors_activation,
            list(flatten_dict(areas_counter).values())
        )
        )

        with open(settings.path + str(individual.id) + '_behavioral_features.dat', 'a') as b:
            np.savetxt(b, (behavioral_features,), delimiter=',',
                       fmt='%d,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%d,%1.3f,%d,%d,%1.3f,%d,%d,%1.3f,%d')

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
    steps = 0

    if (vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot) == -1):
        return

    # collistion detection initialization
    _, collision_handle = vrep.simxGetCollisionHandle(
        client_id, 'wall_collision', vrep.simx_opmode_blocking)
    _, collision = vrep.simxReadCollision(
        client_id, collision_handle, vrep.simx_opmode_streaming)

    # areas detection initlaization
    areas_name = ('area0', 'area1', 'area2')
    areas_handle = [(area,) + vrep.simxGetCollisionHandle(
        settings.client_id, area, vrep.simx_opmode_blocking) for area in areas_name]

    _ = [(handle[0],) + vrep.simxReadCollision(
        settings.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

    areas_counter = dict([(area, dict(count=0, percentage=0.0, total=0))
                          for area in areas_name])
    # Behavioral Features #1 and #2
    wheel_speeds = []
    sensor_activations = []

    now = datetime.now()

    while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):
        # The first simulation step waits for a trigger before being executed
        vrep.simxSynchronousTrigger(client_id)
        _, collision = vrep.simxReadCollision(
            client_id, collision_handle, vrep.simx_opmode_buffer)

        # Behavioral Feature #3
        areas = [(handle[0],) + vrep.simxReadCollision(
            settings.client_id, handle[2], vrep.simx_opmode_streaming) for handle in areas_handle]

        for area, _, detected in areas:
            if detected:
                areas_counter.get(area).update(
                    count=areas_counter.get(area)['count']+1)

        individual.v_neuro_loop()
        # Net output [0, 1]
        output = network.activate(individual.v_norm_sensor_activation)
        # [-2, 2] wheel speed thymio
        scaled_output = np.array(
            [scale(xi, -2.0, 2.0) for xi in output])

        # Collect behavioral feature data
        wheel_speeds.append(output)
        sensor_activations.append(
            list(map(lambda x: 1 if x > 0.0 else 0, individual.v_norm_sensor_activation)))
        # set motor wheel speeds
        individual.v_set_motors(*list(scaled_output))
        # After this call, the first simulation step is finished
        vrep.simxGetPingTime(client_id)
        runtime += dt
        steps += 1

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
    fitness = np.sum(fitness_agg)/settings.run_time

    # Now send some data to V-REP in a non-blocking fashion:
    vrep.simxAddStatusbarMessage(
        settings.client_id, '{} genome_id: {} fitness: {:.4f} runtime: {:.2f} s'.format(
            str(t.getName()), individual.id, fitness, runtime), vrep.simx_opmode_oneshot)

    # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
    vrep.simxGetPingTime(settings.client_id)

    if (vrep.simxStopSimulation(client_id, settings.op_mode) == -1):
        return

    print('{} genome_id: {} fitness: {:.4f} runtime: {:.2f} s'.format(
        str(t.getName()), individual.id, fitness, runtime))

    # Compute and store behavioral featuers
    total_steps_in_areas = sum(val['count']
                               for _, val in areas_counter.items())
    for _, value in areas_counter.items():
        value.update(
            percentage=value['count']/total_steps_in_areas,
            total=total_steps_in_areas
        )

    avg_wheel_speeds = np.mean(np.array(wheel_speeds), axis=0)
    avg_sensors_activation = np.mean(np.array(sensor_activations), axis=0)

    behavioral_features = np.concatenate((
        [individual.id],
        avg_wheel_speeds,
        avg_sensors_activation,
        list(flatten_dict(areas_counter).values()))
    )

    with open(settings.path + str(individual.id) + '_behavioral_features.dat', 'a') as b:
        np.savetxt(b, (behavioral_features,), delimiter=',',
                   fmt='%d,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%d,%1.3f,%d,%d,%1.3f,%d,%d,%1.3f,%d')

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


def eval_transferability(vrep_bot, thymio_bot, settings, genomes, config):

    t = 0
    for genome_id, genome in genomes:

        # reset the vrep individual
        vrep_bot.v_reset_init()
        vrep_bot.chromosome = genome
        vrep_bot.id = genome_id

        # evaluation specific props
        collision = False
        scaled_output = np.array([])
        fitness_agg = np.array([])

        # neural network initialization
        network = neat.nn.FeedForwardNetwork.create(genome, config)

        # Enable the synchronous mode
        vrep.simxSynchronous(vrep_bot.client_id, True)

        # timetep 50 ms
        dt = 0.05
        runtime = 0

        # start the simulation
        if (vrep.simxStartSimulation(vrep_bot.client_id, vrep.simx_opmode_oneshot) == -1):
            return

        # collistion detection initialization
        _, collision_handle = vrep.simxGetCollisionHandle(
            vrep_bot.client_id, 'wall_collision', vrep.simx_opmode_blocking)
        _, collision = vrep.simxReadCollision(
            vrep_bot.client_id, collision_handle, vrep.simx_opmode_streaming)

        now = datetime.now()

        while not collision and datetime.now() - now < timedelta(seconds=settings.run_time):
            # The first simulation step waits for a trigger before being executed
            vrep.simxSynchronousTrigger(vrep_bot.client_id)
            _, collision = vrep.simxReadCollision(
                vrep_bot.client_id, collision_handle, vrep.simx_opmode_buffer)

            vrep_bot.v_neuro_loop()
            output = network.activate(vrep_bot.v_norm_sensor_activation)
            scaled_output = np.array(
                [scale(xi, -2.0, 2.0) for xi in output])

            # set motor wheel speeds
            vrep_bot.v_set_motors(*list(scaled_output))

            # After this call, the first simulation step is finished
            # Now we can safely read all  values
            vrep.simxGetPingTime(vrep_bot.client_id)
            runtime += dt

            #  fitness_t at time stamp
            (
                fitness_t,
                wheel_center,
                straight_movements,
                obstacles_distance
            ) = f_t_obstacle_avoidance(
                scaled_output, vrep_bot.v_norm_sensor_activation, 'vrep')

            fitness_agg = np.append(fitness_agg, fitness_t)

            # dump individuals data
            if settings.debug:
                with open(settings.path + str(vrep_bot.id) + '_simulation.txt', 'a') as f:
                    f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n'.format(
                        str(
                            vrep_bot.id), output[0], output[1], scaled_output[0], scaled_output[1],
                        np.array2string(
                            vrep_bot.v_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        np.array2string(
                            vrep_bot.v_norm_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                        wheel_center, straight_movements, obstacles_distance, np.amax(
                            vrep_bot.v_norm_sensor_activation), fitness_t))

        # calculate the fitnesss
        fitness = np.sum(fitness_agg)/settings.run_time

        # Now send some data to V-REP in a non-blocking fashion:
        vrep.simxAddStatusbarMessage(
            vrep_bot.client_id, 'genome_id: {} fitness: {:.4f} runtime: {:.2f} s'.format(
                vrep_bot.id, fitness, runtime), vrep.simx_opmode_oneshot)

        # Before closing the connection to V-REP, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        vrep.simxGetPingTime(vrep_bot.client_id)

        if (vrep.simxStopSimulation(vrep_bot.client_id, settings.op_mode) == -1):
            return

        print('vrep genome_id: {} fitness: {:.4f} runtime: {:.2f} s'.format(
            vrep_bot.id, fitness, runtime))

        time.sleep(1)
        genome.fitness = fitness
        t += 1

        if (t % 2 == 0):
            # print('Transfering.....:\n {0}'.format(genome))
            f = eval_genome_hardware(thymio_bot, settings, genome, config)


def eval_genome_hardware(individual, settings, genome, config):

    robot_m = get_marker_object(7)
    while robot_m.realxy() is None:
        # obtain goal marker postion
        robot_m = get_marker_object(7)
    init_position = np.array([0.19, 0.22])

    # individual reset
    individual.n_t_sensor_activation = np.array([])
    individual.chromosome = genome
    individual.id = genome.key
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

    # timetep 50 ms
    dt = 0.05
    runtime = 0

    # update position and orientation of the robot in vrep
    position, orientation = transform_pos_angle(
        robot_current_position, robot_m.orientation())
    individual.v_set_pos_angle(position, orientation)

    if (vrep.simxStartSimulation(individual.client_id, vrep.simx_opmode_oneshot) == -1):
        print('Failed to start the simulation\n')
        return

    # collistion detection initialization
    _, collision_handle = vrep.simxGetCollisionHandle(
        individual.client_id, 'wall_collision', vrep.simx_opmode_blocking)
    _, collision = vrep.simxReadCollision(
        individual.client_id, collision_handle, vrep.simx_opmode_streaming)

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
            individual.client_id, collision_handle, vrep.simx_opmode_buffer)
        # read proximity sensors data
        individual.t_read_prox()

        # input data to the neural network
        net_output = net.activate(individual.n_t_sensor_activation)
        # normalize motor wheel wheel_speeds [0.0, 2.0] - robot
        scaled_output = np.array([scale(xi, -200, 200)
                                  for xi in net_output])
        # set thymio wheel speeds
        individual.t_set_motors(*list(scaled_output))

        runtime += dt
        #  fitness_t at time stamp
        (
            fitness_t,
            wheel_center,
            straight_movements,
            obstacles_distance
        ) = f_t_obstacle_avoidance(
            scaled_output, individual.n_t_sensor_activation, 'thymio')

        fitness_agg = np.append(fitness_agg, fitness_t)

        # dump individuals data
        if settings.debug:
            with open(settings.path + str(individual.id) + '_hw_simulation.txt', 'a') as f:
                f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n'.format(
                    individual.id, net_output[0], net_output[1], scaled_output[0], scaled_output[1],
                    np.array2string(
                        individual.t_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                    np.array2string(
                        individual.n_t_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                    wheel_center, straight_movements, obstacles_distance, np.max(
                        individual.n_t_sensor_activation), fitness_t, robot_current_position))

    individual.t_stop()
    # calculate the fitnesss
    fitness = np.sum(fitness_agg)/settings.run_time

    print('thymio genome_id: {} fitness: {:.4f} runtime: {:.2f} s'.format(
        individual.id, fitness, runtime))

    follow_path(individual, init_position,
                get_marker_object, vrep, individual.client_id, log_time=settings.logtime_data)

    if (vrep.simxStopSimulation(individual.client_id, settings.op_mode) == -1):
        print('Failed to stop the simulation')
        print('Program ended')
        return

    time.sleep(1)

    return fitness
