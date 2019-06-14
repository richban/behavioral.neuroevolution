import numpy as np
import time
import yaml
from vision.tracker import get_marker_object
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def normalize_1_1(x, min, max):
    return np.array([((2 * ((x[0]-(min))/(max-(min)))) - 1), ((2 * ((x[1]-(min))/(max-(min)))) - 1)])


def normalize_0_1(x, min, max):
    return np.array([(x[0]-(min))/(max-(min)), (x[1]-(min))/(max-(min))])


def interval_map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def normalize(x, x_min, x_max, a=0.0, b=1.0):
    return interval_map(x, x_min, x_max, a, b)


def scale(x, a, b):
    return interval_map(x, 0.0, 1.0, a, b)


def scale_thymio_sensors(x, a, b):
    return interval_map(x, 0.0, 0.1, a, b)


def sensors_offset(distance, minDetection, noDetection):
    return (1 - ((distance - minDetection) / (noDetection - minDetection)))


def f_wheel_center(wheels, min, max):
    return normalize((np.abs((wheels[0]) + (wheels[1])) / 2), min, max)


def f_straight_movements(wheels, min, max):
    return (1 - (np.sqrt(normalize(np.absolute(wheels[0] - wheels[1]), min, max))))


def f_obstacle_dist(sensors):
    return (1 - np.amax(sensors))


def euclidean_distance(points1, points2):
    a = np.array(points1)
    b = np.array(points2)

    if len(a) > len(b):
        return np.linalg.norm(a[:len(b)]-b)

    if len(b) > len(a):
        return np.linalg.norm(a-b[:len(a)])
    # c = a-b
    # return np.sqrt(np.sum(np.square([c]), axis=1))
    return np.linalg.norm(a-b)


def vrep_ports():
    """Load the vrep ports"""
    with open("ports.yml", 'r') as f:
        portConfig = yaml.load(f, Loader=Loader)
    return portConfig['ports']


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print('{}  {:.2f} seconds'.format(
                  method.__name__, (te - ts)))
        return result
    return timed


def f_t_obstacle_avoidance(wheels, sensors, simulation):

    if simulation == 'thymio':
        wheel_center = f_wheel_center(wheels, 0, 200)
        straight_movements = f_straight_movements(wheels, 0, 400)
        obstacles_distance = f_obstacle_dist(sensors)
    elif simulation == 'vrep':
        wheel_center = f_wheel_center(wheels, 0.0, 2.0)
        straight_movements = f_straight_movements(wheels, 0.0, 4.0)
        obstacles_distance = f_obstacle_dist(sensors)

    fitness_t = wheel_center * straight_movements * obstacles_distance

    return (fitness_t, wheel_center, straight_movements, obstacles_distance)


def thymio_position():
    thymio = get_marker_object(7)
    while thymio.realxy() is None:
        thymio = get_marker_object(7)
    return (thymio.realxy()[:2], thymio.orientation())


def flatten_dict(d):
    def expand(key, value):
        if isinstance(value, dict):
            return [(key + '.' + k, v) for k, v in flatten_dict(value).items()]
        else:
            return [(key, value)]

    items = [item for k, v in d.items() for item in expand(k, v)]

    return dict(items)


def calc_behavioral_features(areas_counter,
                             wheel_speeds,
                             sensor_activations,
                             f_path,
                             genome_id,
                             gen,
                             simulation='UNDEFINED'):
    # Compute and store behavioral featuers
    total_steps_in_areas = sum(val['count']
                               for _, val in areas_counter.items())

    for _, value in areas_counter.items():
        try:
            p = value['count']/total_steps_in_areas
        except (ZeroDivisionError, ValueError):
            print(areas_counter)
            p = 0.0
        value.update(
            percentage=p,
            total=total_steps_in_areas
        )

    avg_wheel_speeds = np.mean(np.array(wheel_speeds), axis=0)
    avg_sensors_activation = np.mean(np.array(sensor_activations), axis=0)
    avg_areas = list(flatten_dict(areas_counter).values())

    features_file = np.concatenate(
        (
            [gen],
            [genome_id],
            [simulation],
            avg_wheel_speeds,
            avg_sensors_activation,
            avg_areas
        )
    )

    # return avg_left, avg_right, s1-s7, area0_percentage, area1_percentage, area2_percentage
    features = np.concatenate(
        (
            avg_wheel_speeds,
            avg_sensors_activation,
            np.delete(avg_areas, [0, 2, 3, 5, 6, 8])
        )
    )

    try:
        with open(f_path + 'behavioral_features.dat', 'a') as b:
            np.savetxt(b, (features_file,), delimiter=',', fmt='%s')
    except FileNotFoundError as error:
        print('File not found {}'.format(error))

    return features


def save_debug_data(f_path,
                    genome_id,
                    sensor_activation,
                    norm_sensor_activation,
                    net_output,
                    scaled_output,
                    wheel_center,
                    straight_movements,
                    obstacles_distance,
                    fitness_t,
                    sim_type,
                    robot_current_position=None
                    ):

    if sim_type == 'VREP':
        name = '_vrep_simulation.dat'
    elif sim_type == 'THYMIO':
        name = '_thymio_simulation.dat'
    else:
        name = '_simulation.dat'

    with open(f_path + str(genome_id) + name, 'a') as f:
        f.write('{0!s},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}\n'.format(
            genome_id, net_output[0], net_output[1], scaled_output[0], scaled_output[1],
            np.array2string(
                sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
            np.array2string(
                norm_sensor_activation, precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
            wheel_center, straight_movements, obstacles_distance, np.max(
                norm_sensor_activation), fitness_t, robot_current_position))


def save_moea_data(path, genome):
    with open(path + 'genomes_moea.dat', 'a') as f:
        try:
            (fitness, transferability, diversity) = genome.fitness.values
        except AttributeError:
            (fitness, transferability, diversity) = genome.task_fitness, -1.0, -1.0
        f.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(genome.gen, genome.key, genome.evaluation,
                                                           fitness, transferability, diversity, np.array2string(
                                                               np.array(genome.features), precision=4, formatter={'float_kind': lambda x: "%.4f" % x}),
                                                           np.array2string(
                                                               np.array(genome.position), precision=4, formatter={'float_kind': lambda x: "%.4f" % x})
                                                           ))


def save_fitness_moea(pop, gen, path):
    for ind in pop:
        try:
            (fitness, disparity, diversity) = ind.fitness.values
        except AttributeError:
            (fitness, disparity, diversity) = ind.task_fitness, -1.0, -1.0
        with open(path + 'fitness.dat', 'a') as f:
            f.write('{0},{1},{2},{3},{4}\n'.format(
                gen, ind.key, fitness, disparity, diversity))


def calc_str_disparity(transfered, simulation):
    if len(transfered) > len(simulation):
        t = np.array(transfered[:len(simulation)])
        s = np.array(simulation)
    elif len(simulation) > len(transfered):
        t = np.array(transfered)
        s = np.array(simulation[:len(transfered)])
    else:
        t = np.array(transfered)
        s = np.array(simulation)

    t_mean = np.mean(t, axis=0)
    s_mean = np.mean(s, axis=0)

    x = np.sum((np.power(s.T[0] - t.T[0], 2) / (s_mean[0] * t_mean[0])))
    y = np.sum((np.power(s.T[1] - t.T[1], 2) / (s_mean[1] * t_mean[1])))

    return x + y
