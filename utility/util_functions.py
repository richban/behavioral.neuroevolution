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
    return normalize((((wheels[0]) + (wheels[1])) / 2), min, max)


def f_straight_movements(wheels, min, max):
    return (1 - (np.sqrt(normalize(np.absolute(wheels[0] - wheels[1]), min, max))))


def f_obstacle_dist(sensors):
    return (1 - np.amax(sensors))


def euclidean_distance(points1, points2):
    a = np.array(points1)
    b = np.array(points2)
    c = a - b
    return np.sqrt(np.sum(np.square([c]), axis=1))


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
        wheel_center = f_wheel_center(wheels, -200, 200)
        straight_movements = f_straight_movements(wheels, 0, 400)
        obstacles_distance = f_obstacle_dist(sensors)
    elif simulation == 'vrep':
        wheel_center = f_wheel_center(wheels, -2.0, 2.0)
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
