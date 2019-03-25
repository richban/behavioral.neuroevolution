import numpy as np


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


def f_wheel_center(wheels):
    return normalize((((wheels[0]) + (wheels[1])) / 2), -2.0, 2.0)


def f_straight_movements(wheels):
    return (1 - (np.sqrt(normalize(np.absolute(wheels[0] - wheels[1]), 0.0, 4.0))))


def f_obstacle_dist(sensors):
    s = np.array([interval_map(s, 0.1, 0.0, 0.0, 1.0)
                  for s in sensors])
    if np.all(s == s[0]):
        return 1
    else:
        return (1 - np.amax(s))


def euclidean_distance(points1, points2):
    a = np.array(points1)
    b = np.array(points2)
    c = a - b
    return np.sqrt(np.sum(np.square([c]), axis=1))
