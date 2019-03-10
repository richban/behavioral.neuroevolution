from queue import PriorityQueue
import numpy as np
import ctypes
from math import degrees
import time
import vrep.vrep as vrep


class pid():
    """PID Controller"""

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error = 0.0
        self.error_old = 0.0
        self.error_sum = 0.0
        self.d_error = self.error - self.error_old

    def control(self, error):
        """
        P - Proportional control is adjusting the motor 
            speed by adding the value of the error.
        I - Integral control helps to deliver steady 
            state performance by adjusting for slowly changing errors.
        D - Derivative control looks at how quickly or 
            slowly the error is changing
        adjustment = (error × KP) + (previous_error × KD) + (sum_of_errors × KI)
        """
        self.error = error
        self.error_sum += error
        self.d_error = self.error - self.error_old
        P = self.kp * self.error
        I = self.ki * self.error_sum
        D = self.kd * self.d_error
        self.error_old = self.error
        return P+I+D


def search(grid, init, goal, cost, D=1, fnc='Euclidean', D2=1):
    """A start algorithm"""

    def Euclidean_fnc(current_indx, goal_indx, D=1):
        return np.sqrt(((current_indx[0]-goal_indx[0])**2 + (current_indx[1]-goal_indx[1])**2))

    def Manhattan_fnc(current_indx, goal_indx, D=1):
        dx = np.sqrt((current_indx[0]-goal_indx[0])**2)
        dy = np.sqrt((current_indx[1]-goal_indx[1])**2)
        return D * (dx + dy)

    def Diagonal_fnc(current_indx, goal_indx, D=1):
        dx = np.sqrt((current_indx[0]-goal_indx[0])**2)
        dy = np.sqrt((current_indx[1]-goal_indx[1])**2)
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    if fnc == 'Euclidean':
        hueristic_fnc = Euclidean_fnc
    elif fnc == "Manhattan":
        hueristic_fnc = Manhattan_fnc
    elif fnc == "Diagonal":
        hueristic_fnc = Diagonal_fnc

    def near_obstacles(point, half_kernel=5):
        x_start = int(max(point[0] - half_kernel, 0))
        x_end = int(min(point[0] + half_kernel, grid.shape[0]))
        y_start = int(max(point[1] - half_kernel, 0))
        y_end = int(min(point[1] + half_kernel, grid.shape[1]))
        return np.any(grid[x_start:x_end, y_start:y_end] < 128)

    def delta_gain(gain=1):
        delta = np.array([[-1, 0],  # go up
                          [-1, -1],  # up left
                          [0, -1],  # go left
                          [1, -1],  # down left
                          [1, 0],  # go down
                          [1, 1],  # down right
                          [0, 1],  # go right
                          [-1, 1]  # up right
                          ])
        return delta*gain

    delta = delta_gain(gain=5)
    front = PriorityQueue()
    G = 0
    H = hueristic_fnc(init, goal, D)
    F = G+H
    front.put((F, G, init))
    discovered = []
    discovered.append(init)

    actions = np.ones_like(grid)*-1
    count = 0
    path = []

    def policy_draw(indx):
        indx_old = tuple(indx)
        indx_new = tuple(indx)
        path.append(tuple(goal))
        while indx_new != init:
            indx_new = tuple(np.array(indx_old) -
                             delta[int(actions[indx_old])])
            path.append(indx_new)
            indx_old = indx_new

    while not front.empty():
        front_element = front.get()
        G = front_element[1]
        indx = front_element[2]
        if ((indx[0] >= goal[0]-20) and (indx[0] < goal[0]+20)) and ((indx[1] >= goal[1]-20) and (indx[1] < goal[1]+20)):
            policy_draw(indx)
            print("found goal")
            print(count)
            print(front_element)
            break
        else:
            for y in range(len(delta)):
                indx_new = tuple(indx + delta[y])
                if ((np.any(np.array(indx_new) < 0)) or (indx_new[0] > grid.shape[0]-1) or (indx_new[1] > grid.shape[1]-1)):
                    continue
                if (grid[indx_new] >= 128) and (indx_new not in discovered):
                    count += 1
                    # if the obstacle is inside the robot :D, have a really high cost
                    if near_obstacles(indx_new, half_kernel=35):
                        g_new = G + 1500*cost
                    # if the obstacle is about a robot's length near it , have a high cost
                    elif near_obstacles(indx_new, half_kernel=70):
                        g_new = G + 15*cost
                    # as before
                    elif near_obstacles(indx_new, half_kernel=100):
                        g_new = G + 10*cost
                    # as before
                    elif near_obstacles(indx_new, half_kernel=110):
                        g_new = G + 5*cost
                    else:
                        g_new = G + cost
                    # trying to increase the cost of rapidly changing direction
                    if y == actions[indx]:
                        g_new = g_new
                    elif (y-1) % len(delta) == actions[indx] or (y+1) % len(delta) == actions[indx]:
                        g_new = g_new + 5*cost
                    else:
                        g_new = g_new + 10*cost
                    h_new = hueristic_fnc(indx_new, goal, D)
                    f_new = (g_new + h_new)-0.0001*count
                    front.put((f_new, g_new, indx_new))
                    discovered.append(indx_new)
                    actions[indx_new] = y
    else:
        print(count)
        print("fail")
    return actions, np.array(path[::-1])


def smooth(path, grid, weight_data=0.5, weight_smooth=0.1, tolerance=0.000001, number_of_iter=1e3):
    """Perform path smoothing"""
    newpath = np.copy(path).astype('float64')

    def get_near_obstacles(point, area=5):
        x_start = int(max(point[0] - area, 0))
        x_end = int(point[0] + area)
        y_start = int(max(point[1] - area, 0))
        y_end = int(point[1] + area)
        points = np.argwhere(grid[x_start:x_end, y_start:y_end] < 128)
        points[:, 0] += x_start
        points[:, 1] += y_start
        if not points.size:
            points = point.copy()
        return points

    def near_obstacles(point, half_kernel=2):
        x_start = int(max(point[0] - half_kernel, 0))
        x_end = int(point[0] + half_kernel)
        y_start = int(max(point[1] - half_kernel, 0))
        y_end = int(point[1] + half_kernel)
        return np.any(grid[x_start:x_end, y_start:y_end] < 128)

    error = np.ones(path.shape[0])*tolerance+tolerance
    num_points = path.shape[0]
    for count in range(int(number_of_iter)):
        for i in range(1, num_points-1):
            old_val = np.copy(newpath[i])
            update1 = weight_data*(path[i] - newpath[i])
            update2 = weight_smooth*(newpath[i-1]+newpath[i+1]-2*newpath[i])
            newpath[i] += update1+update2
            if near_obstacles(newpath[i], half_kernel=35):
                newpath[i] = old_val
            error[i] = np.abs(np.mean(old_val-newpath[i]))
        if np.mean(error) < tolerance:
            break
    print(count)
    return newpath


def transform_points_from_image2real(points, scale=1/1000):
    """transform from grid frame to real coordinates"""
    if points.ndim < 2:
        flipped = np.flipud(points)
    else:
        flipped = np.fliplr(points)
    points2send = (flipped*scale)
    return points2send


def transform2robot_frame(r_current_pos, goal_points, theta):
    """all coordinates must first be transformed to vehicle
    coordinates in order for the algorithm to work properly
    xgv = (xg – xr)cos(Φ) + (yg-yr)sin(Φ)
    ygv = -(xg – xr)sin(Φ) + (yg-yr)cos(Φ)
    (xgv,ygv) is the goal point in vehicle coordinates and Φ is the current vehicle heading
    """
    r_current_pos = np.asarray(r_current_pos)
    goal_points = np.asarray(goal_points)
    T_matrix = np.array([[np.cos(theta), np.sin(theta)],
                         [-1*np.sin(theta), np.cos(theta)], ])
    trans = goal_points - r_current_pos
    if trans.ndim >= 2:
        trans = trans.T
        point_t = np.dot(T_matrix, trans).T
    else:
        point_t = np.dot(T_matrix, trans)
    return point_t


def is_near(robot_center, point, dist_thresh=0.025):
    dist = np.sqrt((robot_center[0]-point[0]) **
                   2 + (robot_center[1]-point[1])**2)
    return dist <= dist_thresh


def pioneer_robot_model(v_des, omega_des, w_axis, w_radius):
    """ v_des - desired velocity
        omega_des - desired rotation
    """
    v_r = (v_des+w_axis*omega_des)
    v_l = (v_des-w_axis*omega_des)

    omega_right = v_r/w_radius
    omega_left = v_l/w_radius

    return omega_right, omega_left


def send_path_4_drawing(path, sleep_time=0.07, clientID=0):
    """ send path to VREP; the bigger the sleep time the 
        more accurate the points are placed but yo
    """
    for i in path:
        point2send = transform_points_from_image2real(i, 4/1000)
        packedData = vrep.simxPackFloats(point2send.flatten())
        raw_bytes = (ctypes.c_ubyte * len(packedData)
                     ).from_buffer_copy(packedData)
        _ = vrep.simxWriteStringStream(
            clientID, "path_coord", raw_bytes, vrep.simx_opmode_oneshot)
        time.sleep(sleep_time)


def get_distance(points1, points2):
    return np.sqrt(np.sum(np.square(points1 - points2), axis=1))


def transform_pos_angle(position, orientation):
    (x, y) = position
    pos = [x*4, y*4, 0.1388]
    angle = [0, 0, degrees(orientation)]
    return pos, angle


def follow_path(robot, init_position, get_marker_object, vrep, clientID):
    try:
        print('Back to initial position!')
        robot.t_stop()
        grid = np.full((880, 1190), 255)
        lad = 0.09  # look ahead distance in meters (m)
        wheel_axis = 0.11  # wheel axis distance in meters (m)
        wheel_radius = 0.02  # wheel radius in meters (m)
        _, look_ahead_sphere = vrep.simxGetObjectHandle(
            clientID, 'look_ahead', vrep.simx_opmode_oneshot_wait)
        indx = 0
        theta = 0.0
        count = 0
        om_sp = 0
        d_controller = pid(kp=0.5, ki=0, kd=0)
        omega_controller = pid(kp=0.5, ki=0, kd=0)

        robot_m = get_marker_object(7)
        while robot_m.realxy() is None:
            # obtain current position of the robot
            robot_m = get_marker_object(7)

        # transform robot position to grid system
        robot_current_position = (robot_m.realxy()[:2]*1000).astype(int)

        # transform goal position to grid system
        goal_position = (init_position*1000).astype(int)

        # set position of the robot in simulator
        position, orientation = transform_pos_angle(robot_m.realxy()[:2],
                                                    (2*np.pi - robot_m.orientation()))
        robot.v_set_pos_angle(position, orientation)

        # Search for the path in grid system
        _, path = search(grid,
                         (robot_current_position[1],
                          robot_current_position[0]),
                         (goal_position[1],
                          goal_position[0]),
                         cost=1,
                         D=0.5,
                         fnc='Manhattan')

        # Path smoothing
        newpath = smooth(path,
                         grid,
                         weight_data=0.1,
                         weight_smooth=0.6,
                         number_of_iter=1000)

        # transform GRID points to  real (x, y) coordinates
        path_to_track = transform_points_from_image2real(newpath)

        # Send data to VREP
        send_path_4_drawing(newpath, 0.05, clientID)

        # transform GRID goal to real (x, y) coordinates
        goal_position = init_position

        while not is_near(robot_current_position, goal_position, dist_thresh=0.05):
            # get robot marker
            robot_m = get_marker_object(7)
            if robot_m.realxy() is not None:
                # update current position of the robot
                robot_current_position = robot_m.realxy()[:2]

            # calculate robot orientation
            theta = 2*np.pi - robot_m.orientation()
            theta = np.arctan2(np.sin(theta), np.cos(theta))

            # update position and orientation of the robot in vrep
            position, orientation = transform_pos_angle(
                robot_current_position, theta)
            robot.v_set_pos_angle(position, orientation)

            # path transformation to vehicle coordinates; relative to the robot
            path_transformed = transform2robot_frame(
                robot_current_position, path_to_track, theta)

            # get distance of each carrot point; relative to the robots
            dist = get_distance(path_transformed, np.array([0, 0]))

            # loop to determine which point will be the carrot/goal point
            for i in range(dist.argmin(), dist.shape[0]):
                if dist[i] < lad and indx <= i:
                    indx = i

            # mark the carrot with the sphere
            _ = vrep.simxSetObjectPosition(
                clientID,
                look_ahead_sphere,
                -1,
                (path_to_track[indx, 0]*4,
                 path_to_track[indx, 1]*4,
                 0.005),
                vrep.simx_opmode_oneshot
            )
            # orientation error relative to the robot
            orient_error = np.arctan2(
                path_transformed[indx, 1], path_transformed[indx, 0])
            # PID controller; desired velocity and rotation
            v_sp = d_controller.control(dist[indx])
            om_sp = omega_controller.control(orient_error)
            vr, vl = pioneer_robot_model(v_sp, om_sp, wheel_axis, wheel_radius)

            robot.t_set_motors(vl*40, vr*40)
            count += 1
        else:
            print('GOAAAAAAALL !!')
            print('robot_position: ', robot_current_position)
            print('robot_goal: ', goal_position)
            robot.t_stop()
    finally:
        if (vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot) == -1):
            print('Failed to stop the simulation\n')
            print('Program ended\n')
            return
        time.sleep(1)
