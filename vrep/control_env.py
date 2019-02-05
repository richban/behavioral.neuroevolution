import vrep
import time
import array
import traceback
import numpy as np


OP_MODE = vrep.simx_opmode_oneshot_wait
PORT_NUM = 19997


def create_object(client_id, display_name, transform=None, parent_handle=-1, debug=None, operation_mode=vrep.simx_opmode_blocking):
    """Create a dummy object in the simulation
    # Arguments
        transform_display_name: name string to use for the object in the vrep scene
        transform: 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
        parent_handle: -1 is the world frame, any other int should be a vrep object handle
    """
    if transform is None:
        transform = np.array([0., 0., 0., 0., 0., 0., 1.])
    empty_buffer = bytearray()
    res, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
        client_id,
        'remoteApiCommandServer',
        vrep.sim_scripttype_childscript,
        'createObject_function',
        [parent_handle],
        transform,
        [display_name],
        empty_buffer,
        operation_mode)
    if res == vrep.simx_return_ok:
        if debug is not None and 'print_transform' in debug:
            print('Dummy name:', display_name, ' handle: ',
                  ret_ints[0], ' transform: ', transform)
    else:
        print('create_object remote function call failed.')
        print(''.join(traceback.format_stack()))
        return -1
    return ret_ints[0]


def set_pose(client_id, object_handle, transform=None, operation_mode=vrep.simx_opmode_oneshot_wait):
    """Set the pose of an object in the simulation
        transform: 3 cartesian (x, y, z) and 4 quaternion (x, y, z, w) elements, same as vrep
        parent_handle: -1 is the world frame, any other int should be a vrep object handle
    """
    if transform is None:
        transform = np.array([0., 0., 0., 0., 0., 0., 1.])
    res = vrep.simxSetObjectPosition(
        client_id,
        object_handle,
        -1,
        transform,
        operation_mode)
    if res == vrep.simx_return_ok:
        print('SetPose object:', object_handle, ' position: ', transform)
    else:
        print('setPose remote function call failed.')
        print(''.join(traceback.format_stack()))
        return -1
    return transform


def get_pose(client_id, object_handle, operation_mode=vrep.simx_opmode_oneshot_wait):
    """Get the pose of an object in the simulation"""
    res, position = vrep.simxGetObjectPosition(
        client_id,
        object_handle,
        -1,
        operation_mode)
    if res == vrep.simx_return_ok:
        print('Get Pose object:', object_handle, ' position: ', position)
    else:
        print('Get Pose remote function call failed.')
        print(''.join(traceback.format_stack()))
        return -1
    return position


def get_object_handle(client_id, display_name, operation_mode=vrep.simx_opmode_oneshot_wait):
    """get object handler"""
    res, obj_handle = vrep.simxGetObjectHandle(
        client_id,
        display_name,
        operation_mode)
    if res == vrep.simx_return_ok:
        print('Get object handle: ', obj_handle)
    else:
        print('get_object_handle remote function call failed.')
        print(''.join(traceback.format_stack()))
        return -1
    return obj_handle


def vrep_print(client_id, message):
    """Print a message in both the python command line and on the V-REP Statusbar.
    The Statusbar is the white command line output on the bottom of the V-REP GUI window.
    """
    vrep.simxAddStatusbarMessage(client_id, message, vrep.simx_opmode_oneshot)
    print(message)


def exec_function(client_id, code):
    """send code string to vrep to execute some function"""
    empty_buffer = bytearray()
    res, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(
        client_id,
        'remoteApiCommandServer',
        vrep.sim_scripttype_childscript,
        'executeCode_function',
        [],
        [],
        [code],
        empty_buffer,
        vrep.simx_opmode_blocking)
    if res == vrep.simx_return_ok:
        print('Code execution returned: ', ret_ints[0])
    else:
        print('setPose remote function call failed.')
        print(''.join(traceback.format_stack()))
        return -1
    return ret_ints[0]


def run():
    # Just in case, close all opened connections
    vrep.simxFinish(-1)

    client_id = vrep.simxStart(
        '127.0.0.1',
        PORT_NUM,
        True,
        True,
        5000,
        5)  # Connect to V-REP

    if client_id == -1:
        print('Failed connecting to remote API server')
        print('Program ended')
        return

    if (vrep.simxStartSimulation(client_id, OP_MODE) == -1):
        print('Failed to start the simulation\n')
        print('Program ended\n')
        return

    # move already created object in vrep
    cuboid = get_object_handle(client_id, 'Cuboid')

    # get cuboid position
    get_pose(client_id, cuboid)

    # set new position
    set_pose(client_id, cuboid, [0.2, 0.4, 0.0])

    # create dummy object
    dummy = create_object(client_id, 'Apple', [0.1, 0.2, 0.3])

    time.sleep(10)

    if (vrep.simxStopSimulation(client_id, OP_MODE) == -1):
        print('Failed to stop the simulation\n')
        return

    vrep.simxFinish(client_id)


if __name__ == '__main__':
    run()
