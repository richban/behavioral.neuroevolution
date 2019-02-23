import numpy as np
import json
import cv2
import os.path


def get_transform(markers, poses):
    """Process marker and pose arrays containing matching points
    and return the resulting transformation matrix

    Keyword arguments:
    markers -- marker array with indentifiers (reference points)
    poses   -- poses (x, y, z)
    """
    return get_transform_lstsqr(markers, poses)


def get_transform_lstsqr(markers, poses):
    markers.sort(key=lambda m: m.mid)

    a = np.array([[m.center()[0], m.center()[1], 1.0]
                  for m in markers if (m.mid <= 4 and m.mid > 0)])
    b = np.array([(p[:2] + [1.0]) for p in poses])

    t, retval = cv2.findHomography(a, b)
    t /= t[2, 2]

    return t


def save(transform, poses, height, cfile='./vision/cdata.json'):

    if(os.path.isfile(cfile)):
        with open(cfile, 'w') as f:
            obj = {'transform': transform.tolist(),
                   'poses': poses, 'height': height}
            json.dump(obj, f)
    else:
        cfile = "./vision" + cfile
        with open(cfile, 'w') as f:
            obj = {'transform': transform.tolist(),
                   'poses': poses, 'height': height}
            json.dump(obj, f)


def restore(cfile='./vision/cdata.json'):
    if(os.path.isfile(cfile)):
        with open(cfile, 'r') as f:
            data = json.load(f)
    else:
        cfile = "./vision" + cfile
        with open(cfile, 'r') as f:
            data = json.load(f)
    return (np.array(data['transform']), data['poses'], data['height'])


def redo_transform(markers):
    # position of markers in arena
    poses = [[0.0, 0.0, 0.0],
             [0.0, 0.8, 0.0],
             [1.19, 0.8, 0.0],
             [1.19, 0.0, 0.0]]

    transform = get_transform(markers, poses)
    height = np.average(poses, 0)[2]
    save(transform, poses, height)
    return transform, height
