from vision.tracker import Tracker, get_marker_object, get_markers
from robot.thymio_robot import ThymioII
from datetime import datetime, timedelta
from vision.calibration import restore

def thymio(name='thymio-II'):

    robot = ThymioII(name)

    now = datetime.now()

    robot.t_set_motors(-80, -80)

    while datetime.now() - now < timedelta(seconds=10):
        # print(robot.t_read_prox())
        continue

    robot.t_stop()


if __name__ == "__main__":

    transform = restore()
    vision_thread = Tracker(mid=5,
                            transform=None,
                            mid_aux=0,
                            video_source=-1,
                            capture=False,
                            show=True,
                            debug=True
                            )

    vision_thread.start()

    thymio()
    for i in range(10):
        marker = get_marker_object(9)
        if marker is not None:
            print(marker.realxy())
