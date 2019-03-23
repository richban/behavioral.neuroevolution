from evolution.run_simulation import run_vrep_simluation, run_hardware_simulation, restore_vrep_simulation
from utility.evolution import log_statistics, visualize_results
from vision.tracker import Tracker
from settings import Settings
import time
import sys
import os

thymio = {
    'name': 'thymio',
    'body': 'Thymio',
    'left_motor': 'leftMotor',
    'right_motor': 'rightMotor',
    'sensor': 'Proximity_sensor',
    'num_sensors': 7
}

pd3x = {
    'name': 'pd3x',
    'body': 'Pioneer_p3dx',
    'left_motor': 'Pioneer_p3dx_leftMotor',
    'right_motor': 'Pioneer_p3dx_rightMotor',
    'sensor': 'Pioneer_p3dx_ultrasonicSensor',
    'num_sensors': 16
}


def start_vision():
    vision_thread = Tracker(mid=5,
                            transform=None,
                            mid_aux=0,
                            video_source=-1,
                            capture=False,
                            show=True,
                            debug=False,
                            )
    vision_thread.start()

    while vision_thread.cornersDetected is not True:
        time.sleep(2)


def run_hardware(settings, config_file):
    start_vision()
    config, stats, winner = run_hardware_simulation(settings, config_file)
    log_statistics(stats, winner, settings.path)
    visualize_results(config, stats, winner, settings.path)


def run_simulation(settings, config_file):
    config, stats, winner = run_vrep_simluation(settings, config_file)
    log_statistics(stats, winner, settings.path)
    visualize_results(config, stats, winner, settings.path)


def restore_simulation(settings, config_file, checkpoint, path):
    if checkpoint:
        config, stats, winner = restore_vrep_simulation(
            settings, config_file, checkpoint, path)
        log_statistics(stats, winner, settings.path)
        visualize_results(config, stats, winner, settings.path)
    restore_vrep_simulation(
        settings, config_file, checkpoint, path)


if __name__ == '__main__':
    local_dir = os.path.abspath('evolution')
    try:
        if (sys.argv[1] == 'vrep' and sys.argv[2] == 'pd3x'):
            config = os.path.join(local_dir, 'config_pd3x.ini')
            settings = Settings(pd3x)
            run_simulation(settings, config)

        elif (sys.argv[1] == 'vrep' and sys.argv[2] == 'thymio'):
            config = os.path.join(local_dir, 'config_thymio.ini')
            settings = Settings(thymio, True)
            run_simulation(settings, config)

        elif (sys.argv[1] == 'hw' and sys.argv[2] == 'thymio'):
            config = os.path.join(local_dir, 'config_thymio.ini')
            settings = Settings(thymio)
            run_hardware(settings, config)
        elif (sys.argv[1] == 'restore'):
            data = os.path.abspath('data/neat/')
            if sys.argv[2] == 'checkpoint':
                checkpoint = sys.argv[3]
                config = os.path.join(local_dir, 'config_thymio.ini')
                settings = Settings(thymio)
                restore_simulation(settings, config, checkpoint, None)
            elif sys.argv[2] == 'file':
                date = sys.argv[3]
                path = os.path.join(data, date)
                config = os.path.join(local_dir, 'config_thymio.ini')
                settings = Settings(thymio)
                restore_simulation(settings, config, None, path)
        else:
            print('Error!')
    except IndexError:
        print('Wrong Arguments!')
