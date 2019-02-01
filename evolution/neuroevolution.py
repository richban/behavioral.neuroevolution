import os
import neat


def run(config_file):
    pass


if __name__ == '__main__':
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')
    run(config_path)
