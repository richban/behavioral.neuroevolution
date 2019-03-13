import os
from settings import Settings
from evolution.run_simulation import run_vrep_simluation

if __name__ == '__main__':
    # Determine path to configuration file.
    local_dir = os.path.abspath('evolution')
    config = os.path.join(local_dir, 'config.ini')

    settings = Settings()

    run_vrep_simluation(settings, config)
