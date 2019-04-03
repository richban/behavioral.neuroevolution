from evolution.simulation import Simulation
from argparse import ArgumentParser
from settings import Settings
from evolution.eval_genomes import \
    eval_genomes_simulation, \
    eval_genomes_hardware, \
    eval_genome
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


if __name__ == '__main__':
    local_dir = os.path.abspath('evolution')
    config = os.path.join(local_dir, 'config_thymio.ini')
    settings = Settings(thymio, True)

    parser = ArgumentParser(
        description='I am here to guide you through the evolution!')
    parser.add_argument('-simulation', choices=['vrep', 'thymio'],
                        description='Run the evolution in vrep \
                        simulator or on the physical robot (thymio).',
                        required=True)
    parser.add_argument('-threaded', type=bool, default=False, choices=[
                        True, False], description='Runs evolution using threads only works in vrep.')
    parser.add_argument('-restore_genome', type=str,
                        description='Restore a specific genome. Path to the genome is required!')
    parser.add_argument('-checkpoint', type=str,
                        description='Restore evolution from a checkpoint. Path to the checkpoint is required!')
    parser.add_argument('-config', type=str,
                        description='Load specific NEAT configuration file. Path to the config is required!')

    args = parser.parse_args()

    kwargs = {'config_file': config}
    if args.simulation == 'vrep':
        kwargs.update({'simulation_type': 'vrep'})

        if args.thread and not args.restore_genome:
            kwargs.update({'eval_function': eval_genome})
            kwargs.update({'threaded': True})
        else:
            kwargs.update({'eval_function': eval_genomes_simulation})

        if args.restore_genome not args.thread:
            kwargs.update({'genome_path': args.restore_genome})

        if args.checkpoint:
            kwargs.update({'checkpoint': args.checkpoint})

        if args.config:
            kwargs.update({'config_file': args.config})
    else:
        kwargs.update({'simulation_type': 'thymio'})
        kwargs.update({'eval_function': eval_genomes_hardware})

        if args.restore_genome:
            kwargs.update({'genome_path': args.restore_genome})

        if args.checkpoint:
            kwargs.update({'checkpoint': args.checkpoint})

        if args.config:
            kwargs.update({'config_file': args.config})

    simulation = Simulation(settings, **kwargs)
