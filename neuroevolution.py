from evolution.simulation import Simulation
from argparse import ArgumentParser
from settings import Settings
from evolution.eval_genomes import \
    eval_genomes_simulation, \
    eval_genomes_hardware, \
    post_eval_genome, \
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

    parser = ArgumentParser(
        description='I am here to guide you through the evolution!')
    parser.add_argument('-simulation', choices=['vrep', 'thymio'],
                        help='Run the evolution in vrep \
                        simulator or on the physical robot (thymio).',
                        required=True)
    parser.add_argument('-threaded', type=bool, default=False, choices=[
                        True, False], help='Runs evolution using threads only works in vrep.')
    parser.add_argument('-restore_genome', type=str,
                        help='Restore a specific genome. Path to the genome is required!')
    parser.add_argument('-checkpoint', type=str,
                        help='Restore evolution from a checkpoint. Path to the checkpoint is required!')
    parser.add_argument('-config', type=str,
                        help='Load specific NEAT configuration file. Path to the config is required!')
    parser.add_argument('-headless', type=bool, default=False,
                        help='Run vrep in headless mode.')
    parser.add_argument('-generations', type=int, default=10,
                        help='Number of generations.')
    parser.add_argument('-save_data', type=bool, default=False,
                        help='Number of generations.')
    parser.add_argument('-post_eval', type=bool, default=False,
                        help='Run postevaluation on the genome')
    parser.add_argument('-debug', type=bool, default=False,
                        help='Log fitness, inputs, etc.')

    args = parser.parse_args()
    kwargs = {'config_file': config}
    simulation = 'simulation'
    settings = Settings(thymio, args.save_data, args.debug, args.generations)

    if args.simulation == 'vrep':
        kwargs.update({'simulation_type': 'vrep'})

        if args.threaded and not args.restore_genome:
            kwargs.update({'eval_function': eval_genome})
            kwargs.update({'threaded': True})
            simulation = 'simulation_parallel'
        else:
            kwargs.update({'eval_function': eval_genomes_simulation})

        if args.restore_genome and not args.threaded:
            kwargs.update({'genome_path': args.restore_genome})
            simulation = 'simulation_genome'

        if args.restore_genome and args.post_eval:
            kwargs.update({'genome_path': args.restore_genome})
            kwargs.update({'eval_function': post_eval_genome})
            simulation = 'post_eval'

        if args.checkpoint:
            kwargs.update({'checkpoint': args.checkpoint})

        if args.config:
            kwargs.update({'config_file': args.config})

        if args.headless:
            kwargs.update({'headless': args.headless})
    else:
        kwargs.update({'simulation_type': 'thymio'})
        kwargs.update({'eval_function': eval_genomes_hardware})

        if args.restore_genome:
            kwargs.update({'genome_path': args.restore_genome})
            simulation = 'simulation_genome'

        if args.checkpoint:
            kwargs.update({'checkpoint': args.checkpoint})

        if args.config:
            kwargs.update({'config_file': args.config})

    sim = Simulation(settings, **kwargs)
    sim.start(simulation)

    if simulation != 'simulation_genome' and simulation != 'post_eval':
        sim.log_statistics()
        sim.visualize_results()
