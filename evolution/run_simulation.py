from robot.vrep_robot import VrepRobot
from evolution.eval_genomes import eval_genomes_simulation, eval_genomes_hardware
try:
    from robot.evolved_robot import EvolvedRobot
except ImportError as error:
    print(error.__class__.__name__ + ": " + 'DBus works only on linux!')
from functools import partial
import vrep.vrep as vrep
import neat


def run_vrep_simluation(settings, config_file):
    print('Neuroevolutionary program started!')
    vrep.simxFinish(-1)
    settings.client_id = vrep.simxStart(
        '127.0.0.1',
        settings.port_num,
        True,
        True,
        5000,
        5)

    if settings.client_id == -1:
        return

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)

    individual = VrepRobot(
        client_id=settings.client_id,
        id=None,
        op_mode=settings.op_mode,
        robot_type=settings.robot_type
    )

    # Run for up to N_GENERATIONS generations.
    winner = p.run(partial(eval_genomes_simulation,
                           individual, settings), settings.n_gen)

    return config, stats, winner


def run_hardware_simulation(settings, config_file):
    print('Neuroevolutionary program started!')
    vrep.simxFinish(-1)
    settings.client_id = vrep.simxStart(
        settings.address,
        settings.port_num,
        True,
        True,
        5000,
        5)  # Connect to V-REP

    if settings.client_id == -1:
        print('Failed connecting to remote API server')
        print('Program ended')
        return

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)

    individual = EvolvedRobot(
        'thymio-II',
        client_id=settings.client_id,
        id=None,
        op_mode=settings.op_mode,
        chromosome=None,
        robot_type=settings.robot_type
    )

    # Run for up to N_GENERATIONS generations.
    winner = p.run(partial(eval_genomes_hardware,
                           individual, settings), settings.n_gen)

    return config, stats, winner
