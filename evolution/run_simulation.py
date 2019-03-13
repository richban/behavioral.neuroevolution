from robot.vrep_robot import VrepRobot
from evolution.eval_genomes import eval_genomes_simulation
from functools import partial
import vrep.vrep as vrep
import neat


def run_vrep_simluation(config_file, settings):
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
        chromosome=None
    )

    # Run for up to N_GENERATIONS generations.
    winner = p.run(partial(eval_genomes_simulation, settings, individual), 2)

    return stats, winner
