import os
import neat
from datetime import datetime, timedelta
import time
import visualize
from evolved_thymio import EvolvedThymio

RUNTIME = 10
N_GENERATIONS = 2


def run(config_file):
    print('Evolutionary program started!')
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:

            individual = EvolvedThymio('thymio-II', genome)
            individual.stop()
            time.sleep(1)
            now = datetime.now()

            net = neat.nn.FeedForwardNetwork.create(genome, config)

            while datetime.now() - now < timedelta(seconds=RUNTIME):

                input = individual.check_prox()
                output = net.activate(input)
                scaled = list(map(lambda x: x*500, output))
                individual.set_motor(*list(scaled))

            individual.stop()
            time.sleep(1)
            genome.fitness = 1.0

    # Run for up to N_GENERATIONS generations.
    winner = p.run(eval_genomes, N_GENERATIONS)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', -5: 'E',
                  -6: 'F', -7: 'G', 0: 'LEFT', 1: 'RIGHT', }
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')
    run(config_path)
