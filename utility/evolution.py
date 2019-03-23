import utility.visualize as visualize
import pickle


def log_statistics(stats, winner, path):
    # Write run statistics to file.
    stats.save_genome_fitness(filename=path+'fitnesss_history.csv')
    stats.save_species_count(filename=path+'speciation.csv')
    stats.save_species_fitness(filename=path+'species_fitness.csv')

    # log the winner network
    with open(path + 'winner_network.txt', 'w') as s:
        s.write('\nBest genome:\n{!s}'.format(winner))
        s.write('\nBest genomes:\n{!s}'.format(print(stats.best_genomes(5))))

    # Save the winner.
    with open(path + 'winner_genome', 'wb') as f:
        pickle.dump(winner, f)


def visualize_results(config, stats, winner,  path):
    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', -5: 'E',
                  -6: 'F', -7: 'G', 0: 'LEFT', 1: 'RIGHT', }

    visualize.draw_net(config, winner, True, node_names=node_names,
                       filename=path+'network')

    visualize.plot_stats(stats, ylog=False, view=False,
                         filename=path+'feedforward-fitness.svg')
    visualize.plot_species(
        stats, view=False, filename=path+'feedforward-speciation.svg')

    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename=path+'winner-feedforward.gv')
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename=path+'winner-feedforward-enabled.gv', show_disabled=False)
    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename=path+'winner-feedforward-enabled-pruned.gv', show_disabled=False, prune_unused=False)
