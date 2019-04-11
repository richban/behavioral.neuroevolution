from __future__ import print_function

import copy
import warnings

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import csv


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())
    median_fitness = np.array(statistics.get_fitness_median())

    plt.figure(figsize=(12, 9))

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.plot(generation, avg_fitness, 'b-', label="average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")
    plt.plot(generation, median_fitness, 'y-', label="median")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig = plt.figure()
    plt.subplot(4, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(4, 1, 2)
    plt.ylabel("Fired")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, f_values, "r-")

    plt.subplot(4, 1, 3)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(4, 1, 4)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    plt.close()
    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    plt.figure(figsize=(12, 9))

    _, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn(
            "This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box'}
        input_attrs['fillcolor'] = node_colors.get(k, 'lightgray')
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled'}
        node_attrs['fillcolor'] = node_colors.get(k, 'lightblue')

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add((cg.in_node_id, cg.out_node_id))

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={
                     'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot


def plot_species_stagnation(body, imgfilename):
    body = body[3:-2]
    stagnation = []
    id = []
    fitness = []
    size = []
    adj_fit = []
    age = []
    for line in body:
        line = line.split(' ')
        line = [x for x in line if x]
        line[-1] = line[-1].strip()
        id.append(line[0])
        age.append(line[1])
        size.append(line[2])
        fitness.append(line[3])
        adj_fit.append(line[4])
        stagnation.append(line[5])

    if len(id) < 2:
        return None

    plt.figure(figsize=(12, 9))

    stagnation = np.array(stagnation).astype(np.float)
    id = np.array(id)
    points = plt.bar(id, stagnation, width=0.7)

    for ind, bar in enumerate(points):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, 'fit {} / size {}'.format(fitness[ind], size[ind]),
                 ha='center', va='bottom', rotation=90, fontsize=7)
    plt.ylabel('stagnation')
    plt.xlabel('Species ID')
    plt.axis([0, plt.axis()[1], 0, plt.axis()[3] + 20])
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(imgfilename)
    plt.clf()
    plt.close('all')
    return imgfilename


def plot_fitness_over_gen(file, imgfilename):
    with open(file, 'r') as csvfile:
        data = csv.reader(csvfile)

        gen = []
        avg_fit = []
        stdv = []
        max_fit = []
        median = []
        for row in data:
            gen.append(int(row[0]))
            avg_fit.append(float(row[1]))
            stdv.append(float(row[2]))
            max_fit.append(float(row[3]))
            median.append(float(row[4]))

    if len(gen) < 2:
        return None

    plt.figure(figsize=(12, 9))

    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    plt.grid()
    plt.plot(gen, avg_fit, 'b', linewidth=0.5,)
    plt.plot(gen, stdv, 'g', linewidth=0.5,)
    plt.plot(gen, max_fit, 'r', linewidth=0.5,)
    plt.plot(gen, median, 'y', linewidth=0.5,)

    plt.plot(gen, max_fit, 'r', markersize=5, label='Max fitness')
    plt.plot(gen, avg_fit, 'b', markersize=5, label='Average fitness')
    plt.plot(gen, stdv, 'g', markersize=5, label='Standard deviation')
    plt.plot(gen, median, 'y', markersize=5, label='Median')

    plt.ylabel('Fitness')
    plt.xlabel('Generation')

    # xmin, xmax, ymin, ymax = plt.axis()
    # plt.axis([xmin, xmax, ymin, ymax])
    plt.legend(bbox_to_anchor=(1, 1), loc='best')
    plt.tight_layout()

    plt.savefig(imgfilename)
    plt.clf()
    plt.close('all')
    return imgfilename
