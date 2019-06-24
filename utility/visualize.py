from __future__ import print_function

import os
import csv
import graphviz
import numpy as np
import plotly.graph_objs as go
import plotly
import plotly.plotly as py
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import copy
import warnings
import matplotlib as mpl
mpl.use('TkAgg')

plotly.tools.set_credentials_file(username=os.environ['PLOTLY_USERNAME'],
                                  api_key=os.environ['PLOTLY_API_KEY'])


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


def plot_single_run_scatter(scatter, dt, title):
    """Plots a single run with MAX, AVG, MEDIAN, All individuals"""
    l = []
    y = []

    N = len(scatter.gen.unique())

    c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

    for i in range(int(N)):
        subset = scatter.loc[scatter['gen'] == i]

        trace0 = go.Scatter(
            x=subset.loc[:, 'gen'],
            y=subset.loc[:, 'fitness'],
            mode='markers',
            marker=dict(size=7,
                        line=dict(width=1),
                        color=c[i],
                        opacity=0.5
                        ),
            name='gen {}'.format(i),
            text=subset.loc[:, 'genome']
        )
        l.append(trace0)

    trace0 = go.Scatter(
        x=dt.loc[:, 'gen'],
        y=dt.loc[:, 'max'],
        mode='lines',
        name='Max',
        line=dict(
            color="rgb(204, 51, 51)",
            dash="solid",
            shape="spline",
            smoothing=1.0,
            width=2
        ),
    )

    trace1 = go.Scatter(
        x=dt.loc[:, 'gen'],
        y=dt.loc[:, 'median'],
        mode='lines',
        name='Median',
        line=dict(
            color="rgb(173, 181, 97)",
            shape="spline",
            dash="solid",
            smoothing=1.0,
            width=2
        )
    )

    trace2 = go.Scatter(
        x=dt.loc[:, 'gen'],
        y=dt.loc[:, 'avg'],
        mode='lines',
        name='Average',
        line=dict(
            color="rgb(62, 173, 212)",
            shape="spline",
            dash="solid",
            smoothing=1.0,
            width=2
        )
    )

    data = [trace0, trace1, trace2]

    layout = go.Layout(
        title='Fitness of Population Individuals - {}'.format(title),
        hovermode='closest',
        xaxis=dict(
            title='Generations',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Fitness',
            ticklen=5,
            gridwidth=1,
        ),
        showlegend=False
    )

    fig = go.Figure(data=data+l, layout=layout)

    return py.iplot(fig, filename='single-run-scater-line-plot', layout=layout)


def _set_plot_params(title, ratio):
    # Optionally fix the aspect ratio
    if ratio:
        plt.figure(figsize=plt.figaspect(ratio))

    mpl.style.use('seaborn-dark-palette')

    if title:
        plt.title(title)


def _save_or_show(save):
    if save:
        plt.savefig(save)
    else:
        plt.show()

    # exit()


def plot_single_run(gen, fit_mins, fit_avgs, fit_maxs, title=None, ratio=None, save=None):
    _set_plot_params(title, ratio)

    line1 = plt.plot(gen, fit_mins, 'C1:', label="Minimum Fitness")
    line2 = plt.plot(gen, fit_avgs, "C2-", label="Average Fitness")
    line3 = plt.plot(gen, fit_maxs, "C3:", label="Max Fitness")

    lns = line1 + line2 + line3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="lower right")

    _save_or_show(save)


def plot_runs(dt, title, offline=True):
    """Plots the Max/Average/Median"""
    trace0 = go.Scatter(
        x=dt.index,
        y=dt.loc[:, 'max'],
        mode='lines',
        name='Max',
        line=dict(
            color="rgb(204, 51, 51)",
            dash="solid",
            shape="spline",
            smoothing=0.0,
            width=2
        ),
    )

    trace1 = go.Scatter(
        x=dt.index,
        y=dt.loc[:, 'median'],
        mode='lines',
        name='Median',
        line=dict(
            color="rgb(173, 181, 97)",
            shape="spline",
            dash="solid",
            smoothing=0.0,
            width=2
        )
    )

    trace2 = go.Scatter(
        x=dt.index,
        y=dt.loc[:, 'avg'],
        mode='lines',
        name='Average',
        line=dict(
            color="rgb(62, 173, 212)",
            shape="spline",
            dash="solid",
            smoothing=0.0,
            width=2
        )
    )

    layout = go.Layout(
        showlegend=True,
        hovermode='closest',
        title=title,
        xaxis=dict(
            autorange=False,
            range=[0, 20],
            showspikes=False,
            title="Generations",
            ticklen=5,
            gridwidth=1,
        ),
        yaxis=dict(
            autorange=True,
            title="Fitness",
            ticklen=5,
            gridwidth=1,
        ),
    )

    data = [trace0, trace1, trace2]
    fig = go.Figure(data, layout=layout)

    return py.iplot(fig, filename=title)

    l = []
    y = []

    N = len(scatter.gen.unique())

    c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

    for i in range(int(N)):
        subset = scatter.loc[scatter['gen'] == i]

        trace0 = go.Scatter(
            x=subset.loc[:, 'gen'],
            y=subset.loc[:, 'fitness'],
            mode='markers',
            marker=dict(size=7,
                        line=dict(width=1),
                        color=c[i],
                        opacity=0.5
                        ),
            name='gen {}'.format(i),
            text=subset.loc[:, 'genome']
        )
        l.append(trace0)

    trace0 = go.Scatter(
        x=dt.loc[:, 'gen'],
        y=dt.loc[:, 'max'],
        mode='lines',
        name='Max',
        line=dict(
            color="rgb(204, 51, 51)",
            dash="solid",
            shape="spline",
            smoothing=0.0,
            width=2
        ),
    )

    trace1 = go.Scatter(
        x=dt.loc[:, 'gen'],
        y=dt.loc[:, 'median'],
        mode='lines',
        name='Median',
        line=dict(
            color="rgb(173, 181, 97)",
            shape="spline",
            dash="solid",
            smoothing=0.0,
            width=2
        )
    )

    trace2 = go.Scatter(
        x=dt.loc[:, 'gen'],
        y=dt.loc[:, 'avg'],
        mode='lines',
        name='Average',
        line=dict(
            color="rgb(62, 173, 212)",
            shape="spline",
            dash="solid",
            smoothing=0.0,
            width=2
        )
    )

    data = [trace0, trace1, trace2]

    layout = go.Layout(
        title='Fitness of Population Individuals - {}'.format(title),
        hovermode='closest',
        xaxis=dict(
            title='Generations',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Fitness',
            ticklen=5,
            gridwidth=1,
        ),
        showlegend=False
    )

    fig = go.Figure(data=data+l, layout=layout)

    return py.iplot(fig, filename='fitness-average-n-runs', layout=layout)


def plot_scatter(dt, title):
    """Plots a Scatter plot of each individual in the population"""
    l = []
    y = []

    N = len(dt.gen.unique())

    c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]
    for i in range(int(N)):
        subset = dt.loc[dt['gen'] == i]

        trace0 = go.Scatter(
            x=subset.loc[:, 'gen'],
            y=subset.loc[:, 'fitness'],
            mode='markers',
            marker=dict(size=14,
                        line=dict(width=1),
                        color=c[i],
                        opacity=0.3
                        ),
            name='gen {}'.format(i),
            text=subset.loc[:, 'genome'],
        )
        l.append(trace0)

    layout = go.Layout(
        title='Fitness  of Population Individuals - {}'.format(title),
        hovermode='closest',
        xaxis=dict(
            title='Generations',
            ticklen=5,
            zeroline=False,
            gridwidth=2,
        ),
        yaxis=dict(
            title='Fitness',
            ticklen=5,
            gridwidth=1,
        ),
        showlegend=False
    )

    fig = go.Figure(data=l, layout=layout)

    return py.iplot(fig, filename='population-scatter')


def plot_grid(grid):
    trace = go.Heatmap(z=grid, colorscale='Viridis')
    data = [trace]

    layout = go.Layout(
        title='Environment and obstacles',
        showlegend=False
    )

    return py.iplot(data, filename='grid-heatmap', layout=layout)


def plot_fitness(dt, title):

    upper_bound = go.Scatter(
        name='75%',
        x=dt.index.values,
        y=dt.loc[:, 'q3'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    trace = go.Scatter(
        name='Median',
        x=dt.index.values,
        y=dt.loc[:, 'q2'],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
        fillcolor='rgba(68, 68, 68, 0.3)',  
        fill='tonexty')

    lower_bound = go.Scatter(
        name='25%',
        x=dt.index.values,
        y=dt.loc[:, 'q1'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines')

    trace_max = go.Scatter(
        x=dt.index.values,
        y=dt.loc[:, 'q4'],
        mode='lines',
        name='Max',
        line=dict(
            color="rgb(204, 51, 51)",
            dash="solid",
            shape="spline",
            smoothing=0.0,
            width=2
        ),
    )

    data = [lower_bound, trace, upper_bound, trace_max]

    layout = go.Layout(
        title=title,
        hovermode='closest',
        xaxis=dict(
            title='Generations',
            ticklen=5,
            zeroline=False,
            gridwidth=1,
        ),
        yaxis=dict(
            title='Fitness',
            ticklen=5,
            gridwidth=1,
        ),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig, filename='fitness-graph-quartile')


def plot_n_fitness(dt_list, title):

    rgb_colors = [
        "rgb(204, 51, 51)",
        "rgb(255, 153, 204)",
        "rgb(255, 204, 102)",
        "rgb(102, 204, 0)",
        "rgb(51, 51, 255)"
    ]

    data = [
        go.Scatter(
            name=str(dt.genome_id.iloc[0]),
            x=dt.index.values,
            y=dt.loc[:, 'fitness'],
            mode='lines',
            line=dict(
                color=rgb,
                dash="solid",
                shape="spline",
                smoothing=0.0,
                width=2
            )
        )
        for (rgb, dt) in zip(rgb_colors, dt_list)
    ]

    layout = go.Layout(
        title=title,
        hovermode='closest',
        xaxis=dict(
            title='# of the post-evaluation',
            ticklen=5,
            zeroline=False,
            gridwidth=1,
        ),
        yaxis=dict(
            title='Fitness',
            ticklen=5,
            gridwidth=1,
        ),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig, filename='fitness-post-evaluated-individuals')


def plot_boxplot_sensors(dt):

    colors = [
        "#3D9970",
        "#FF4136",
        "#ff9933",
        "#6666ff",
        "#33cccc",
        "#39e600",
        "#3333cc"
    ]

    data = [
        go.Box(
            y=dt.loc[:, 's{}'.format(i+1)],
            name='sensor {}'.format(i+1),
            marker=dict(color=color)
        )
        for i, color in enumerate(colors)
    ]

    layout = go.Layout(
        yaxis=dict(
            title='Sensors Activations',
            zeroline=False
        ),
        title='Sensors Behavioral Features of Individual {}'.format(
            dt.loc[:, 'genome_id'].iloc[0]),
    )

    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig)


def plot_boxplot_fitness(dt_list):
    
    colors = [
        "#3D9970",
        "#FF4136",
        "#ff9933",
        "#6666ff",
        "#33cccc",
        "#39e600",
        "#3333cc"
    ]

    data = [
        go.Box(
            y=dt.loc[:, 'fitness'],
            name='Individual {}'.format(dt.loc[:, 'genome_id'].iloc[0]),
            marker=dict(color=color)
        )
        for (color, dt) in zip(colors, dt_list)
    ]

    layout = go.Layout(
        yaxis=dict(
            title='Fitness',
            zeroline=False
        ),
        title='Noise in fitness performance of best controllers.',
    )

    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig)


def plot_boxplot_wheels(dt_list):
    data = [
        go.Box(
            x=['individual {0}'.format(genome_id)
               for genome_id in dt.loc[:, 'genome_id']],
            y=dt.loc[:, '{}'.format(wheel)],
            name='individual {0} {1}'.format(
                dt.loc[:, 'genome_id'].iloc[0], wheel),
            marker=dict(color=color),
        )
        for dt in dt_list
        for (color, wheel) in zip(['#FF9933', '#6666FF'], ['avg_left', 'avg_right'])
    ]

    layout = go.Layout(
        yaxis=dict(
            title='Wheel Speed Activation Values',
            zeroline=False
        ),
        boxmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig)


def plot_path(genomes):

    colors = [
        "#3D9970",
        "#FF4136",
        "#ff9933",
        "#6666ff",
        "#33cccc",
        "#39e600",
        "#3333cc",
        "#42f498",
        "#3c506d",
        "#ada387"
    ]

    data = [
        go.Scatter(
            x=np.array(genome.position)[:, 0],
            y=np.array(genome.position)[:, 1],
            mode='lines',
            name='path {0}'.format(genome.key),
            marker=dict(color=color)
        ) for (color, genome) in zip(colors, genomes)
    ]

    layout = go.Layout(
        title='Path travelled by best genomes of the simulation.', #.format(genomes[0].key),
        xaxis=dict(
            zeroline=True,
            showline=True,
            mirror='ticks',
            zerolinecolor='#969696',
            zerolinewidth=4,
            linecolor='#636363',
            linewidth=6,
            range=[0.06, 1.10]
        ),
        yaxis=dict(
            zeroline=True,
            showline=True,
            mirror='ticks',
            zerolinecolor='#969696',
            zerolinewidth=4,
            linecolor='#636363',
            linewidth=6,
            range=[0.0, 0.78]
        ),
        shapes=[
            # filled Rectangle
            dict(
                type='rect',
                x0=0.80,
                y0=0.0,
                x1= 0.86,
                y1= 0.3,
                line=dict(
                        color="rgba(128, 0, 128, 1)",
                        width=2,
                    ),
                fillcolor='rgba(128, 0, 128, 0.7)',
            ),
            dict(
                type='rect',
                x0=0.06,
                y0=0.40,
                x1= 0.33,
                y1= 0.46,
                line=dict(
                        color="rgba(128, 0, 128, 1)",
                        width=2,
                    ),
                fillcolor='rgba(128, 0, 128, 0.7)',
            ),
            dict(
                type='rect',
                x0=0.57,
                y0=0.40,
                x1= 0.68,
                y1= 0.78,
                line=dict(
                        color="rgba(128, 0, 128, 1)",
                        width=2,
                    ),
                fillcolor='rgba(128, 0, 128, 0.7)',
            )  
        ]
    )

    fig = go.Figure(data=data, layout=layout)

    return py.iplot(fig, filename='path-traveled-genomes')


def plot_thymio_fitness(thymio1, thymio2, title):

    thymio1 = go.Scatter(
        name='Thymio 1 - genome_id: {0}'.format(thymio1.genome_id.iloc[0]),
        x=thymio1.index.values,
        y=thymio1.loc[:, 'fitness'],
        mode='lines',
        line=dict(
            color="rgb(255, 204, 102)",
            dash="solid",
            shape="spline",
            smoothing=0.0,
            width=2
        )
    )

    thymio2 = go.Scatter(
        name='Thymio 2 - genome_id: {0}'.format(thymio2.genome_id.iloc[0]),
        x=thymio2.index.values,
        y=thymio2.loc[:, 'fitness'],
        mode='lines',
        line=dict(
            color="rgb(102, 204, 0)",
            dash="solid",
            shape="spline",
            smoothing=0.0,
            width=2
        )
    )

    data = [thymio1, thymio2]

    layout = go.Layout(
        title=title,
        hovermode='closest',
        xaxis=dict(
            title='# of the post-evaluation',
            ticklen=5,
            zeroline=False,
            gridwidth=1,
        ),
        yaxis=dict(
            title='Fitness',
            ticklen=5,
            gridwidth=1,
        ),
        showlegend=True
    )

    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig, filename='fitness-difference-thymio1-thymio2')


def plot_thymio_behaviors(behaviors_list):

    colors = [
        "#3D9970",
        "#FF4136",
        "#ff9933",
        "#6666ff",
        "#33cccc",
        "#39e600",
        "#3333cc",
        "#42f498",
        "#3c506d",
        "#ada387"
    ]

    data = [
            go.Box(
                y=dt.iloc[:, 2:].sum(axis=1),
                name='Behavioral Features {0}'.format(dt.loc[:, 'genome_id'].iloc[0]),
                marker=dict(color=color)
            )
            for (color, dt) in zip(colors, behaviors_list)
    ]

    # thymio1 = go.Box(
    #     y=thymio1,
    #     name='Behavioral Features Thymio 1',
    #     marker=dict(color="#FF4136")
    # )

    # thymio2 = go.Box(
    #     y=thymio2,
    #     name='Behavioral Features Thymio 2',
    #     marker=dict(color="#39e600")
    # )

    # data = [thymio1, thymio2]

    layout = go.Layout(
        yaxis=dict(
            title='Summed Behavioral Featuers of 10 runs',
            zeroline=False
        ),
        title='Behavioral differences of controllers'
    )

    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig)


def plot_moea_fitness(fitness_data, hof, title='Evaluation objectives. MOEA. Transferability.'):
    trace1 = go.Scatter3d(
        x=fitness_data.loc[:, 'fitness'],
        y=fitness_data.loc[:, 'str_disparity'],
        z=fitness_data.loc[:, 'diversity'],
        mode='markers',
        marker=dict(
            size=4,
            # color=fitness_data.loc[:, 'diversity'], # set color to an array/list of desired values
            # colorscale='Viridis',   # choose a colorscale
            opacity=0.8
        ),
        text=fitness_data.loc[:, 'genome_id'],
    )

    data = [trace1]
    layout = go.Layout(
        title=title,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        scene = dict(
            xaxis = dict(
                title='Task-fitness'),
            yaxis = dict(
                title='STR Disparity'),
            zaxis = dict(
                title='Diversity'),
            annotations= [dict(
                showarrow = True,
                x = ind.fitness.values[0],
                y = ind.fitness.values[1],
                z = ind.fitness.values[2],
                text = ind.key,
                xanchor = "left",
                xshift = 10,
                opacity = 0.7,
                textangle = 0,
                ax = 0,
                ay = -75,
                font = dict(
                color = "black",
                size = 12
                ),
                arrowcolor = "black",
                arrowsize = 3,
                arrowwidth = 1,
                arrowhead = 1
            ) for ind in hof
            ]
        ),
        showlegend=True
    )
    fig = go.Figure(data=data, layout=layout)
    return py.iplot(fig, filename='3d-scatter-colorscale')


def plot_surrogate_model(fitness_data, title='STR Disparity Over Generations'):
    dt = fitness_data[['gen', 'str_disparity']].groupby('gen').first()

    trace0 = go.Scatter(
        x=dt.index,
        y=dt.loc[:, 'str_disparity'],
        mode='lines',
        name='STR Disparity',
        line=dict(
            color="rgb(204, 51, 51)",
            dash="solid",
            shape="spline",
            smoothing=0.0,
            width=2
        ),
    )

    layout = go.Layout(
        showlegend=True,
        hovermode='closest',
        title=title,
        xaxis=dict(
            autorange=False,
            range=[0, 20],
            showspikes=False,
            title="Generations",
            ticklen=5,
            gridwidth=1,
        ),
        yaxis=dict(     
            autorange=True,
            title="Approximated STR Disparity",
            ticklen=5,
            gridwidth=1,
        ),
    )

    data = [trace0]
    fig = go.Figure(data, layout=layout)

    return py.iplot(fig, filename=title)


def plot_str_disparity(str_disparities, title='STR Disparities of transfered controllers'):
    genome_id = np.array([str_disparity[1] for str_disparity in str_disparities])
    str_disparity = np.array([str_disparity[3] for str_disparity in str_disparities])
    real_disparity = np.array([real_disparity[4] for real_disparity in str_disparities])

    trace0 = go.Scatter(
        x=str_disparity,
        y=real_disparity,
        mode='markers',
        name='STR Disparity',
        line=dict(
            color="rgb(204, 51, 51)",
            dash="solid",
            shape="spline",
            smoothing=0.0,
            width=2
        ),
        text=genome_id
    )
    
    trace1 = go.Scatter(
                  x=np.arange(0, 15),
                  y=np.arange(0, 15),
                  mode='lines',
                  line=dict(color='rgb(31, 119, 180)'),
    )

    layout = go.Layout(
        showlegend=True,
        hovermode='closest',
        title=title,
        xaxis=dict(
            autorange=False,
            range=[0, 20],
            showspikes=False,
            title="Approximated STR Disparity",
            ticklen=5,
            gridwidth=1,
        ),
        yaxis=dict(
            autorange=True,
            title="Actual Disparity",
            ticklen=5,
            gridwidth=1,
        ),
    )

    data = [trace0, trace1]
    fig = go.Figure(data, layout=layout)

    return py.iplot(fig, filename=title)