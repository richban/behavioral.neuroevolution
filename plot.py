import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import csv

__credits__ = 'vladas-v'

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

    stagnation = np.array(stagnation).astype(np.float)
    id = np.array(id)
    x_size = int(len(id) / 2) + 1
    params = {'figure.figsize': (x_size, 5),
              'xtick.labelsize':'x-small'}
    pylab.rcParams.update(params)
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

        for row in data:
            gen.append(int(row[0]))
            avg_fit.append(float(row[1]))
            stdv.append(float(row[2]))
            max_fit.append(float(row[3]))

    if len(gen) < 2:
        return None

    x_size = int(len(gen) / 10) + 1
    params = {'figure.figsize': (x_size, 5),
              'xtick.labelsize':'x-small'}
    pylab.rcParams.update(params)

    plt.plot(gen, avg_fit, 'b', linewidth=0.5,)
    plt.plot(gen, stdv, 'r', linewidth=0.5,)
    plt.plot(gen, max_fit, 'g', linewidth=0.5,)

    plt.plot(gen, max_fit, 'g^', markersize=5, label='Max fitness')
    plt.plot(gen, avg_fit, 'bo', markersize=5, label='Average fitness')
    plt.plot(gen, stdv, 'rs', markersize=5, label='Standard deviation')

    plt.ylabel('Fitness')
    plt.xlabel('Generation')
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin, xmax, ymin, ymax + 50])
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')
    plt.tight_layout()

    plt.savefig(imgfilename)
    plt.clf()
    plt.close('all')
    return imgfilename
