from pathlib import Path, PurePath
from utility.path_tracking import create_grid
import numpy as np
import pandas as pd
import pickle
import os


def process_data(fitness_file, population_file):
    # list of data frame for each fitness of simuation run
    dt_list = [pd.read_csv(
        f, names=['gen', 'avg', 'std', 'max', 'median']) for f in fitness_file]
    # concatenate the list to single data frame
    dt_concat = pd.concat(tuple(d for d in dt_list))
    # average the fitness values
    dt_avg = dt_concat.groupby(dt_concat.gen).mean()

    gen_data = []

    for f in population_file:
        (_, _, sim, _) = f.parts
        df = pd.read_csv(f, names=['gen', 'genome', 'fitness'])
        df['sim'] = sim
        gen_data.append(df)

    # concat all data into one dataframe
    gen_data = pd.concat(tuple(d for d in gen_data)).sort_values(by=['gen'])

    quartiles = compute_quartiles(gen_data)

    # q1         q2         q3       avg       std        max    median
    # fitness_dt = q_dt.join(dt_avg).sort_values(by='gen')

    return dt_avg, quartiles, gen_data


def compute_quartiles(dt):
    q = (dt[['fitness']].groupby(dt.gen)
         .quantile([.25, .5, .75, 1.])
         .rename(index=str, columns={'fitness': 'fitness'})
         .unstack(level=-2)
         .T
         .reset_index()
         .set_index('gen')
         .drop(columns='level_0')
         .rename(index=str, columns={'0.25': 'q1', '0.5': 'q2', '0.75': 'q3', '1.0': 'q4'})
         )
    q.index = q.index.astype(int)

    return q.sort_index()


def read_behaviors(files):
    columns = [
        'gen',
        'genome_id',
        'simulation',
        'avg_left', 'avg_right',
        's1', 's2', 's3', 's4', 's5', 's6', 's7',
        'area0_count', 'area0_percentage', 'area0_total',
        'area1_count', 'area1_percentage', 'area1_total',
        'area2_count', 'area2_percentage',
        'total',
    ]

    USE_COLUMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 19]

    converters = {
        0: lambda x: int(x),
        1: lambda x: int(x),
        2: lambda x: str(x),
        3: lambda x: float(x),
        4: lambda x: float(x),
        5: lambda x: float(x),
        6: lambda x: float(x),
        7: lambda x: float(x),
        8: lambda x: float(x),
        9: lambda x: float(x),
        10: lambda x: float(x),
        11: lambda x: float(x),
        12: lambda x: int(x),
        13: lambda x: float(x),
        14: lambda x: int(x),
        15: lambda x: int(x),
        16: lambda x: float(x),
        17: lambda x: int(x),
        18: lambda x: int(x),
        19: lambda x: float(x),
        20: lambda x: int(x),
    }

    features = [pd.read_csv(f, names=columns, converters=converters,
                            usecols=USE_COLUMS) for f in files]

    return features


def read_restored_behaviors(files):
    columns = [
        'genome_id',
        'fitness',
        'avg_left', 'avg_right',
        's1', 's2', 's3', 's4', 's5', 's6', 's7',
        'area0_percentage',
        'area1_percentage',
        'area2_percentage',
    ]

    converters = {
        0: lambda x: int(float(x)),
        1: lambda x: float(x),
        2: lambda x: float(x),
        3: lambda x: float(x),
        4: lambda x: float(x),
        5: lambda x: float(x),
        6: lambda x: float(x),
        7: lambda x: float(x),
        8: lambda x: float(x),
        9: lambda x: float(x),
        10: lambda x: float(x),
        11: lambda x: float(x),
        12: lambda x: float(x),
    }

    features = [pd.read_csv(f, names=columns, converters=converters)
                for f in files]

    return features


def obstacle_grid():
    obstacle_markers = [
        dict([(9, dict(dimension=[80, 400]))]),
        dict([(10, dict(dimension=[40, 250]))]),
        dict([(11, dict(dimension=[260, 60]))]),
    ]

    obstacles_pos = [[620, 590, 0], [880, 100, 0], [150, 430, 0]]
    for position, marker in zip(obstacles_pos, obstacle_markers):
        for _, value in marker.items():
            value.update(center=position[:2])

    grid = create_grid(obstacle_markers)

    return grid


def return_genome(file):
    with open(file, 'rb') as f:
        genome = pickle.load(f)
    return genome


def restore_genomes(files):
    genomes = [return_genome(f) for f in files]
    for genome in genomes:
        genome.position = np.array(genome.position)
    return genomes
