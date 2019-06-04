from pathlib import Path, PurePath
import numpy as np
import pandas as pd
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
