import sys
import copy

import pandas as pd


import sweden

import workers
import v_ij

data_sweden = sweden.GroundTruthLoader()
data_sweden.load_zones()
data_sweden.create_boundary()
data_sweden.load_population()
distances = workers.zone_distances(data_sweden.zones)
df_d = distances.unstack(level=1)

# parameter = ln(f_max/f_min), f_min = 1/T, f_max = 1 T = 7
parameter = 1.9459
area = 1 # 1 km * km
radius = 0.5 #0.5 km
population = copy.deepcopy(data_sweden.population)
population_information = dict(zip(population['zone'], population['pop']))

ODM = copy.deepcopy(df_d)



for i in population['zone']:
    print(i)
    for j in population['zone']:
        if i != j:
            # filter out the error divide by zero
            ODM[i][j] = float(v_ij.average_daily_trips(population_information[j], area, radius, df_d[i][j], parameter))

ODM.to_csv("./ODM.csv", index=False, sep=',')