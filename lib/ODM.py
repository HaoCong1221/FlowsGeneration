import sys
sys.path.append('../lib/')
import pandas as pd
import numpy as np
import math
# Import self-defined libraries
import sweden
import netherlands
import workers
import v_ij

data_sweden = sweden.GroundTruthLoader()
data_sweden.load_zones()
data_sweden.create_boundary()
#data_sweden.load_odm()
data_sweden.load_population()
distances = workers.zone_distances(data_sweden.zones)
df_d = distances.unstack(level=1)

dict_zone = dict(zip(data_sweden.zones['zone'], data_sweden.zones['geometry']))
area = dict(zip(data_sweden.zones['zone'], data_sweden.zones.area))
# change unit from m*m to km *km
for i in area.keys():
    area[i] = area[i] / 1000000
#print(area)

# Use r_average to denote r_j, which is the distance to the boundary of the location j.
# A=pi*r_average^2
r_average = []
for i in area.keys(): 
    r_average.append(math.sqrt(area[i]/math.pi))
r_average_dict = dict(zip(data_sweden.zones['zone'], r_average))

#print(r_average_dict)
# parameter = ln(f_max/f_min), f_min = 1/T, f_max = 1 T = 1000
T = 1000
f_max = 1
f_min = 1/T
parameter = math.log(f_max / f_min)


population_density = dict(zip(data_sweden.population['zone'], data_sweden.population['pop']))
ODM_data = []

for i in data_sweden.population['zone']:
    for j in data_sweden.population['zone']:
        element = dict()
        element['origin_main_deso'] = i
        element['desti_main_deso'] = j
        if i != j:
            # filter out the error divide by zero
            element['trip_weight'] = v_ij.average_daily_trips(population_density[j], area[i], r_average_dict[j], df_d[i][j], parameter)
        if i == j:
            element['trip_weight'] = 0
        ODM_data.append(element)
ODM = pd.DataFrame(ODM_data)

ODM.to_csv('./ODM.csv', index=False)