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

population_information = dict(zip(data_sweden.population['zone'], data_sweden.population['pop']))
ODM = dict() # {zone {zone : distance}}

for i in data_sweden.population['zone']:
    print(i)
    odm_row = dict()
    for j in data_sweden.population['zone']:
      if i != j:
      # filter out the error divide by zero
        odm_row[j] = v_ij.average_daily_trips(population_information[j], area, radius, df_d[i][j], parameter)
      if i == j:
        odm_row[j] = 0
    ODM[i] = odm_row

result_file = './odm.txt'

with open(result_file, 'w') as f:
    for i in data_sweden.population['zone']:
        for j in data_sweden.population['zone']:
            f.write(str(ODM[i][j]))
            f.write('\t')
        f.write('\n')