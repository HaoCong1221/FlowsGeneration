# # This is Vij*distance for 1 km，3km,4km,5km,10km in VG
import sys
sys.path.append('../lib/')

import pandas as pd
import numpy as np
import geopandas as gpd

import math

from copy import deepcopy

from collections import defaultdict
from multiprocessing import Process, Manager
import time

import scipy.interpolate


def preprocess(grid_file,zone_file):
    global population_density, geometry, big_within, bigzone_name, sp_data_sweden, sp_data_simulation
    grids = gpd.read_file(grid_file)
    zones = gpd.read_file(zone_file)
    
    zones.loc[:, 'deso_3'] = zones.loc[:, 'deso'].apply(lambda x: x[:2])
    zones_subset = zones.loc[zones['deso_3'] == '14', :]
    zone_name = list(zones_subset['deso'])

    within = defaultdict(list)

    #Filter out the grids which do not have the density data
    deleteIndex = []

    for index, row in grids.iterrows():
        if row['deso'] == None or math.isnan(row['density']):
            deleteIndex.append(index)
        else:
            within[row['deso']].append(row['zone'])

    grids = grids.drop(index = deleteIndex)

    population_density = dict(zip(grids.zone, grids.density))
    geometry = dict(zip(grids.zone, grids.geometry))

    bigzone_name = []
    bigCover = []
    subCover = []
    old_name = zone_name[0][0:5]

    bigzone_name.append(old_name)
    subCover.extend(within[zone_name[0]])

    for i in range(1, len(zone_name)):
        new_name = zone_name[i][0:5]

        if new_name == old_name:
        # this two zones belong the same big zone
            subCover.extend(within[zone_name[i]])

        if new_name != old_name:
            # find a new big zone
            #store old results
            bigCover.append(deepcopy(subCover))
            subCover.clear()

            #store new resutls
            bigzone_name.append(new_name)
            subCover.extend(within[zone_name[i]])

        old_name = new_name

    # handle the lastest case
    bigCover.append(subCover)
    big_within = dict(zip(bigzone_name, bigCover))


    df_data = pd.read_csv('../results/distance_ratio_data.csv')
    df_simulation = pd.read_csv('../results/distance_ratio_simulation.csv')


    sp_data_sweden = scipy.interpolate.interp1d(df_data.loc[df_data.country == 'sweden', ['distance']].values.reshape(-1),
                                            df_data.loc[df_data.country == 'sweden', ['ratio']].values.reshape(-1), bounds_error = False, fill_value = 1.5)

    sp_data_simulation = scipy.interpolate.interp1d(df_simulation.loc[:, ['distance']].values.reshape(-1),
                                                df_simulation.loc[:, ['ratio']].values.reshape(-1),  bounds_error = False, fill_value = 1.5)

def ODM_zone_level(index, area, begin_index, end_index, bigzone_name, population_density, geometry, big_within, model_output, demand_d, demand_D_survey, demand_D_simulation, sp_data_sweden, sp_data_simulation):
    r_average = area/math.pi

    # parameter = ln(f_max/f_min), f_min = 1/T, f_max = 1 T = 1000
    T = 1000
    f_max = 1
    f_min = 1/T
    parameter = math.log(f_max / f_min)

    p = area * r_average * parameter

    
    for i in range(begin_index, end_index):
        print("I am process {}, I am computing index {}...".format(index, i))
        element = dict()
        element1 = dict()
        element2 = dict()
        element3 = dict()
        for j in range(i, len(bigzone_name)): 
            average_daily_trips = 0
            demand_d_tmp = 0
            demand_D_tmp = 0
            demand_D_simulation_tmp = 0
            if i == j:
                for begin in range(0, len(big_within[bigzone_name[i]])):
                    for end in range(0, len(big_within[bigzone_name[j]])):
                        if begin != end:
                            deltax = (geometry[big_within[bigzone_name[i]][begin]].centroid.x - geometry[big_within[bigzone_name[j]][end]].centroid.x) / 1000                    
                            deltay = (geometry[big_within[bigzone_name[i]][begin]].centroid.y - geometry[big_within[bigzone_name[j]][end]].centroid.y) / 1000
                    
                            distance = deltax * deltax + deltay * deltay

                            root_distance = math.sqrt(distance)
                        
                            tmp =  ( population_density[big_within[bigzone_name[i]][begin]] + population_density[big_within[bigzone_name[j]][end]] ) / distance
                            tmp1 = tmp * root_distance
                            tmp2 = tmp * float(sp_data_sweden(root_distance)) * root_distance 
                            tmp3 = tmp * float(sp_data_simulation(root_distance)) * root_distance
 
                            average_daily_trips = average_daily_trips + tmp
                            demand_d_tmp = demand_d_tmp + tmp1
                            demand_D_tmp = demand_D_tmp + tmp2
                            demand_D_simulation_tmp = demand_D_simulation_tmp + tmp3

                element[bigzone_name[j]] = p * average_daily_trips
                element1[bigzone_name[j]] = p * demand_d_tmp
                element2[bigzone_name[j]] = p * demand_D_tmp
                element3[bigzone_name[j]] = p * demand_D_simulation_tmp
            else:
                for begin in range(0, len(big_within[bigzone_name[i]])):
                    for end in range(0, len(big_within[bigzone_name[j]])):
                            deltax = (geometry[big_within[bigzone_name[i]][begin]].centroid.x - geometry[big_within[bigzone_name[j]][end]].centroid.x) / 1000                    
                            deltay = (geometry[big_within[bigzone_name[i]][begin]].centroid.y - geometry[big_within[bigzone_name[j]][end]].centroid.y) / 1000
                    
                            distance = deltax * deltax + deltay * deltay

                            root_distance = math.sqrt(distance)
                        
                            tmp =  ( population_density[big_within[bigzone_name[i]][begin]] + population_density[big_within[bigzone_name[j]][end]] ) / distance
                            tmp1 = tmp * root_distance
                            tmp2 = tmp * float(sp_data_sweden(root_distance)) * root_distance
                            tmp3 = tmp * float(sp_data_simulation(root_distance)) * root_distance
 
                            average_daily_trips = average_daily_trips + tmp
                            demand_d_tmp = demand_d_tmp + tmp1
                            demand_D_tmp = demand_D_tmp + tmp2
                            demand_D_simulation_tmp = demand_D_simulation_tmp + tmp3

                element[bigzone_name[j]] = p * average_daily_trips
                element1[bigzone_name[j]] = p * demand_d_tmp
                element2[bigzone_name[j]] = p * demand_D_tmp
                element3[bigzone_name[j]] = p * demand_D_simulation_tmp
        model_output[bigzone_name[i]] = element
        demand_d[bigzone_name[i]] = element1
        demand_D_survey[bigzone_name[i]] = element2
        demand_D_simulation[bigzone_name[i]] = element3
    print('I am process {}, I have finished my work!'.format(index))
    
    


def store(results_output, model_output):
    result = np.zeros(len(bigzone_name)*len(bigzone_name))
    for i in range(0, len(bigzone_name)):
        for j in range(0, len(bigzone_name)):
            if i <= j:
                result[i + j * len(bigzone_name)] = model_output[bigzone_name[i]][bigzone_name[j]]
            if i > j:
                result[i + j * len(bigzone_name)] = model_output[bigzone_name[j]][bigzone_name[i]]
    np.savetxt(results_output, result)

if __name__ == '__main__':
    global bigzone_name, population_density, geometry, big_within, sp_data_sweden, sp_data_simulation
    totalProcess = 4

    area = 1
    gpd_file = '../results/grids_vgr_1km_density_deso.shp'
    zone_file = '../dbs/sweden/zones/DeSO/DeSO_2018_v2.shp'
    results_output = "../results/model_output_1km.txt"
    results_demand_d = '../results/demand_d_1km.txt'
    results_demand_D_servey = '../results/demand_D_survey_1km.txt'
    results_demand_D_simulation = '../results/demand_D_simulation_1km.txt'
    

    preprocess(gpd_file,zone_file)
    print("Start computing.....")
    startTime = time.time()
    processes = []
    totalWork = len(bigzone_name)
    paritionSize = int(totalWork / totalProcess)
    with Manager() as manager:
        model_output = manager.dict()
        demand_d = manager.dict()
        demand_D_survey = manager.dict()
        demand_D_simulation = manager.dict()

        for i in range(0, totalProcess):
            start = int(i * paritionSize)
            end = 0
            if i + 1 == totalProcess:
                end = totalWork
            else:
                end = start + paritionSize
            p = Process(target=ODM_zone_level, args=(i, area, start, end, bigzone_name, population_density, geometry, big_within, model_output, demand_d, demand_D_survey, demand_D_simulation, sp_data_sweden, sp_data_simulation))
            processes.append(p)
            p.start()
    
        # wait for all thread finish
        for p in processes:
            p.join()
        print("Finish! Total time: {} s.".format(time.time() - startTime))
        store(results_output, model_output)
        store(results_demand_d, demand_d)
        store(results_demand_D_servey, demand_D_survey)
        store(results_demand_D_simulation, demand_D_simulation)
    