import sys
sys.path.append('../lib/')

import pandas as pd
import numpy as np
import geopandas as gpd

import math

import scipy.interpolate
from copy import deepcopy
from shapely.geometry import Point
from collections import defaultdict
from multiprocessing import Process, Manager
import time


def preprocess(grid_file, zone_file):
    global population_density, geometry, big_within, bigzone_name, big_within_1, den_big, geo_big

    #Filter out the grids which do not have the density data
    grids = gpd.read_file(grid_file)

    deleteIndex = []

    within_1 = defaultdict(list)


    for index, row in grids.iterrows():
        if row['deso'] == None or math.isnan(row['density']):
            deleteIndex.append(index)
        else:
            within_1[row['deso']].append(row['zone'])
    grids = grids.drop(index = deleteIndex)

    
    population_density = dict(zip(grids.zone, grids.density))
    geometry = dict(zip(grids.zone, grids.geometry))

    big_grid = grids[["upper_dens", "upper_zone", "upper_xcoo", "upper_ycoo"]].drop_duplicates()
    name_big = list(big_grid.upper_zone)
    den_big = dict(zip(big_grid.upper_zone, big_grid.upper_dens))
    geo_big = dict(zip(big_grid.upper_zone, list(zip(big_grid.upper_xcoo, big_grid.upper_ycoo))))


    base_info = gpd.read_file(zone_file)
    zones_info = dict(zip(base_info['deso'], base_info['geometry']))
    zone_name = list(base_info['deso'])


    cover = []
    checked = set()
    for i in range(0, len(zone_name)):
        sub_cover = []
        for j in range(0, len(name_big)):
            if j in checked:
                continue
            point_j = Point(geo_big[name_big[j]][0], geo_big[name_big[j]][1])
            if zones_info[zone_name[i]].contains(point_j) == True:
                # grid_j in zone_i
                sub_cover.append(name_big[j])
                checked.add(j) # This grid has been occupied, we do not need to check it again.
        cover.append(sub_cover)
    within = dict(zip(zone_name, cover))


    bigzone_name = []
    bigCover = []
    subCover = []
    old_name = zone_name[0][0:5]

    bigCover_1 = []
    subCover_1 = []

    bigzone_name.append(old_name)
    subCover.extend(within[zone_name[0]])
    subCover_1.extend(within_1[zone_name[0]])

    for i in range(1, len(zone_name)):
        new_name = zone_name[i][0:5]

        if new_name == old_name:
            # this two zones belong the same big zone
            subCover.extend(within[zone_name[i]])
            subCover_1.extend(within_1[zone_name[i]])

        if new_name != old_name:
            # find a new big zone
            #store old results
            bigCover.append(deepcopy(subCover))
            subCover.clear()

            bigCover_1.append(deepcopy(subCover_1))
            subCover_1.clear()

            #store new resutls
            bigzone_name.append(new_name)
            subCover.extend(within[zone_name[i]])
            subCover_1.extend(within_1[zone_name[i]])

        old_name = new_name

    # handle the lastest case
    bigCover.append(subCover)
    bigCover_1.append(subCover_1)

    big_within = dict(zip(bigzone_name, bigCover))
    big_within_1 = dict(zip(bigzone_name, bigCover_1))


def ODM_zone_level(index, area, begin_index, end_index, bigzone_name, population_density, geometry, big_within, model_output, big_within_1, geo_big, den_big):
    df_data = pd.read_csv('../results/distance_ratio_data.csv')
    df_simulation = pd.read_csv('../results/distance_ratio_simulation.csv')


    sp_data_sweden = scipy.interpolate.interp1d(df_data.loc[df_data.country == 'sweden', ['distance']].values.reshape(-1),
                                                df_data.loc[df_data.country == 'sweden', ['ratio']].values.reshape(-1), bounds_error = False, fill_value = 1.5)

    sp_data_simulation = scipy.interpolate.interp1d(df_simulation.loc[:, ['distance']].values.reshape(-1),
                                                    df_simulation.loc[:, ['ratio']].values.reshape(-1),  bounds_error = False, fill_value = 1.5)


    model_output = dict()
    demand_d = dict()
    demand_D_survey = dict()
    demand_D_simulation = dict()


    area_1 = 1
    r_average_1 = area_1 / math.pi

    
    r_average_5= area / math.pi

    T = 1000
    f_max = 1
    f_min = 1/T
    parameter = math.log(f_max / f_min)

    p_1 = area_1 * r_average_1 * parameter
    p_5 = area * r_average_5 * parameter


    for i in range(begin_index, end_index):
        print("I am process {}, I am computing index {}, total index {}.".format(index, i, len(bigzone_name)))
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
                for begin in range(0, len(big_within_1[bigzone_name[i]])):
                    for end in range(0, len(big_within_1[bigzone_name[j]])):
                        if begin != end:
                            deltax = (geometry[big_within_1[bigzone_name[i]][begin]].centroid.x - geometry[big_within_1[bigzone_name[j]][end]].centroid.x) / 1000                    
                            deltay = (geometry[big_within_1[bigzone_name[i]][begin]].centroid.y - geometry[big_within_1[bigzone_name[j]][end]].centroid.y) / 1000
                    
                            distance = deltax * deltax + deltay * deltay

                            root_distance = math.sqrt(distance)
                        
                            tmp =  ( population_density[big_within_1[bigzone_name[i]][begin]] + population_density[big_within_1[bigzone_name[j]][end]] ) / distance
                            tmp1 = tmp * root_distance
                            tmp2 = tmp * float(sp_data_sweden(root_distance)) * root_distance
                            tmp3 = tmp * float(sp_data_simulation(root_distance)) * root_distance

                            average_daily_trips = average_daily_trips + tmp
                            demand_d_tmp = demand_d_tmp + tmp1
                            demand_D_tmp = demand_D_tmp + tmp2
                            demand_D_simulation_tmp = demand_D_simulation_tmp + tmp3

                element[bigzone_name[j]] = p_1 * average_daily_trips
                element1[bigzone_name[j]] = p_1 * demand_d_tmp
                element2[bigzone_name[j]] = p_1 * demand_D_tmp
                element3[bigzone_name[j]] = p_1 * demand_D_simulation_tmp
            else:
                for begin in range(0, len(big_within[bigzone_name[i]])):
                    for end in range(0, len(big_within[bigzone_name[j]])):
                            deltax = (geo_big[big_within[bigzone_name[i]][begin]][0] - geo_big[big_within[bigzone_name[j]][end]][0]) / 1000                    
                            deltay = (geo_big[big_within[bigzone_name[i]][begin]][1] - geo_big[big_within[bigzone_name[j]][end]][1]) / 1000
                    
                            distance = deltax * deltax + deltay * deltay

                            root_distance = math.sqrt(distance)
                        
                            tmp =  ( den_big[big_within[bigzone_name[i]][begin]] + den_big[big_within[bigzone_name[j]][end]] ) / distance
                            tmp1 = tmp * root_distance
                            tmp2 = tmp * float(sp_data_sweden(root_distance)) * root_distance
                            tmp3 = tmp * float(sp_data_simulation(root_distance)) * root_distance

                            average_daily_trips = average_daily_trips + tmp
                            demand_d_tmp = demand_d_tmp + tmp1
                            demand_D_tmp = demand_D_tmp + tmp2
                            demand_D_simulation_tmp = demand_D_simulation_tmp + tmp3

                element[bigzone_name[j]] = p_5 * average_daily_trips
                element1[bigzone_name[j]] = p_5 * demand_d_tmp
                element2[bigzone_name[j]] = p_5 * demand_D_tmp
                element3[bigzone_name[j]] = p_5 * demand_D_simulation_tmp
        model_output[bigzone_name[i]] = element
        demand_d[bigzone_name[i]] = element1
        demand_D_survey[bigzone_name[i]] = element2
        demand_D_simulation[bigzone_name[i]] = element3


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
    global population_density, geometry, big_within, bigzone_name, big_within_1, den_big, geo_big
    totalProcess = 4

    area = 100
    grid_file = '../results/grids_sweden/grids_1km_density_deso_with_10km_upper_grids.shp'
    zone_file = '../dbs/sweden/zones/DeSO/DeSO_2018_v2.shp'
    results_output = "../results/model_output_10km_1km_sweden.txt"
    demand_output = "../results/demand_10km_1km_sweden.txt"
    demand_output_survey = "../results/demand_survey_10km_1km_sweden.txt"
    demand_output_simulation = "../results/demand_simulation_10km_1km_sweden.txt"
    

    preprocess(grid_file, zone_file)
    print("Start computing.....")
    startTime = time.time()
    processes = []
    totalWork = len(bigzone_name)
    paritionSize = int(totalWork / totalProcess)
    with Manager() as manager:
        model_output = manager.dict()
        demand_d = manager.dict()
        demand_D_survey =  manager.dict()
        demand_D_simulation =  manager.dict()

        for i in range(0, totalProcess):
            start = int(i * paritionSize)
            end = 0
            if i + 1 == totalProcess:
                end = totalWork
            else:
                end = start + paritionSize
            p = Process(target=ODM_zone_level, args=(i, area, start, end, bigzone_name, population_density, geometry, big_within, model_output, big_within_1, geo_big, den_big))
            processes.append(p)
            p.start()
    
        # wait for all thread finish
        for p in processes:
            p.join()
        print("Finish! Total time: {} s.".format(time.time() - startTime))
        store(results_output, model_output)
        store(demand_output, demand_d)
        store(demand_output_survey, demand_D_survey)
        store(demand_output_simulation, demand_D_simulation)