# This is only for 2km in VG，because some format is different from the others. 
import multiprocessing
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


def preprocess(grid_file):
    global population_density, geometry, big_within, bigzone_name
    grids = gpd.read_file(grid_file)

    big_within = defaultdict(list)

    for index, row in grids.iterrows():
        big_within[row['deso_5']].append(row['zone'])


    population_density = dict(zip(grids.zone, grids.density))
    geometry = dict(zip(grids.zone, grids.geometry))

    bigzone_name = list(grids.deso_5.drop_duplicates())

def ODM_zone_level(index, area, begin_index, end_index, bigzone_name, population_density, geometry, big_within, model_output):
    r_average = area/math.pi

    # parameter = ln(f_max/f_min), f_min = 1/T, f_max = 1 T = 1000
    T = 1000
    f_max = 1
    f_min = 1/T
    parameter = math.log(f_max / f_min)

    
    for i in range(begin_index, end_index):
        print("I am process {}, I am computing index {}, total index {}.".format(index, i, len(bigzone_name)))
        element = dict()
        for j in range(i, len(bigzone_name)): 
            average_daily_trips = 0
            for begin in range(0, len(big_within[bigzone_name[i]])):
                for end in range(0, len(big_within[bigzone_name[j]])):
                    if begin != end:
                        deltax = (geometry[big_within[bigzone_name[i]][begin]].centroid.x - geometry[big_within[bigzone_name[j]][end]].centroid.x) / 1000                    
                        deltay = (geometry[big_within[bigzone_name[i]][begin]].centroid.y - geometry[big_within[bigzone_name[j]][end]].centroid.y) / 1000
                    
                        distance = deltax * deltax + deltay * deltay

                        tmp =  area * r_average / distance * parameter
                        
                        average_daily_trips = average_daily_trips + tmp * ( population_density[big_within[bigzone_name[i]][begin]] + population_density[big_within[bigzone_name[j]][end]] )
            element[bigzone_name[j]] = average_daily_trips
        model_output[bigzone_name[i]] = element


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
    global bigzone_name, population_density, geometry, big_within
    totalProcess = 4

    area = 4
    gpd_file = '../results/grids_vgr_2km_density_deso5.shp'
    results_output = "../results/model_output_2km.txt"
    

    preprocess(gpd_file)
    print("Start computing.....")
    startTime = time.time()
    processes = []
    totalWork = len(bigzone_name)
    paritionSize = int(totalWork / totalProcess)
    with Manager() as manager:
        model_output = manager.dict()

        for i in range(0, totalProcess):
            start = int(i * paritionSize)
            end = 0
            if i + 1 == totalProcess:
                end = totalWork
            else:
                end = start + paritionSize
            p = Process(target=ODM_zone_level, args=(i, area, start, end, bigzone_name, population_density, geometry, big_within, model_output))
            processes.append(p)
            p.start()
    
        # wait for all thread finish
        for p in processes:
            p.join()
        print("Finish! Total time: {} s.".format(time.time() - startTime))
        store(results_output, model_output)
    