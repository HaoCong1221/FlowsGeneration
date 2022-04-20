# After running this code, we found the time needed doesn't reduced.
# We shouldn't use this code, because multithread couldn't speed up our calculation. 
# The reason is that our task is CPU-bound, we should use multiprocess to speed up.
import sys
sys.path.append('../lib/')

import pandas as pd
import numpy as np
import geopandas as gpd

import math

from copy import deepcopy

from collections import defaultdict
import threading
import time

class myThread (threading.Thread):   
    # Inheritance from the thread
    def __init__(self, threadID, totalThread, totalWork):
        threading.Thread.__init__(self)
        self.threadID = int(threadID)
        self.totalThread = int(totalThread)
        self.totalWork = int(totalWork)
        self.paritionSize = int(totalWork / totalThread)
    
    def run(self):                   
        #When thread starts, it will execute the run method
        start = int(self.threadID * self.paritionSize)
        end = 0
        if self.threadID + 1 == self.totalThread:
            end = self.totalWork
        else:
            end = start + self.paritionSize
        
        ODM_zone_level(start, end)



def preprocess(grid_file, zone_file):
    global population_density, geometry, big_within, bigzone_name
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


def ODM_zone_level(begin_index, end_index):
    global bigzone_name, population_density, geometry, model_output, area

    r_average = math.sqrt(area/math.pi)

    # parameter = ln(f_max/f_min), f_min = 1/T, f_max = 1 T = 1000
    T = 1000
    f_max = 1
    f_min = 1/T
    parameter = math.log(f_max / f_min)

    model_output = np.zeros(len(bigzone_name)*len(bigzone_name))   # the vector of model_output 
    for i in range(begin_index, end_index):
        print("Computing index {}, total index {}.".format(i, len(bigzone_name)))
        for j in range(i, len(bigzone_name)): 
            average_daily_trips = 0
            for begin in range(0, len(big_within[bigzone_name[i]])):
                for end in range(0, len(big_within[bigzone_name[j]])):
                    if begin != end:
                        deltax = (geometry[big_within[bigzone_name[i]][begin]].centroid.x - geometry[big_within[bigzone_name[j]][end]].centroid.x) / 1000                    
                        deltay = (geometry[big_within[bigzone_name[i]][begin]].centroid.y - geometry[big_within[bigzone_name[j]][end]].centroid.y) / 1000
                    
                        distance = deltax * deltax + deltay * deltay

                        tmp =  area * r_average * r_average / distance * parameter
                        
                
                        average_daily_trips = tmp * ( population_density[big_within[bigzone_name[i]][begin]] + population_density[big_within[bigzone_name[j]][end]] )

            index_1 = i * len(bigzone_name) + j
            index_2 = j * len(bigzone_name) + i
            model_output[index_1] = average_daily_trips
            model_output[index_2] = average_daily_trips

def store(results_output):
    global model_output
    np.savetxt(results_output, model_output)

if __name__ == '__main__':
    
    global area

    totalThread = 2

    area = 9
    gpd_file = '../results/grids_vgr_5km_density_deso.shp'
    zone_file = '../dbs/sweden/zones/DeSO/DeSO_2018_v2.shp'
    results_output = "../results/model_output_5km.txt"
    

    preprocess(gpd_file, zone_file)
    print("Start computing.....")
    startTime = time.time()
    threads = []
    totalWork = len(bigzone_name)
    for i in range(0, totalThread):
        thread = myThread(i, totalThread, totalWork)
        threads.append(thread)
        thread.start()
    
    # wait for all thread finish
    for t in threads:
        t.join()
    print("Finish! Total time: {} s.".format(time.time() - startTime))
    store(results_output)
    