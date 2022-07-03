import pandas as pd
import geopandas as gpd
import sys
import numpy as np
sys.path.append('../lib/')

import math
import gc


import time
import numba
from numba import jit

def get_miu(density=None, r=None, f_home=None):
    return density*r**2*f_home

def preprocess(grid_file):
    grids = gpd.read_file(grid_file)

    name_list = grids.zone

    
    geometry = dict(zip(grids.zone, grids.geometry))

    grids.loc[:, 'Radius'] = grids.loc[:, "Area"].apply(lambda x : math.sqrt(x / math.pi))
    grids.loc[:, 'miu'] = 0
    for i in grids.index:
        grids.loc[i, 'miu'] = get_miu(density=grids.loc[i, 'pop'], r=grids.loc[i, 'Radius'], f_home=1)

    point_x_info = np.zeros(shape=len(name_list))
    point_y_info = np.zeros(shape=len(name_list))
    for i in range(0, len(name_list)):
        point_x_info[i] = geometry[name_list[i]].centroid.x / 1000
        point_y_info[i] = geometry[name_list[i]].centroid.y / 1000
    
    area = np.array(list(grids.Area))
    miu = np.array(list(grids.miu))

    df_data = pd.read_csv('../results/distance_ratio_data.csv')
    df_simulation = pd.read_csv('../results/distance_ratio_simulation.csv')


    sp_data_sweden_x = df_data.loc[df_data.country == 'sweden', ['distance']].values.reshape(-1)
    sp_data_sweden_y = df_data.loc[df_data.country == 'sweden', ['ratio']].values.reshape(-1)
    
    sp_data_simulation_x = df_simulation.loc[:, ['distance']].values.reshape(-1)
    sp_data_simulation_y = df_simulation.loc[:, ['ratio']].values.reshape(-1)
 
    T = 1000
    f_max = 1
    f_min = 1/T
    parameter = math.log(f_max / f_min)
    
    del grids
    del df_data
    del df_simulation
    del geometry
    gc.collect()

    return area, miu, parameter, sp_data_sweden_x, sp_data_sweden_y, sp_data_simulation_x, sp_data_simulation_y, point_x_info, point_y_info


@jit(nopython=True, parallel=True)
def computing(area, miu, parameter, sp_data_sweden_x, sp_data_sweden_y, sp_data_simulation_x, sp_data_simulation_y, point_x_info, point_y_info):
    flows = 0
    travel_demand = 0
    travel_demand_D_survey = 0
    travel_demand_D_simulation = 0
    total_distance = 0
    for i in numba.prange(0, len(area)):    
        for j in range(0, len(area)): 
            if i != j:
                delta_x = point_x_info[i] - point_x_info[j]
                delta_y =  point_y_info[i] - point_y_info[j]
                distance_square = delta_x * delta_x + delta_y * delta_y
                distance = math.sqrt(distance_square)

                tmp = miu[j] * area[i] / distance_square * parameter
                flows = flows + tmp * distance
                travel_demand = travel_demand + tmp * distance * distance
                travel_demand_D_survey = travel_demand_D_survey  + tmp * np.interp(distance, sp_data_sweden_x, sp_data_sweden_y) * distance
                travel_demand_D_simulation = travel_demand_D_simulation + tmp * np.interp(distance, sp_data_simulation_x, sp_data_simulation_y) * distance
                total_distance = total_distance + distance
    average_distance = total_distance / ((len(area) - 1) * (len(area) - 1))
    return flows, travel_demand, travel_demand_D_survey, travel_demand_D_simulation, average_distance


if __name__ == "__main__":  
    target_name = ["ssp1_yr2010", "ssp1_yr2020", "ssp1_yr2030", "ssp1_yr2040", "ssp1_yr2050", "ssp1_yr2060", "ssp1_yr2070", "ssp1_yr2080", "ssp1_yr2090", "ssp1_yr2100"]
    for name in target_name:
        print(name)
        grid_file = '../ssps/sweden_vg_{}.shp'.format(name)
        area, miu, parameter, sp_data_sweden_x, sp_data_sweden_y, sp_data_simulation_x, sp_data_simulation_y, point_x_info, point_y_info = preprocess(grid_file)
        print("Start computing {}".format(name))
        startTime = time.time()
    
        flows, travel_demand, travel_demand_D_survey, travel_demand_D_simulation, average_distance = computing(area, miu, parameter, sp_data_sweden_x, sp_data_sweden_y, sp_data_simulation_x, sp_data_simulation_y, point_x_info, point_y_info)

        result_file = "../results/{}_result.txt".format(name)
        with open(result_file, 'w') as f:
            f.write("Distance based flows: {}\n".format(flows))
            f.write("Distance based travel_demand_d: {}\n".format(travel_demand))
            f.write('Distance based travel_demand_D_survey: {}\n'.format(travel_demand_D_survey))
            f.write("Distance based travel_demand_D_simulation: {}\n".format(travel_demand_D_simulation))
            f.write("-----\n")
            f.write("Distance weighted flows: {}\n".format(flows / average_distance))
            f.write("Distance weighted travel_demand_d: {}\n".format(travel_demand / average_distance))
            f.write('Distance weighted travel_demand_D_survey: {}\n'.format(travel_demand_D_survey / average_distance))
            f.write("Distance weighted travel_demand_D_simulation: {}\n".format(travel_demand_D_simulation / average_distance))
    
        print("computing {} has finished! Total time: {} s.".format(name, time.time() - startTime))
        