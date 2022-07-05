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
    total_poplution = 0
    for i in grids.index:
        grids.loc[i, 'miu'] = get_miu(density=grids.loc[i, 'pop'], r=grids.loc[i, 'Radius'], f_home=1)
        total_poplution = total_poplution + grids.loc[i, 'pop'] * grids.loc[i, 'Area']

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

    return area, miu, parameter, sp_data_sweden_x, sp_data_sweden_y, sp_data_simulation_x, sp_data_simulation_y, point_x_info, point_y_info, total_poplution


@jit(nopython=True, parallel=True)
def computing(area, miu, parameter, sp_data_sweden_x, sp_data_sweden_y, sp_data_simulation_x, sp_data_simulation_y, point_x_info, point_y_info):
    flows = 0
    flows_d = 0
    flows_D_survey = 0
    flows_D_simulation = 0
    for i in numba.prange(0, len(area)):    
        for j in range(0, len(area)): 
            if i != j:
                delta_x = point_x_info[i] - point_x_info[j]
                delta_y =  point_y_info[i] - point_y_info[j]
                distance_square = delta_x * delta_x + delta_y * delta_y
                distance = math.sqrt(distance_square)

                tmp = miu[j] * area[i] / distance_square * parameter
                
                flows = flows + tmp 
                flows_d = flows_d + tmp * distance
                flows_D_survey = flows_D_survey + tmp * np.interp(distance, sp_data_sweden_x, sp_data_sweden_y) * distance
                flows_D_simulation = flows_D_simulation  + tmp * np.interp(distance, sp_data_simulation_x, sp_data_simulation_y) * distance
                
                
    return flows, flows_d, flows_D_survey, flows_D_simulation

if __name__ == "__main__":  
    target_ssp = 'ssp1_{}'
    target_name = ["yr2010", "yr2020", "yr2030", "yr2040", "yr2050", "yr2060", "yr2070", "yr2080", "yr2090", "yr2100"]
    poplution = []
    
    total_flows = []
    total_travel_demand_d = []
    total_travel_demand_D_survey = []
    total_travel_demand_D_simulation = []
    
    total_flows_p = []
    total_travel_demand_d_p = []
    total_travel_demand_D_survey_p = []
    total_travel_demand_D_simulation_p = []

    for name in target_name:
        file_name = target_ssp.format(name)
        print(file_name)
        grid_file = '../results/ssps/ssps/sweden_vg_{}.shp'.format(file_name)
        area, miu, parameter, sp_data_sweden_x, sp_data_sweden_y, sp_data_simulation_x, sp_data_simulation_y, point_x_info, point_y_info, total_popultion = preprocess(grid_file)
        print("Start computing {}".format(file_name))
        startTime = time.time()
    
        flows, flows_d, flows_D_survey, flows_D_simulation = computing(area, miu, parameter, sp_data_sweden_x, sp_data_sweden_y, sp_data_simulation_x, sp_data_simulation_y, point_x_info, point_y_info)

        flows_p = flows / total_popultion
        flows_d_p = flows_d / total_popultion
        flows_D_survey_p = flows_D_survey / total_popultion
        flows_D_simulation_p = flows_D_simulation / total_popultion


        poplution.append(total_popultion)
        total_flows.append(flows)
        total_travel_demand_d.append(flows_d)
        total_travel_demand_D_survey.append(flows_D_survey)
        total_travel_demand_D_simulation.append(flows_D_simulation)
        total_flows_p.append(flows_p)
        total_travel_demand_d_p.append(flows_d_p)
        total_travel_demand_D_survey_p.append(flows_D_survey_p)
        total_travel_demand_D_simulation_p.append(flows_D_simulation_p)


        result_file = "../results/{}_result.txt".format(file_name)
        with open(result_file, 'w') as f:
            f.write("Total popultion: {}\n".format(total_popultion))
            f.write("Total flows: {}\n".format(flows))
            f.write("Total travel_demand_d: {}\n".format(flows_d))
            f.write('Total travel_demand_D_survey: {}\n'.format(flows_D_survey))
            f.write("Total travel_demand_D_simulation: {}\n".format(flows_D_simulation))
            f.write("Flows per people: {}\n".format(flows_p))
            f.write("Travel_demand_d per people: {}\n".format(flows_d_p))
            f.write("Travel_demand_D_survey per people: {}\n".format(flows_D_survey_p))
            f.write("Travel_demand_D_simulation per people: {}\n".format(flows_D_simulation_p))
    
        del flows
        del flows_d
        del flows_D_survey 
        del flows_D_simulation
        gc.collect()
        print("computing {} has finished! Total time: {} s.".format(name, time.time() - startTime))

    print(target_ssp.format("finish!"))
    total_result = target_ssp.format("result")
    result_path = '../results/{}.{}'.format(total_result, 'txt')
    with open(result_path, 'w') as f:
        f.write(total_result)
        f.write('\n')
        f.write('Total popultion: [{},{},{},{},{},{},{},{},{},{}]\n'.format(poplution[0], poplution[1], poplution[2], poplution[3], poplution[4], poplution[5], poplution[6], poplution[7], poplution[8], poplution[9]))

        f.write('Total flows: [{},{},{},{},{},{},{},{},{},{}]\n'.format(flows[0], flows[1], flows[2], flows[3], flows[4], flows[5], flows[6], flows[7], flows[8], flows[9]))

        f.write('Total travel_demand_d: [{},{},{},{},{},{},{},{},{},{}]\n'.format(flows_d[0], flows_d[1], flows_d[2], flows_d[3], flows_d[4], flows_d[5], flows_d[6], flows_d[7], flows_d[8], flows_d[9]))

        f.write('Total travel_demand_D_survey: [{},{},{},{},{},{},{},{},{},{}]\n'.format(flows_D_survey[0], flows_D_survey[1], flows_D_survey[2], flows_D_survey[3], flows_D_survey[4], flows_D_survey[5], flows_D_survey[6], flows_D_survey[7], flows_D_survey[8], flows_D_survey[9]))

        f.write('Total travel_demand_D_simulation: [{},{},{},{},{},{},{},{},{},{}]\n'.format(flows_D_simulation[0], flows_D_simulation[1], flows_D_simulation[2], flows_D_simulation[3], flows_D_simulation[4], flows_D_simulation[5], flows_D_simulation[6], flows_D_simulation[7], flows_D_simulation[8], flows_D_simulation[9]))

        f.write('Flows per people: [{},{},{},{},{},{},{},{},{},{}]\n'.format(flows_p[0], flows_p[1], flows_p[2], flows_p[3], flows_p[4], flows_p[5], flows_p[6], flows_p[7], flows_p[8], flows_p[9]))

        f.write('Travel_demand_d per people: [{},{},{},{},{},{},{},{},{},{}]\n'.format(flows_d_p[0], flows_d_p[1], flows_d_p[2], flows_d_p[3], flows_d_p[4], flows_d_p[5], flows_d_p[6], flows_d_p[7], flows_d_p[8], flows_d_p[9]))

        f.write('Travel_demand_D_survey per people: [{},{},{},{},{},{},{},{},{},{}]\n'.format(flows_D_survey_p[0], flows_D_survey_p[1], flows_D_survey_p[2], flows_D_survey_p[3], flows_D_survey_p[4], flows_D_survey_p[5], flows_D_survey_p[6], flows_D_survey_p[7], flows_D_survey_p[8], flows_D_survey_p[9]))

        f.write('Travel_demand_D_simulation per people: [{},{},{},{},{},{},{},{},{},{}]\n'.format(flows_D_simulation_p[0], flows_D_simulation_p[1], flows_D_simulation_p[2], flows_D_simulation_p[3], flows_D_simulation_p[4], flows_D_simulation_p[5], flows_D_simulation_p[6], flows_D_simulation_p[7], flows_D_simulation_p[8], flows_D_simulation_p[9]))