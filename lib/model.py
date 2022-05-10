import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import math
from sklearn.metrics import pairwise_distances
import sweden
import scipy.interpolate

df_data = pd.read_csv('D:/FlowsGeneration/results/distance_ratio_data.csv')
df_simulation = pd.read_csv('D:/FlowsGeneration/results/distance_ratio_simulation.csv')
sp_data_sweden = scipy.interpolate.interp1d(df_data.loc[df_data.country == 'sweden', ['distance']].values.reshape(-1),
                                            df_data.loc[df_data.country == 'sweden', ['ratio']].values.reshape(-1),
                                            bounds_error=False, fill_value=1.5)
sp_data_simulation = scipy.interpolate.interp1d(df_simulation.loc[:, ['distance']].values.reshape(-1),
                                                df_simulation.loc[:, ['ratio']].values.reshape(-1),
                                                bounds_error=False, fill_value=1.5)


def miu(density=None, r=None, f_home=None):
    return density*r**2*f_home


def para(f_max=None, T=None):
    f_min = 1/T
    return math.log(f_max / f_min)


def zone_flows(grids=None, T=1000, f_max=1, area=1, r=1, f_home=1):
    """
    :param grids
    GeoDataFrame [*index, zone, upper_zone, deso, density, geometry]
    Must be in a CRS of unit: metre
    """
    for ax in grids.crs.axis_info:
        assert ax.unit_name == 'metre'
    par = para(f_max=f_max, T=T)
    grids.loc[:, 'miu'] = grids.loc[:, 'density'].apply(lambda x: miu(density=x, r=r, f_home=f_home))
    distances_meters = pairwise_distances(
        list(zip(
            grids.geometry.centroid.x.to_list(),
            grids.geometry.centroid.y.to_list(),
        ))
    )
    miu_list = grids['miu'].to_list()
    vij_total = [(x*area + y*area) * par for x in miu_list for y in miu_list]
    flows = pd.DataFrame(
        distances_meters / 1000,
        columns=grids.zone,
        index=grids.zone,
    ).stack().rename_axis(['ozone', 'dzone'])
    flows.name = 'd_ij'
    flows = flows.reset_index()
    flows.loc[:, 'v_ij'] = vij_total
    flows.loc[:, 'v_ij'] = flows.loc[:, 'v_ij'] / flows.loc[:, 'd_ij']**2
    flows.loc[:, 'D_ij_data'] = sp_data_sweden(flows.loc[:, 'd_ij'])
    flows.loc[flows['d_ij'] == 0, 'D_ij_data'] = 0
    flows.loc[:, 'D_ij_sim'] = sp_data_simulation(flows.loc[:, 'd_ij'])
    flows.loc[flows['d_ij'] == 0, 'D_ij_sim'] = 0
    return flows


def odm(grids=None, grids_upper=None, grid_size=(1, 10)):
    """
    :param grids, grids_upper
    GeoDataFrame [*index, zone, upper_zone, deso, density, geometry]
    Must be in a CRS of unit: metre
    """
    print('Generating flows within upper zones...')
    tqdm.pandas()
    flows_within = grids.groupby('upper_zone').progress_apply(lambda x: zone_flows(grids=x,
                                                                            T=1000, f_max=1,
                                                                            area=1, r=1, f_home=1)).reset_index()
    print('Generating flows between upper zones...')
    flows_between = zone_flows(grids=grids_upper, T=1000, f_max=1, area=100, r=10, f_home=1)
    print('Set diagonal to zero.')
    flows_within.replace([np.inf, -np.inf], 0, inplace=True)
    flows_between.replace([np.inf, -np.inf], 0, inplace=True)
    # Indicate the size of grids
    flows_within.loc[:, 'grid_type'] = grid_size[0]
    flows_between.loc[:, 'grid_type'] = grid_size[1]
    selected_cols = ['ozone', 'dzone', 'd_ij', 'D_ij_data', 'D_ij_sim', 'v_ij', 'grid_type']
    return pd.concat([flows_within.loc[:, selected_cols], flows_between.loc[:, selected_cols]])


def odm_aggregation(df_odm=None, grids=None, grids_upper=None, grid_size=(1, 10), agg_level=5):
    grids_dict = dict(zip(grids.zone, grids.deso))
    grids_upper_dict = dict(zip(grids_upper.zone, grids_upper.deso))
    df_odm.loc[df_odm['grid_type'] == grid_size[0], 'ozone_deso'] = df_odm.loc[
        df_odm['grid_type'] == grid_size[0], 'ozone']\
        .map(grids_dict).apply(lambda x: x[:agg_level])
    df_odm.loc[df_odm['grid_type'] == grid_size[1], 'ozone_deso'] = df_odm.loc[
        df_odm['grid_type'] == grid_size[1], 'ozone'] \
        .map(grids_upper_dict).apply(lambda x: x[:agg_level])
    df_odm.loc[df_odm['grid_type'] == grid_size[0], 'dzone_deso'] = df_odm.loc[
        df_odm['grid_type'] == grid_size[0], 'dzone']\
        .map(grids_dict).apply(lambda x: x[:agg_level])
    df_odm.loc[df_odm['grid_type'] == grid_size[1], 'dzone_deso'] = df_odm.loc[
        df_odm['grid_type'] == grid_size[1], 'dzone'] \
        .map(grids_upper_dict).apply(lambda x: x[:agg_level])
    df_odm_agg = df_odm.groupby(['ozone_deso', 'dzone_deso'])['v_ij'].sum().reset_index()
    return df_odm_agg


class ModelEvaluation:
    def __init__(self):
        self.gt = sweden.GroundTruthLoader()
        self.odm = None

    def load_gt(self):
        # Load zones
        self.gt.load_zones()
        # Load ground-truth survey data into ODM form
        self.gt.load_odm()

    def gt_aggregation(self, agg_level=5):
        self.odm = self.gt.odm.copy()
        self.odm.loc[:, 'ozone_deso'] = self.odm.loc[:, 'ozone'].apply(lambda x: x[:agg_level])
        self.odm.loc[:, 'dzone_deso'] = self.odm.loc[:, 'dzone'].apply(lambda x: x[:agg_level])
        self.odm = self.odm.groupby(['ozone_deso', 'dzone_deso'])['v_ij_gt'].sum().reset_index()

    def comparison_ssi(self, df_odm_agg=None):
        flows_c = self.odm.loc[self.odm['v_ij_gt'] != 0, :].merge(df_odm_agg, on=['ozone_deso', 'dzone_deso'],
                                                                  how='inner')
        # Convert trip number to trip frequency (ranging between 0 and 1)
        flows_c.loc[:, 'v_ij_gt'] = flows_c.loc[:, 'v_ij_gt'] / flows_c.loc[:, 'v_ij_gt'].sum()
        flows_c.loc[:, 'v_ij'] = flows_c.loc[:, 'v_ij'] / flows_c.loc[:, 'v_ij'].sum()
        flows_c.loc[:, 'v_ij_min'] = flows_c.apply(lambda row: min(row['v_ij_gt'], row['v_ij']),
                                                                     axis=1)
        SSI = 2 * flows_c.loc[:, 'v_ij_min'].sum() / \
              (flows_c.loc[:, 'v_ij_gt'].sum() + flows_c.loc[:, 'v_ij'].sum())
        return SSI