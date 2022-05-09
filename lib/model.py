import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import math
from sklearn.metrics import pairwise_distances
import sweden


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
    selected_cols = ['ozone', 'dzone', 'd_ij', 'v_ij', 'grid_type']
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
        self.gt.odm = self.gt.odm.rename_axis(['ozone', 'dzone'])
        self.gt.odm.name = 'v_ij_gt'
        self.gt.odm = self.gt.odm.reset_index()

    def gt_aggregation(self, agg_level=5):
        self.odm = self.gt.odm.copy()
        self.odm.loc[:, 'ozone_deso'] = self.odm.loc[:, 'ozone'].apply(lambda x: x[:agg_level])
        self.odm.loc[:, 'dzone_deso'] = self.odm.loc[:, 'dzone'].apply(lambda x: x[:agg_level])
        self.odm = self.odm.groupby(['ozone_deso', 'dzone_deso'])['v_ij_gt'].sum().reset_index()

    def comparison_ssi(self, df_odm_agg=None):
        flows_c = self.odm.loc[self.odm['v_ij_gt'] != 0, :].merge(df_odm_agg, on=['ozone_deso', 'dzone_deso'],
                                                                           how='inner')
        flows_c.loc[:, 'v_ij_gt'] = flows_c.loc[:, 'v_ij_gt'] / flows_c.loc[:, 'v_ij_gt'].sum()
        flows_c.loc[:, 'v_ij'] = flows_c.loc[:, 'v_ij'] / flows_c.loc[:, 'v_ij'].sum()
        flows_c.loc[:, 'v_ij_min'] = flows_c.apply(lambda row: min(row['v_ij_gt'], row['v_ij']),
                                                                     axis=1)
        SSI = 2 * flows_c.loc[:, 'v_ij_min'].sum() / \
              (flows_c.loc[:, 'v_ij_gt'].sum() + flows_c.loc[:, 'v_ij'].sum())
        return SSI