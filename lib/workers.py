from sklearn.metrics import pairwise_distances
import pandas as pd
import numpy as np


def zone_distances(zones):
    """
    :param zones
    GeoDataFrame [*index, zone, geometry]
    Must be in a CRS of unit: metre
    """
    for ax in zones.crs.axis_info:
        assert ax.unit_name == 'metre'

    print("Calculating distances between zones...")
    distances_meters = pairwise_distances(
        list(zip(
            zones.geometry.centroid.x.to_list(),
            zones.geometry.centroid.y.to_list(),
        ))
    )
    distances = pd.DataFrame(
        distances_meters / 1000,
        columns=zones.zone,
        index=zones.zone,
    ).stack()
    return distances


def zone_flows(zones, area=None, para=None):
    """
    :param zones
    GeoDataFrame [*index, zone, geometry]
    Must be in a CRS of unit: metre
    """
    for ax in zones.crs.axis_info:
        assert ax.unit_name == 'metre'

    distances_meters = pairwise_distances(
        list(zip(
            zones.geometry.centroid.x.to_list(),
            zones.geometry.centroid.y.to_list(),
        ))
    )
    miu = zones.miu.to_list()
    vij_total = [(x*area + y*area) * para for x in miu for y in miu]
    flows = pd.DataFrame(
        distances_meters / 1000,
        columns=zones.zone,
        index=zones.zone,
    ).stack().rename_axis(['ozone', 'dzone'])
    flows.name = 'd_ij'
    flows = flows.reset_index()
    flows.loc[:, 'v_ij'] = vij_total
    flows.loc[:, 'v_ij'] = flows.loc[:, 'v_ij'] / flows.loc[:, 'd_ij']**2
    return flows
