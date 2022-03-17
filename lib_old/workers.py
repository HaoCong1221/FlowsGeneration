from sklearn.metrics import pairwise_distances
import pandas as pd


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