import geopandas as gpd
import pandas as pd
import os
import subprocess


def get_repo_root():
    """Get the root directory of the repo."""
    dir_in_repo = os.path.dirname(os.path.abspath('__file__')) # os.getcwd()
    return subprocess.check_output('git rev-parse --show-toplevel'.split(),
                                   cwd=dir_in_repo,
                                   universal_newlines=True).rstrip()


ROOT_dir = get_repo_root()


class GroundTruthLoader:
    def __init__(self):
        self.zones = None
        self.odm = None
        self.boundary = None
        self.bbox = None
        self.trip_distances = None
        self.population = None

    def load_zones(self):
        # EPSG: 3006
        _zones = gpd.read_file(ROOT_dir + '/dbs/sweden/zones/DeSO/DeSO_2018_v2.shp')
        self.zones = _zones.rename(columns={"deso": "zone"})[['zone', 'geometry']]

    def load_population(self):
        self.population = pd.read_csv(ROOT_dir + "/dbs/sweden/zones/population.csv")

    def create_boundary(self):
        self.boundary = self.zones.assign(a=1).dissolve(by='a').simplify(tolerance=0.2).to_crs("EPSG:4326")

    def load_odm(self):
        trips = pd.read_csv(ROOT_dir + "/dbs/sweden/survey/day_trips.csv")
        trips = trips.loc[:, ["sub_id", 'trip_id', 'trip_main_id', 'distance_main',
                              'date', "origin_main_deso", "desti_main_deso", 'trip_weight']]
        trips = trips.drop_duplicates(subset=["sub_id", 'trip_id', 'trip_main_id'])
        trips["T"] = trips["date"].apply(lambda x: pd.to_datetime(x))
        trips = trips.loc[~trips["T"].apply(lambda x: x.weekday()).isin([5, 6]), :]
        trips.dropna(axis=0, how='any', inplace=True)
        # Prepare ODM
        odms = trips.groupby(['origin_main_deso', 'desti_main_deso']).sum()['trip_weight']
        print(odms.head())
        z = self.zones.zone
        odms = odms.reindex(pd.MultiIndex.from_product([z, z], names=['ozone', 'dzone']), fill_value=0)
        self.odm = odms / odms.sum()
        # self.odm = odms
        # Prepare the actual trip distances
        self.trip_distances = trips.loc[:, ['distance_main', 'trip_weight']].rename(columns={'distance_main': 'distance',
                                                                                             'trip_weight': 'weight'})
