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


def trip_row(df):
    row = df.iloc[0]
    row['dest_zip'] = df.iloc[-1]['dest_zip']
    row['dest_time'] = df.iloc[-1]['dest_time']
    return row


class GroundTruthLoader:
    def __init__(self):
        self.zones = None
        self.odm = None
        self.boundary = None
        self.bbox = None
        self.trip_distances = None

    def load_zones(self):
        # EPSG:28992
        _zones = gpd.read_file(ROOT_dir + '/dbs/netherlands/zones/CBS_PC4_2017_v1.shp')
        self.zones = _zones.rename(columns={"PC4": "zone"})[['zone', 'geometry']]

    def create_boundary(self):
        self.boundary = self.zones.assign(a=1).dissolve(by='a').simplify(tolerance=0.2).to_crs("EPSG:4326")

    def load_odm(self):
        sheet1 = pd.read_excel(ROOT_dir + "/dbs/netherlands/survey/OViN2017_Databestand.xlsx")
        trips = sheet1[
            ['OPID', 'AfstV', 'Wogem', 'Jaar', 'Maand', 'Dag', 'VerplID',
             'VertUur', 'VertPC', 'AankUur', 'AankPC', 'FactorV']]
        trips = trips.rename(columns={
            'Wogem': 'home_city',
            'Jaar': 'year',
            'Maand': 'month',
            'Dag': 'day',
            'VerplID': 'trip_id',
            'VertUur': 'origin_time',
            'VertPC': 'origin_zip',
            'AankUur': 'dest_time',
            'AankPC': 'dest_zip',
            'FactorV': 'weight_trip',
            'AfstV': 'distance'
        })
        # Prepare the actual trip distances
        trips_d = trips.dropna(subset=['distance'])
        trips_d.loc[:, 'distance'] = trips_d.loc[:, 'distance'] / 10 # hectometer to km
        self.trip_distances = trips_d.loc[:, ['distance', 'weight_trip']].rename(columns={'weight_trip': 'weight'})

        # Prepare ODM
        trips = trips.dropna(subset=['trip_id'])
        trips = trips.groupby(['OPID', 'trip_id']).apply(trip_row)
        trips['origin_zip'] = trips['origin_zip'].astype('int64')
        trips['dest_zip'] = trips['dest_zip'].astype('int64')
        odms = trips.groupby(['origin_zip', 'dest_zip']).sum()['weight_trip']
        print(odms.head())
        z = self.zones.zone
        odms = odms.reindex(pd.MultiIndex.from_product([z, z], names=['ozone', 'dzone']), fill_value=0)
        print(odms.head())
        self.odm = odms / odms.sum()
