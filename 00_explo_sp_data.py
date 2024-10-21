import geopandas as gpd
import pandas as pd



def explo(path_sp, path_prod):


    sp = gpd.read_file(path_sp)

    prod = pd.read_csv(path_prod)

    return None

if __name__ == '__main__':
    path_sp = r'data\mine-comm\snl_mining_properties.gpkg'

    path_prod = r'data\mine-comm\snl_production_values.csv'

    explo(path_sp, path_prod)
