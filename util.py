import os
import inspect
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import itertools
from plotnine import ggplot, geom_point, facet_wrap, facet_grid, aes
import geopandas as gpd
def get_path(name):
    '''
    Get the path of the specified file or folder within the root directory.
    Searches recursively starting from the directory of the current script.
    
    Parameters:
    - name: str, the name of the file or folder to search for.

    Returns:
    - str or None: the full path to the file or folder if found, otherwise None.
    '''
    # Set the root directory to the directory where this script is located
    root_directory = Path(__file__).resolve().parent
    
    # Use rglob to recursively search for files and directories with the given name
    for path in root_directory.rglob(name):
        return str(path)  # Return the first match as a string
    
    # If not found, return None
    return None

def save_fig(name, dpi=600):
    base_folder = 'fig'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, name)

    plt.savefig(file_path, dpi=dpi)

    return None

def save_fig_plotnine(plot, name, w=8, h=6, dpi=600, format='png'):
    base_folder = 'fig'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, name)

    plot.save(file_path, format = format, width=w, height=h, dpi=dpi)

    return None

def df_to_latex(df, filename, multicolumn=False, longtable=False):
    base_folder = 'tab'

    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Convert DataFrame to LaTeX table format
    latex_table = df.to_latex(float_format="%.2f", multicolumn=multicolumn, longtable=longtable)
    # Write LaTeX table to a .tex file
    with open(f'{path}/{filename}.tex', 'w') as f:
        f.write(latex_table)
    return None

def df_to_gpkg(df, filename, crs):
    '''
    Save a DataFrame to a GeoPackage file in the data/gpkg folder. given it is a Geodataframe object
    '''
    base_folder = 'data/int'

    assert 'geometry' in df.columns, 'The DataFrame must have a geometry column to be saved as a GeoPackage file.'
    # transform to gdf
    df = gpd.GeoDataFrame(df, geometry='geometry', crs = crs)

    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, filename + '.gpkg')

    df.to_file(file_path, driver='GPKG')

    return None

def df_to_csv_int(data, name):
    ''' 
    
    Save data to a csv file in the data/int folder.
    Parameters:
    - data: pd.DataFrame, the data to save.
    - name: str, the name of the file to save the data to.

    Returns:
    - None

    '''
    base_folder = r'data\int'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, name + '.csv')

    data.to_csv(file_path)

    return None


def df_to_excel(filename, df, sheet_name):
    base_folder = r'data\int'
    # Get the calling script’s filename
    calling_script = inspect.stack()[1].filename
    script_name = os.path.basename(calling_script).replace('.py', '')

    path = os.path.join(base_folder, script_name)
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, filename + '.xlsx')

    # Use 'a' mode if file exists; otherwise, create a new file with 'w' mode.
    if os.path.exists(file_path):
        mode = 'a'
        sheet_option = 'replace'
    else:
        mode = 'w'
        sheet_option = None  # Not needed when creating a new file

    with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists=sheet_option) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=True)




def get_data_for_column_combo(col1, col2, source, color_column, shape_column):
    '''
    Get the required data for a combination of two columns. Each call to this functions generates
    the data for one of the subplots in the pairplot. Color and shape data are appended as needed. 
    '''
    col_data = (source[[col1, col2]]
            .rename(columns={col1: 'values1', col2: 'values2'})
            .assign(col1=col1, col2=col2))
    if not color_column is None:
        col_data['color_column'] = source[color_column]
    if not shape_column is None:
        col_data['shape_column'] = source[shape_column]
    return col_data

def get_point_args(color_column, shape_column):
    '''
    Generate the appropriate input arguments to our geom_point. The names of the
    columns are fixed as these are generated in a standard way by `get_data_for_column_combo`. 
    But which should be included varies based on wheter or not color and shape are passed. 
    '''
    point_args = dict(x='values1', y='values2')

    if color_column is not None:
        point_args['color'] = 'color_column'
    if shape_column is not None:
        point_args['shape'] = 'factor(shape_column)'

    return point_args

def pairplot(source, columns, color_column=None, shape_column=None, use_facet_grid=False):
    '''
    This function creates a pairplot from the data in `source` based on the columns listed in `columns. 
    Optional arguments included passing a color and shape variabele, those will then determine the color and
    shape in the resulting pairplot. 

    By default we use `facet_wrap` as this allows us to use `scales='free'`. This is not how a pairplot
    usually works, so there is an option to force the use of `facet_grid` to get a more traditional plot.
    '''
    plot_data = pd.concat([
        get_data_for_column_combo(col1, col2, source, color_column, shape_column) for col1, col2 in itertools.permutations(columns, 2)
    ])
    gg = ggplot(plot_data) + geom_point(aes(**get_point_args(color_column, shape_column)), alpha=0.4, size=2) 
    if use_facet_grid:
        return gg + facet_grid('col1 ~ col2')
    else:
        return gg + facet_wrap('~ col1 + col2', scales='free')
    

def get_world_bounds(crs):
    world_bounds_p = r'data\world_bound\world-administrative-boundaries.shp'
    wb = gpd.read_file(world_bounds_p)
    wb.to_crs(crs, inplace=True)
    return wb
