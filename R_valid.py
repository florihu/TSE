
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text 

from R_prediction import get_countries_pred
from util import save_fig


# Params
p_ore = r'data\gru_metal_extraction\Metal_extraction_1970-2022_20240704.xlsx'
p_com_con = r'data\gru_metal_extraction\com_concordance.xlsx'


# Functions


def get_ore_time_alloc():
    ore = pd.read_excel(p_ore)
    conc = pd.read_excel(p_com_con)

    merge = ore.merge(conc, on ='material_id')
    merge = merge.dropna(subset=['Commodity'])

    merge = merge[merge.year <= 2019]

    agg = merge.groupby(['country_name', 'Commodity'])['value'].sum().reset_index()

    agg.rename(columns={'value': 'Cumprod_valid', 'country_name':'Country_name'}, inplace=True)

    return agg

def valid_plot():
    this_study = get_countries_pred()
    this_study.rename(columns={'name': 'Country_name'}, inplace=True)
    valid = get_ore_time_alloc()

    this_study = this_study[
        (this_study['Commodity'].isin(['Copper', 'Nickel', 'Zinc'])) &
        (this_study.Target_var == 'Ore_processed_mass')
    ]

    # Merge
    merge = this_study.merge(valid, on=['Country_name', 'Commodity'], how='left')
    merge['Cumprod_valid'] = merge['Cumprod_valid'].fillna(0)

    #merge['Cumprod_weight'] /= 10**9
    #merge['Cumprod_valid'] /= 10**9

    # Create FacetGrid
    g = sns.FacetGrid(merge, col='Commodity', row='Alloc_type', sharey=True, sharex=True)

    def scatter_with_labels(data, x, y, **kwargs):
        """Custom function to plot scatter and add ISO3 annotations."""
        ax = plt.gca()
        sns.scatterplot(data=data, x=x, y=y, alpha=0.3, ax=ax, **kwargs)

        texts = []
        for _, row in data.iterrows():
            texts.append(ax.text(row[x], row[y], row['iso3'], fontsize=6, ha='center', va='baseline'))

        

    g.map_dataframe(scatter_with_labels, x='Cumprod_weight', y='Cumprod_valid')

    for ax in g.axes.flat:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        min_val = min(xlim[0], ylim[0])
        max_val = max(xlim[1], ylim[1])
        x_vals = [10**5, 10**11]

        ax.plot(x_vals, x_vals, '--', color='black', lw=1, label='1:1 Line')  # 1:1 Line
        ax.plot(x_vals, [v *10 for v in x_vals], '--', color='red', lw=1, label='factor 10')  
        ax.plot(x_vals, [v * 0.1 for v in x_vals], '--', color='blue', lw=1, label='factor .1')  # -50% Line

        # Set log scale
        ax.set_xscale('log')
        ax.set_yscale('log')

    # redudce size of facet titles
    g.set_titles(fontsize=8)
        

    g.set_axis_labels('COP validated log10(t)', 'COP predicted log10(t)')

    save_fig('valid_plot.png', dpi = 800)
    plt.show()

if __name__ == '__main__':
    valid_plot()
    pass
