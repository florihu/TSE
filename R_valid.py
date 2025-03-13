
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
    g = sns.FacetGrid(merge, col='Commodity', row='Alloc_type', sharey=True, sharex=True, height=4, aspect=1)

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
        x_vals = [10**2, 10**11]

        ax.plot(x_vals, x_vals, '--', color='black', lw=1, label='1:1 Line')  # 1:1 Line
        ax.plot(x_vals, [v *10 for v in x_vals], '--', color='#2166ac', lw=1, label='+100%')  
        ax.plot(x_vals, [v *.1 for v in x_vals], '--', color='#b2182b', lw=1, label='-100%')  # -50% Line

        ax.plot(x_vals, [v *2 for v in x_vals], '--', color='#4393c3', lw=1, label='+100%')  
        ax.plot(x_vals, [v / 2 for v in x_vals], '--', color='#d6604d', lw=1, label='-100%')

        ax.set_xlim(x_vals[0], x_vals[1])
        ax.set_ylim(x_vals[0], x_vals[1])

        # Set log scale
        ax.set_xscale('log')
        ax.set_yscale('log')

    # redudce size of facet titles
    g.set_titles(fontsize=8)
        

    g.set_axis_labels('COP this study log10(t)','COP GMF log10(t)')

    save_fig('valid_plot.png', dpi = 800)
    plt.show()

def rel_diff_plot():
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

    merge['Rel_diff'] = ((merge['Cumprod_weight'] - merge['Cumprod_valid']) / merge['Cumprod_valid']) * 100

    #top10_countries = merge.groupby('iso3')['Cumprod_valid'].sum().sort_values(ascending=False).head(10).index

    top10_countries = ['CHN', 'RUS', 'AUS', 'CAN', 'USA', 'BRA', 'IND', 'ZAF', 'PER', 'MEX']


    merge = merge[merge.iso3.isin(top10_countries)]


    # Create FacetGrid
    g = sns.FacetGrid(merge, col='Commodity', col_wrap =3,  hue ='Alloc_type' , sharey=True, sharex=False)

    # make a horizontal barplot color positive (red) and negative (blue) deviations

    g.map(sns.barplot, 'Rel_diff', 'iso3', dodge= False)

    for ax in g.axes.flat:
        # Add a vertical line at zero
        ax.axvline(0, color='black', lw=1)
        # Get current x-axis limits and determine the maximum absolute value
        lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
        # Set symmetric limits around zero
        ax.set_xlim(-lim, lim)

    g.set_axis_labels('Relative difference (%)', 'ISO3')

    save_fig('rel_diff_plot.png', dpi = 800)
    plt.show()



if __name__ == '__main__':
    valid_plot()
    pass
