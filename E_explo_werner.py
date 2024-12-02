from D_load_werner import merge_werner
from M_prod_model import prep_data, hubbert_model, power_law, hubbert_deriv, power_law_deriv, femp_deriv, femp
from util import get_path, save_fig, save_fig_plotnine, df_to_latex, pairplot
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
import pandas as pd
plt.ion()


def prod_trans(prod):

    prod['Primary_commodity'] = prod['Commodity'].apply(lambda x: x.split('-')[0])

    return prod

def cumsum_plot(prod):
    
    prod['Waste_rock_cumsum'] = (
        prod.sort_values(['Mine', 'Year'])
            .groupby('Mine')['t waste rock']
            .cumsum()
    )
    prod['Tailings_cumsum'] = (
        prod.sort_values(['Mine', 'Year'])
            .groupby('Mine')['t tailings']
            .cumsum()
    )

    sns.scatterplot(data=prod, x='Year', y='Waste_rock_cumsum', hue='Primary_commodity', alpha=0.6, s=10)
    
    plt.yscale('log')
    plt.xlabel("Year")
    plt.ylabel("Cumulative Waste Rock log(t)")
    
    save_fig('cumsum_waste_rock.png')
    plt.show()
    plt.close()

    sns.scatterplot(data=prod, x='Year', y='Tailings_cumsum', hue='Primary_commodity', alpha=0.6, s=10)
    
    plt.yscale('log')
    plt.xlabel("Year")
    plt.ylabel("Cumulative Tailings log(t)")
    
    save_fig('cumsum_tailings.png')
    plt.show()
    plt.close()

def waste_vs_ore_grade(prod):
    prod['%Au'] = prod['g/t Au'] * 10**-4
    prod['%Ag'] = prod['g/t Ag'] * 10**-2

    com_prim_ore = {'Cu': '%Cu', 'Pb': '%Pb', 'Zn': '%Zn', 'Au': '%Au'}

    for com, prim in com_prim_ore.items():
        subset = prod[prod['Primary_commodity'] == com]  
        
        sns.jointplot(data=subset[subset[prim] > 0], x=prim, y='t waste rock', color=".3")

        
        plt.xlabel(f"{com} grade (%)")
        plt.ylabel("Waste Rock log(t)")

        save_fig(f'ore_grade_{com}_vs_waste_rock.png')
        plt.show()
        plt.close



    return None
    
def ore_vs_waste(prod):

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
      
    sns.scatterplot(data=prod, x='t ore milled', y='t waste rock', hue='Primary_commodity', alpha=0.4, s=10, size='%OC ore')

    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel(f"Ore Milled log(t)")
    plt.ylabel("Waste Rock log(t)")

    plt.tight_layout()

    save_fig(f'ore_milled_vs_waste_rock.png')
    plt.show()
    plt.close



    return None


def hist_prod(prod):
    target_cols = ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']
    
    melt = prod.melt(['Prop_name', 'Year'], value_vars=target_cols, var_name='Target', value_name='Value')
    melt['Value'] = melt['Value'] / 10**6

    plot = (ggplot(melt, aes(x='Value')) 
                + geom_histogram(bins=30, alpha=0.6)  # Histogram with density
                + facet_wrap('~Target', scales='free') 
                + scale_fill_brewer(type='qual', palette='Set2')
                + theme_minimal()
                + labs(x='Value (Mt)', y='Count')
        )
    
    save_fig_plotnine(plot, 'hist_prod.png', w=12, h=10)
    plot.draw()
    return None


def time_series(prod):
    target_cols = ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']
    
    melt = prod.melt(['Prop_name', 'Year'], value_vars=target_cols, var_name='Target', value_name='Value')
    melt['Value'] = melt['Value'] / 10**6
    melt['Year'] = melt['Year'].dt.year
    melt = melt[melt['Year'] > 1950]
    

    plot = (ggplot(melt, aes(x='Year', y='Value', color='Prop_name')) 
                + geom_point(alpha=0.6) 
                + theme_minimal()
                + labs(x='Year', y='Value (Mt)')
                + facet_wrap('~Target', scales='free')
                + theme(legend_position='none')
        )
    
    save_fig_plotnine(plot, 'time_series.png', w=12, h=10)
    plot.draw()
    return None


def pair_plot(prod):
    target_cols = ['Tailings_production', 'Waste_rock_production', 'Ore_processed_mass', 'Concentrate_production']
    
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    sns.pairplot(data = prod, vars=target_cols, diag_kind='kde')
    save_fig('pair_plot') 
    plt.show()

    return None

def plot_models():
    # Generate years and compute model outputs
    years = np.arange(1, 51, 1)
    hubbert = hubbert_model(years, 10**6, 0.1, 20)
    power = power_law(years, 10**4, 1.1)
    femp_ = femp(years, 10**6, 0.1)
    hubbert_d = hubbert_deriv(years, 10**6, 0.1, 20)
    power_d = power_law_deriv(years, 10**4, 1.1)
    femp_d = femp_deriv(years, 10**6, 0.1)

    # Combine data into a dataframe
    df = pd.DataFrame({
        'Year': years, 
        'Hubbert': hubbert, 
        'Power': power, 
        'FEMP': femp_, 
        'Hubbert_d': hubbert_d, 
        'Power_d': power_d, 
        'FEMP_d': femp_d
    })

    # Reshape data into long format
    melt = df.melt('Year', var_name='Model', value_name='Value')

    melt['Value'] = melt['Value'] / 10**3  # Convert to kt

    # Add a column to distinguish between integrals and derivatives
    melt['Type'] = melt['Model'].apply(lambda x: 'Integral' if not x.endswith('_d') else 'Derivative')
    melt['Model'] = melt['Model'].str.replace('_d', '')  # Remove '_d' from derivative model names

    # Create the plot with facets for Type (left: integral, right: derivative)
    plot = (ggplot(melt, aes(x='Year', y='Value', color='Model'))
                + geom_line()
                + theme_minimal()
                + labs(x='Year', y='Value (kt)')
                + facet_wrap('~Type', scales='free_y')  # Facet by Type
                + theme(subplots_adjust={'wspace': 0.25})  # Adjust spacing between facets
           )

    # Save the plot
    save_fig_plotnine(plot, 'theo_model_fits.png', w=6, h=6)

    plot.draw()
    return None

def main():
    # Load the production data
    werner = merge_werner()

    werner_prep = prep_data(werner)

    plot_models()




if __name__ == '__main__':

    main()