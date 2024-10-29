from load_werner import load_prod, clean_area, clean_production, prod_keys_exclude, file_name
from util import get_path, save_fig
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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

    

    


if __name__ == '__main__':
    # Load the production data
    path = get_path(file_name)
    area, prod = load_prod(path)

    prod_clean = clean_production(prod, prod_keys_exclude)

    prod_t = prod_trans(prod_clean)

    waste_vs_ore_grade(prod_t)

    



