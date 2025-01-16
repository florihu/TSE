
'''
This script refers to the allocation model. Data is imported. Transformed to a network and the basic network properties are studied. Further the weighted degree per commodity is calculated. The file is exported to be used in later steps.



'''
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import seaborn as sns
from plotnine import *  
# Create a dictionary mapping nodes to their self-loop weights, using a defaultdict to avoid key errors
from collections import defaultdict

from util import save_fig, save_fig_plotnine, data_to_csv_int
from E_sample_explo import plot_network


def generate_network(df):
    '''
    
    This function generates a network from the input data. The data is first transformed to a binary format. The co-occurrence matrix is calculated. The network is built based on the co-occurrence matrix. The function returns the network.
    '''

    # Identify vars with Primary or Byprod in the name
    com_vars = [col for col in df.columns if 'Primary' in col or 'Byprod' in col]


    # Convert selected columns to binary format (0 or 1)
    df_b = df[com_vars].astype(int)

    # Filter columns where the sum of non-zero values is greater than zero
    df_b = df_b.loc[:, (df_b.sum(axis=0) != 0)]

    # Compute the co-occurrence matrix
    co_occurrence_matrix = df_b.T.dot(df_b)

    # Step 2: Build a NetworkX graph
    G = nx.Graph()

    # Add nodes from the columns of df_b (commodities)
    G.add_nodes_from(df_b.columns)

    
    for i in range(len(co_occurrence_matrix.columns)):
        # Get the diagonal value (total occurrences of commodity i)
        diagonal = co_occurrence_matrix.iloc[i, i]

        # Compute row sum (total co-occurrences including self)
        row_sum = co_occurrence_matrix.iloc[i, :].sum()

        # Calculate singular occurrences
        singular_occurrences = diagonal - (row_sum - diagonal)

        # Add a self-loop with the singular occurrences weight
        if singular_occurrences > 0:  # Add only if singular occurrences are positive
            G.add_edge(co_occurrence_matrix.columns[i], co_occurrence_matrix.columns[i], weight=singular_occurrences)

        # Add edges for co-occurrences
        for j in range(i + 1, len(co_occurrence_matrix.columns)):
            weight = co_occurrence_matrix.iloc[i, j]
            if weight > 0:
                G.add_edge(co_occurrence_matrix.columns[i], co_occurrence_matrix.columns[j], weight=weight)
    

    return G

def calculate_weighted_degree_per_commodity(G):
    '''
    This function calculates the weighted degree per commodity. The function returns a dictionary with the commodity as key and the weighted degree as value.
    '''

    # Calculate the weighted degree per commodity
    weighted_degree_per_commodity = dict(G.degree(weight='weight'))
    degree_centralities = nx.degree_centrality(G)
    clustering_coeff = nx.clustering(G, weight='weight')
    
    adj_matrix = nx.to_numpy_array(G, weight='weight')

    # Extract the diagonal values (self-loop weights)
    self_loop_weights = np.diag(adj_matrix)

    df = pd.DataFrame.from_dict(weighted_degree_per_commodity, orient='index', columns=['Weighted_degree'])

    # Add self-loop weights to the DataFrame
    df['Self_loop_weight'] = pd.Series(self_loop_weights.flatten(), index=df.index)
    

    df['Degree_centrality'] = pd.Series(degree_centralities)

    # Ignores self loops
    df['Clustering_coeff'] = pd.Series(clustering_coeff)

    # Save the DataFrame to a csv file
    data_to_csv_int(df, 'network_stats_per_commodity')

    return df

def degree_distribution(G):
    '''
    This function calculates the degree distribution of the network. The function returns a dictionary with the degree as key and the number of nodes with that degree as value.
    '''
    # Calculate the degree distribution
    degree_distribution = dict(G.degree(weight='weight'))

    # plot the degree distribution
    p = ggplot(pd.DataFrame(degree_distribution, index=['Degree']).T, aes(x='Degree')) + geom_histogram(bins=20) + theme_minimal()

    save_fig_plotnine(p, 'weighted_degree_distribution.png')

    return degree_distribution

def plot_network_properties(df):
    df.reset_index(inplace = True)
    df.rename(columns = {'index': 'Commodity'}, inplace = True)


    melt = pd.melt(df, id_vars=['Commodity'], value_vars=['Weighted_degree', 'Self_loop_weight', 'Degree_centrality', 'Clustering_coeff'], value_name='value')


    # do hist plots for the different network properties facet wrapped
    p = (ggplot(melt, aes(x='value')) + geom_histogram(bins = 20) + facet_wrap('variable', scales='free') + theme_minimal())

    save_fig_plotnine(p, 'network_properties_hist.png')
    
    # do a plot with the 30 highest commoditeis by weighted degree and show all network properties
    top_30 = df.nlargest(30, 'Weighted_degree')

    # sorted
    top_30 = top_30.sort_values('Weighted_degree', ascending = True)

    # color dict for product and byproduct differntiation
    color_dict = {'Primary': '#542788', 'Byprod': '#e08214'}
    top_30['Type'] = top_30['Commodity'].apply(lambda x: x.split('_')[0])

    #make bar chart horizontal
    p1 = (ggplot(top_30, aes(x='Commodity', y='Weighted_degree', fill = 'Type')) + geom_bar(stat='identity') + theme_minimal() + coord_flip() + scale_fill_manual(values = color_dict))

    # set legend off
    p1 = p1 + theme(legend_position='none')

    save_fig_plotnine(p1, 'top_30_commodities_weighted_network_bar.png')


    return None


def interactive_network(G)
    pass


def main():
    df = gpd.read_file(r'data\int\D_train_tetst_df\features_all_mines_all_coms.gpkg')

    G = generate_network(df)
    df = calculate_weighted_degree_per_commodity(G)
    plot_network_properties(df)

    pass

if __name__ == '__main__': 
    main()