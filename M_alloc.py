
'''
This script refers to the allocation model. Data is imported. Transformed to a network and the basic network properties are studied. Further the weighted degree per commodity is calculated. The file is exported to be used in later steps.



'''
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import geopandas as gpd
import seaborn as sns
from plotnine import *  
# Create a dictionary mapping nodes to their self-loop weights, using a defaultdict to avoid key errors
from collections import defaultdict
from pyvis.network import Network

from util import save_fig, save_fig_plotnine, df_to_csv_int
from E_ml_explo import plot_network


##########################################################Params############################################################


pred_set = r'data\int\D_ml_sample\X_all_instances.csv'



##########################################################Functions############################################################
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


def alloc_com():

    df = pd.read_csv(pred_set)

    prim_based_alloc = prim_com_based_weight(df)

    # generate the network
    G = generate_network(df)

    # calculate weighted degree per commodity
    df_network_stats = calculate_weighted_degree_per_commodity(G)

    # bring everything together
    degree = relative_weighted_degree_per_mine(df, df_network_stats)

    #merge degree and prim

    merge = degree.merge(prim_based_alloc[['id_data_source', 'Commodity', 'Prim_weight']], on = ['id_data_source', 'Commodity'], how = 'left')

    df_to_csv_int(merge, 'alloc_com.csv')


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

    df.reset_index(inplace = True)
    df.rename(columns = {'index': 'Commodity'}, inplace = True)
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

def plot_network_properties():
    df = pd.read_csv(pred_set)

    # Generate the network
    G = generate_network(df)

    # Calculate weighted degree per commodity
    df = calculate_weighted_degree_per_commodity(G)

    melt = pd.melt(df, id_vars=['Commodity'], 
                   value_vars=['Weighted_degree', 'Self_loop_weight', 'Degree_centrality', 'Clustering_coeff'], 
                   value_name='value', var_name='variable')
    
    # Facet histograms for different network properties
    g = sns.FacetGrid(melt, col='variable', col_wrap=2, sharex=False, sharey=False)
    g.map_dataframe(sns.histplot, x='value', bins=20)
    g.set_titles('{col_name}')
    g.set_axis_labels("Value", "Count")
    plt.tight_layout()
    save_fig('network_properties.png')
    plt.show()
    plt.close()
    
    # Get top 30 commodities by weighted degree
    top_30 = df.nlargest(30, 'Weighted_degree').sort_values('Weighted_degree', ascending=True)
  
    top_30['Type'] = top_30['Commodity'].apply(lambda x: x.split('_')[0])

    top_30['Commodity'] = top_30['Commodity'].apply(lambda x: x.split('_')[1])
    
    # Horizontal bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_30, y='Commodity', x='Weighted_degree', hue='Type', palette='Paired')
    plt.xlabel("Weighted Degree")
    plt.ylabel("Commodity")
    plt.legend(title="Type", loc='upper right')
    plt.tight_layout()
    save_fig('top_30_commodities.png')
    plt.show()
    plt.close()
    
    return None

def interactive_network(G):
    def convert_nx_graph_for_pyvis(graph):
        # Convert nodes
        for node, data in graph.nodes(data=True):
            for key, value in data.items():
                if isinstance(value, (np.int64, np.int32)):
                    graph.nodes[node][key] = int(value)
        
        # Convert edges
        for u, v, data in graph.edges(data=True):
            for key, value in data.items():
                if isinstance(value, (np.int64, np.int32)):
                    graph.edges[u, v][key] = int(value)

        return graph

    # Convert the graph before passing it to PyVis
    G = convert_nx_graph_for_pyvis(G)

    # Calculate node degrees and add them as attribute
    node_degrees = dict(G.degree(weight='weight'))
    nx.set_node_attributes(G, node_degrees, name="Weighted_degree")

    # Create the PyVis network object
    nt = Network(notebook=False, height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True)

    # Convert the NetworkX graph to PyVis
    nt.from_nx(G)

    # Customize Node Appearance: Add degree to node labels and adjust appearance
    for node in nt.nodes:
         # Fixed size
        node['color'] = 'grey'  # Default node color
        node_id = node['id']
        node['size'] = node_degrees[node_id] / 500

        if node['size'] < 5:
            node['size'] = 5
        
        # Add degree to the label for display
        node['title'] = f"Node: {node_id}<br>Weighted_degree: {node_degrees[node_id]}"
        
        if 'Primary' in node_id:  # Adjust color for Primary commodities
            node['color'] = '#542788'  # Purple color
        elif 'Byprod' in node_id:  # Adjust color for Byprod commodities
            node['color'] = '#e08214'  # Orange color

    # Customize Edge Appearance: Standardize edge width
    for edge in nt.edges:
        edge['color'] = '#cccccc'  # Light grey color for edges
        edge['title'] = f"Edge_from_{edge['from']}_to_{edge['to']}<br>Co_occurance: {edge['width']}"
        edge['width'] = edge['width'] / 50  # Scale width
    # Add event listener for node selection and highlight connected nodes
    options = """
    var options = {
        "nodes": {
            "borderWidth": 1,
            "borderWidthSelected": 5,
            "shadow": true,
            "font": {
                "size": 12, 
                "bold": true
            }
        },
        "edges": {
            "color": {
                "color": "#cccccc",
                "highlight": "#ff0000"
            },
            "smooth": {
                "type": "continuous"
            }
        },
        "physics": {
            "enabled": true, 
            "maxVelocity": 0.01,
            "minVelocity": 0.001,
            "updateInterval": 50,
            "stabilization": {
                "enabled": true
                },
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
                "springLength": 200, 
                "springConstant": 0.05,  
                "damping": 10, 
                "gravity": -50  
                }
        }
    }
    """
    
    # Set the network options
    nt.set_options(options)

    # Save the graph to an HTML file
    nt.save_graph('com_network.html')

    pass

def relative_weighted_degree_per_mine(df, df_network_stats):
    """
    Calculates the relative weighted degree per mine.
    Returns a dictionary with the mine as the key and the relative weighted degree as the value.
    """
    # Filter commodity columns and stack the DataFrame
    coms = [col for col in df.columns if 'Primary' in col or 'Byprod' in col]
    df_mine = df.set_index('id_data_source')[coms].stack().reset_index()

    # Rename columns for clarity
    df_mine.rename(columns={'id_data_source': 'id_data_source', 'level_1': 'Commodity', 0: 'Extracted_binary'}, inplace=True)

    # Convert extracted binary column to integer
    df_mine['Extracted_binary'] = df_mine['Extracted_binary'].astype(int)

    df_mine = df_mine[df_mine['Extracted_binary'] == 1]

    # Merge with network stats on Commodity
    df_mine = df_mine.merge(df_network_stats, on='Commodity', how='left')

    # Calculate relative weighted degree
    df_mine['Weighted_product'] = df_mine['Weighted_degree'] * df_mine['Extracted_binary']
    total_weighted = df_mine.groupby('id_data_source')['Weighted_product'].transform('sum')
    df_mine['Rel_weight_degree'] = df_mine['Weighted_product'] / total_weighted

    df_mine = df_mine[['id_data_source', 'Commodity', 'Rel_weight_degree']]

    df_mine.rename(columns = { 'Rel_weight_degree': 'Occ_weight'}, inplace = True)


    return df_mine

def prim_com_based_weight(df):
    """
    Calculates the relative weighted degree per mine.
    Returns a dictionary with the mine as the key and the relative weighted degree as the value.
    """
    # Filter commodity columns and stack the DataFrame
    coms = [col for col in df.columns if 'Primary'in col]
    
    df = df.set_index('id_data_source')
    df_mine = df[coms].stack().reset_index()
    df_mine.rename(columns={'id_data_source': 'id_data_source', 'level_1': 'Commodity', 0: 'Extracted_binary'}, inplace=True)
    df_mine['Extracted_binary'] = df_mine['Extracted_binary'].astype(int)
    df_mine = df_mine[df_mine['Extracted_binary'] > 0]	

    # Equal weight for each primary commodity
    df_mine['Prim_count'] = df_mine.groupby('id_data_source')['Commodity'].transform('count')
    df_mine['Prim_weight'] = 1 / df_mine['Prim_count']


    return df_mine

def boxplot_occ_weight(file_p = r'data\int\M_alloc\alloc_com.csv.csv'):
    '''
    This function plots a boxplot of the relative weighted degree per mine.
    '''
    # Plot a boxplot of the relative weighted degree per mine
    df_rel_w = pd.read_csv(file_p)

    important_coms = df_rel_w['Commodity'].value_counts().nlargest(20).index
    
    df_rel_w = df_rel_w[df_rel_w['Commodity'].isin(important_coms)]

    f, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(data=df_rel_w, x='Commodity', y='Occ_weight', showfliers=False, palette='Paired', ax = ax)
    plt.xticks(rotation=60, size = 8)
    plt.yticks(size = 8)
    plt.ylabel('Occurance based weight')
    plt.tight_layout()
    save_fig('boxplot_occ_weight.png')
    plt.show()

    return None

def calc_sum_prim_com(df):

    # primary commodities
    coms = [col for col in df.columns if 'Primary' in col]

    df['sum_prim_com'] = df[coms].sum(axis = 1)
    
    number_of_mines = df['id_data_source'].nunique()
    non_singular_primary = len(df[df['sum_prim_com'] > 1])

    print(f'Number of mines: {number_of_mines}')
    print(f'Number of mines with more than one primary commodity: {non_singular_primary / number_of_mines * 100:.2f}%')

def circular_nx_plot(G):
    # Define a dictionary to assign colors to primary and byproduct commodities
    
    
    color_map = {}
    for col in G.nodes():
        if 'Primary' in col:
            color_map[col] = '#542788'  # Color for primary commodities
        elif 'Byprod' in col:
            color_map[col] = '#e08214'  # Color for byproduct commodities

    pos =  nx.circular_layout(G) # Positioning the nodes
    
    plt.figure(figsize=(14, 14))

    # Extract edge weights
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]  # Edge weights for visualization

     # min max scale weights
    weights = StandardScaler().fit_transform(np.array(weights).reshape(-1, 1)).flatten()

    # Draw nodes and edges with the color map for nodes
    node_colors = [color_map[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.8, edge_color='gray')

    
    plt.axis('off')  # Hide axes
    plt.tight_layout()

    # Save the figure
    save_fig('com_network_colored.png')

    # Show the plot
    plt.show()

    pass

def main():
    df = gpd.read_file(r'data\int\D_train_tetst_df\features_all_mines_all_coms.gpkg')

    prim_com_based_weight(df)

    pass



if __name__ == '__main__': 
    plot_network_properties()
    pass