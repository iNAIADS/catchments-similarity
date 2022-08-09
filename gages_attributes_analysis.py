import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from utils import *
from gages_utils import load_gages_dataset
from find_optimal_dimension import wrapper as optimal_dimension
import matplotlib.pyplot as plt
import json

import warnings
warnings.filterwarnings("ignore")

NOT_FOR_SIM = ['STAID', 'HUC02','LAT_GAGE','LNG_GAGE','STATE', 'CLASS']

make_similarity = make_cosine_similarity


if __name__ == '__main__':
    
    df, cols_for_similarity = load_gages_dataset(cols_not_for_similarity=NOT_FOR_SIM)

    df_for_similarity = df[cols_for_similarity]
    df_standardized = standardize_df(df_for_similarity)
    M = df_standardized.values
    print(M.shape)
    
    U, S, VT = np.linalg.svd(M, full_matrices=True)
    reconstruction_error = [np.sqrt(np.sum((M - U[:,:k].dot(np.diag(S[:k])).dot(VT[:k,:]))**2)) for k in np.arange(len(S))+1]
    k=20
    #k = optimal_dimension(U, S, VT, make_similarity)
    #k = find_optimal_rank_from_cum_var(S, target_cum_var=0.7, reconstruction_error=reconstruction_error)
    print(k)
    
    M_transf = U[:,:k].dot(np.diag(S[:k]))
    PCs = VT[:k, :].T
    M_reconstructed = U[:,:k].dot(np.diag(np.array(S[:k]))).dot(VT[:k,:])
    

    gauges_similarity = make_similarity(M_transf, between_0_1=True)
    attr_similarity = make_similarity(PCs, between_0_1=True)

    #alpha_attr = 0.18836490894898011 # k=12, target=0.95
    alpha_attr = 0.21627185237270202 # k=20, target=0.95
    #alpha_attr = 0.29512092266663853 # k=40, target=0.95
    #alpha_attr = make_backbone_wrapper(attr_similarity, np.logspace(-2,0,401), target=.95, suffix='attr', reciprocated=False); print(alpha_attr)
    
    #alpha_gauges = 0.1548816618912481 #k=12, target=0.95
    alpha_gauges = 0.1548816618912481 # k=20, target=0.95
    #alpha_gauges = 0.15848931924611134 # k=40, target=0.95
    #alpha_gauges = make_backbone_wrapper(gauges_similarity, np.logspace(-1,0,101), target=.95, suffix='gauges', reciprocated=False); print(alpha_gauges)
    

    G_V = make_backbone_network(attr_similarity, cols_for_similarity, alpha = alpha_attr, reciprocated=False)
    _, communities_attr, communities_attr_to_nodes_dict = make_community(G_V, return_inv=True)
    '''
    with open('attributes_in_clusters.json', 'w') as f:
        json.dump({i:[nx.get_node_attributes(G_V, "label")[x] for x in j] for i,j in communities_attr_to_nodes_dict.items()}, f)
    f.close()
    '''

    attr_degrees = nx.degree_centrality(G_V)
    attr_btw = nx.betweenness_centrality(G_V)
    attr_clustering = nx.clustering(G_V)
    attr_comm_max_degree = {i:np.max([attr_degrees[l] for l in j]) for i,j in communities_attr_to_nodes_dict.items()}
    attr_degrees_norm = {i:np.nan_to_num(1.*j/attr_comm_max_degree[communities_attr[i]], nan=1.) for i,j in attr_degrees.items()}

    quantities = [
        {"name":"communities", "values":communities_attr, "discrete":True},
        {"name":"degree", "values":attr_degrees_norm, "discrete":False},
        {"name":"betweenness", "values":attr_btw, "discrete":False},
        {"name":"clustering", "values":attr_clustering, "discrete":False},
        #{"name":"pc1", "values":dict(enumerate(np.abs(V[:,0]).tolist())), "discrete":False},
        #{"name":"all_pcs", "values":dict(enumerate(np.sum(np.abs(V), axis=1).tolist())), "discrete":False}
        ]
 
    plot_network(G_V, quantities, nodesize=None, suffix='attributes')
    list_communities_attr = [[nx.get_node_attributes(G_V, "label")[i] for i in j] for _,j in communities_attr_to_nodes_dict.items()]

    gauges_id = df['STAID']
    G = make_backbone_network(gauges_similarity, gauges_id, alpha = alpha_gauges, reciprocated=False)
    _, communities, communities_inv = make_community(G, return_inv=True)

    gages_clustering = nx.clustering(G)

    '''
    with open('gages_adjacency_matrix_full_network.npy', 'wb') as f:
        np.save(f, gauges_similarity)
        np.save(f, gauges_id)

    gages_id_to_node_id = {j:i for i,j in nx.get_node_attributes(G, "label").items()}
    with open('gages_adjacency_matrix_backbone_network.npy', 'wb') as f:
        np.save(f, nx.adjacency_matrix(G, nodelist=[gages_id_to_node_id[i] for i in gauges_id]))
        np.save(f, gauges_id)
    
    with open('gages_in_clusters.json', 'w') as f:
        json.dump({i:[nx.get_node_attributes(G, "label")[x] for x in j] for i,j in communities_inv.items()}, f)
    f.close()
    
    
    
    nodes_id_giant_comp = list(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    with open('communities_SVD.npy', 'wb') as f:
        np.save(f, make_community_adjacency_matrix(communities))
        np.save(f, np.array(nodes_id_giant_comp))
    '''
    
    plot_distribution([list(gages_clustering.values()), list(attr_clustering.values())],
        name_vals=['catchments', 'attributes'],
        bins_number = [21, 21],
        suffix_filename='clustering_coefficient')


    plot_distribution([[len(i) for i in communities_inv.values()], [len(j) for j in communities_attr_to_nodes_dict.values()]],
        name_vals=['catchments', 'attributes'],
        bins_number = [11, 11],
        suffix_filename='communities_size',
        loglog=True,
        same_binning=True)

    quantities = [{"name":"communities", "values":communities, "discrete":True}]

    plot_network(G, quantities, nodesize=None, suffix='catchments', do_plot_labels=False)
    
    
    gauges_to_nodes = {j:i for i,j in nx.get_node_attributes(G, "label").items()}
    df["community"] = [communities[gauges_to_nodes[x]] for x in gauges_id]
    community_counts_dict = df.community.value_counts().to_dict()
    
    plot_WQ_dynamic(df, community_counts_dict)    

    
    nodes_id_giant_comp = list(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    df_giant_comp = df.iloc[nodes_id_giant_comp, :]
    
    feature_importance = dict(enumerate(get_features_importance_from_forest(M[nodes_id_giant_comp, :], df_giant_comp["community"].values)))
    print([cols_for_similarity[b[0]] for b in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)])
    
    quantities = [{"name": "feature_importance", "values":feature_importance, "discrete":False}]
    plot_network(G_V, quantities, suffix='attributes')

    cumulative_importance = 0
    nodes_included_so_far = []
    attributes_list = []
    while len(nodes_included_so_far)<len(G_V):
        top_nodes_importance = {}
        top_nodes_neighborhood_dict = {}
        nodes_to_consider = [node for node in G_V.nodes() if node not in nodes_included_so_far]
        for node in nodes_to_consider:
            nodes_in_same_community = communities_attr_to_nodes_dict[communities_attr[node]]
            neighbors_and_top = [node]+list(G_V[node])
            neighbors_and_top_minus_already_seen = [i for i in neighbors_and_top if ((i not in nodes_included_so_far) and (i in nodes_in_same_community))]
            top_node = sorted({i: feature_importance[i] for i in neighbors_and_top_minus_already_seen}.items(), key=lambda item: item[1])[-1][0]
            importance_top_and_neighbors = np.sum([feature_importance[i] for i in neighbors_and_top_minus_already_seen])
            top_nodes_importance[top_node] = importance_top_and_neighbors
            top_nodes_neighborhood_dict[top_node] = neighbors_and_top_minus_already_seen

        chosen_node = sorted(top_nodes_importance.items(), key=lambda item: item[1])[-1][0]
        cumulative_importance += top_nodes_importance[chosen_node]#*len(top_nodes_neighborhood_dict[chosen_node])
        nodes_included_so_far.extend(top_nodes_neighborhood_dict[chosen_node])
        print(cols_for_similarity[chosen_node], cumulative_importance, len(nodes_included_so_far))
        attributes_list.append(cols_for_similarity[chosen_node])
    print(attributes_list)

       
    plot_gauges_in_map_dots(df, lat_col='LAT_GAGE', lon_col='LNG_GAGE', column='community')
    plot_gauges_in_map_voronoi(df, lat_col='LAT_GAGE', lon_col='LNG_GAGE', column='community')
    
    community_to_states = df.groupby('community')['STATE'].agg([list]).to_dict()['list']
    community_to_states = {i:Counter(j).most_common(5) for i,j in community_to_states.items()}

    df_community_anomalies = make_df_community_anomalies(df[[x for x in df.columns.values if x not in NOT_FOR_SIM]])

    for community in [i for i,j in community_counts_dict.items() if j>=50]:
        plot_network(G_V,
            [{"name":"anomalies", "values":dict(enumerate(df_community_anomalies.loc[community, :].values)), "discrete":False}],
            suffix='anomalies_%d_cluster'%community,
            bicolor=True, nodesize=dict(enumerate(np.abs(df_community_anomalies.loc[community, :].values))))

    plot_anomalies(df_community_anomalies, community_counts_dict, community_to_states=community_to_states, min_community_size=50)