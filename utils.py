import pandas as pd
import geopandas as gpd
import numpy as np
import os.path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from copy import copy
import networkx as nx
import community as community_louvain
from infomap import Infomap
import matplotlib.colors as mcolors
from networkx.drawing.nx_pydot import graphviz_layout
from shapely.geometry import Polygon, MultiPoint, Point
from scipy.interpolate import interp1d
from scipy import stats
import itertools
from skmisc.loess import loess
from scipy.stats import ttest_1samp, ttest_ind
from scipy.stats import f_oneway
from scipy.stats import shapiro
from scipy.stats import kruskal
from scipy.stats import spearmanr, pearsonr
from scipy.special import betainc
from scipy.spatial.distance import cdist
from collections import OrderedDict, Counter
from scipy.spatial import distance



np.random.seed(42)

COLORS = [np.array(x)/255. for x in [(31, 119, 180), (255, 127, 14), (44, 160, 44),
    (214, 39, 40), (148, 103, 189), (140, 86, 75), (227, 119, 194),
    (127, 127, 127), (188, 189, 34), (23, 190, 207), (174, 199, 232),
    (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
    (196, 156, 148), (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229)]]


def load_map(filename_conus = 'datasets/cb_2018_us_nation_20m/cb_2018_us_nation_20m.shp',
    filename_states = 'datasets/cb_2018_us_state_20m/cb_2018_us_state_20m.shp'):
    conus = gpd.read_file(filename_conus)
    states = gpd.read_file(filename_states)
    areas = [a.area for a in conus.geometry.values[0]]
    greatest_area_idx = np.argmax(areas)
    #print(conus)
    conus = conus.explode().reset_index()
    conus = conus[conus['level_1']==greatest_area_idx]
    conus_states = gpd.overlay(conus, states)
    #print(conus_states)
    return conus, conus_states


def standardize_df(df):

    df[df.columns] = standardize_M(df.values)

    return df

def standardize_M(M):

    return StandardScaler().fit_transform(M)


def make_cosine_similarity(M, between_0_1=True):
    
    M = normalize(M, axis=1, norm='l2')

    if between_0_1:
        S = (M.dot(M.T)+1.)/2
    else:
        S = M.dot(M.T)
    
    np.fill_diagonal(S, 0.)

    return S


def make_backbone_network(M, labels=[], alpha=0.05, reciprocated=False):

    np.fill_diagonal(M, 0.)
    n = M.shape[0]
    k = np.tile(np.sum(M.astype(bool), axis=1).reshape(n, 1), (1,n))
    M_norm = normalize(M, axis=1, norm='l1')


    M_alpha = copy(M)

    idx_to_zero = (1-M_norm)**(k-1)>=alpha
    #idx_to_zero = M_norm<alpha
    if reciprocated:
        idx_to_zero = idx_to_zero | idx_to_zero.T
    else:
        idx_to_zero = idx_to_zero & idx_to_zero.T
    M_alpha[idx_to_zero]=0.

    G = nx.from_numpy_array(M_alpha)

    for i,label in enumerate(labels):
        G.add_node(i, label=label)

    return G


def make_community(G, method="infomap", return_inv=False, do_sort=True):
    """
    Generate cluster communities according to the louvain algorithm
    
    PARAMS:
        G (networkx graph): undirected networkx format graph

    RETURNS:
        communities_names (dict): dictionary of node names and assigned cluster
    """
    if method =='louvain':
        communities = community_louvain.best_partition(G)
        modularity_best_partition = community_louvain.modularity(communities, G)
    elif method == 'infomap':
        communities = {}
        im = Infomap(silent=True, two_level=True, num_trials=20, flow_model="undirected")
        mapping = im.add_networkx_graph(G)
        im.run()
        for node in im.nodes:
            communities[mapping[node.node_id]] = node.module_id
        #communities_2 = im.get_modules()
        #assert(communities==communities_2)
        modularity_best_partition = im.codelength


    if do_sort:
        communities_inv = {}
        for i,j in communities.items():
            communities_inv.setdefault(j, []).append(i)

        communities_list = list(communities_inv.values())
        communities_inv = {i:communities_list[y] for i,y in enumerate(np.argsort([len(x) for x in communities_list])[::-1])}
        
        communities = {}
        for i,j in communities_inv.items():
            for k in j:
                communities[k] = i

        if return_inv:
            return modularity_best_partition, communities, communities_inv

        else:
            return modularity_best_partition, communities
    else:
        if return_inv:
            communities_inv = {}
            for i,j in communities.items():
                communities_inv.setdefault(j, []).append(i)
            return modularity_best_partition, communities, communities_inv
        else:
            return modularity_best_partition, communities            


def make_community_adjacency_matrix(communities):

    M = np.zeros(shape=(len(set(list(communities.values()))), len(communities))).astype(int)

    for i,j in communities.items():
        M[j,i] = 1
    return M



def make_df_community_anomalies(df, do_fillna=None):
    df_mean = df.mean(axis=0)
    df_std = df.std(axis=0)

    df_community = df.groupby('community').agg(['mean'])

    for col in [x for x in df_mean.index.values if x != 'community']:
        df_community[(col, 'ttest')] = (df_community[(col, 'mean')] - df_mean[col])/df_std[col]
        df_community.drop(columns=[(col, 'mean')], inplace=True)
    if do_fillna is not None:
        df_community.fillna(do_fillna, inplace=True)

    #df_community = df_community[df_community.index.isin([i for i,j in community_counts_dict.items() if j>=min_community_size])]
    
    return df_community



def plot_anomalies_agg(df, labels, dict_attribute_cluster, dict_len_attribute_cluster, count_dict,min_community_size=3, suffix=''):
    
    markers = [u"\u2B24", "X", u"\u25B2", u"\u25A0", u"\u25C6", u"\u002B"]

    M_anomalies_agg = []

    communities_to_plot = [i for i,j in count_dict.items() if ((j>=min_community_size) and (i in df.index.values))]
    num_communities = len(communities_to_plot)
    if type(df.columns.values[0])==tuple:
        df_columns_names = [x[0] for x in df.columns.values]
    else:
        df_columns_names = df.columns.values

    to_write_in_report = []
    for i, community in enumerate(communities_to_plot):
        fig, ax = plt.subplots(1, figsize=(30,20), facecolor=(1, 1, 1))
        color= COLORS[community%len(COLORS)]
        #color= COLORS[i%len(COLORS)]
        
        t_test = np.zeros(len(set(list(dict_attribute_cluster.values()))))
        for j,jj in enumerate(df.loc[community, :].values):
            t_test[dict_attribute_cluster[df_columns_names[j]]] += 1.*jj/dict_len_attribute_cluster[dict_attribute_cluster[df_columns_names[j]]]
        M_anomalies_agg.append(t_test)

        ax.bar(np.arange(len(t_test)), t_test, color=color, label='%d'%(community))
        ax.set_xticks(np.arange(len(t_test)))
        ax.set_xticklabels(labels, fontdict={'fontsize': 36})
        #ax.set_ylim(-4,4)
        ax.set_ylabel(r"$\bar{z}$-score", fontdict={'fontsize': 36})
        ax.text(0.8, 0.9, "map symbol     ", transform=ax.transAxes, fontsize=32, color='k', bbox=dict(facecolor='none', edgecolor='k', pad=20.0))
        ax.text(0.905, 0.9, markers[(i//len(COLORS))%len(markers)], transform=ax.transAxes, fontsize=36, color=color)
        plt.yticks(fontsize=32)
        plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
        plt.tight_layout()
        plt.savefig('anomalies_agg_%d_%s' % (community, suffix), bbox_inches="tight")
        #plt.close()
        
    return pd.DataFrame(data=M_anomalies_agg, columns=labels)
    


def plot_anomalies_agg_diff(df, labels, communities_to_plot, dict_attribute_cluster, dict_len_attribute_cluster, count_dict,min_community_size=3, suffix=''):


    num_communities = len(communities_to_plot)

    if type(df.columns.values[0])==tuple:
        df_columns_names = [x[0] for x in df.columns.values]
    else:
        df_columns_names = df.columns.values

    t_test_M = []
    color_1= COLORS[communities_to_plot[0]%len(COLORS)]
    color_2= COLORS[communities_to_plot[1]%len(COLORS)]

    for community in communities_to_plot:
        t_test = np.zeros(len(set(list(dict_attribute_cluster.values()))))
        for j,jj in enumerate(df.loc[community, :].values):
            t_test[dict_attribute_cluster[df_columns_names[j]]] += 1.*jj/dict_len_attribute_cluster[dict_attribute_cluster[df_columns_names[j]]]
        t_test_M.append(t_test)
    t_test_diff = t_test_M[0]-t_test_M[1]

    mask = t_test_diff > 0

    fig, ax = plt.subplots(1, figsize=(30,20), facecolor=(1, 1, 1))
    
    ax.bar(np.arange(len(t_test_diff))[mask], t_test_diff[mask], color=color_1, label='%d'%(community))
    ax.bar(np.arange(len(t_test_diff))[~mask], t_test_diff[~mask], color=color_2, label='%d'%(community))
    #ax.bar(np.arange(len(t_test_diff)), t_test_diff, color='k', label='%d'%(community))
    ax.set_xticks(np.arange(len(t_test_diff)))
    ax.set_xticklabels(labels, fontdict={'fontsize': 36})
    ax.set_ylabel(r"$\bar{z}$-score", fontdict={'fontsize': 36})
    plt.yticks(fontsize=32)
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    plt.tight_layout()
    plt.savefig('anomalies_agg_%d_%d_%s' % (communities_to_plot[0], communities_to_plot[1], suffix))
    #plt.close()
        



def plot_network(G, quantities, nodesize=None, suffix='', do_plot_labels=True, bicolor=False, add_cbar=True, cbar_label=None, label_size=7.5):


    nodelist = G.nodes()
    edgelist = G.edges()

    if len(nodelist)>1000:
        layout_engine="sfdp"
    else:
        layout_engine="fdp"
        #layout_engine="neato"

    if nodesize==None:
        nodesize = 100
    elif type(nodesize) == dict:
        nodesize = [100*nodesize[i] for i in nodelist]

    pos = nx.nx_agraph.graphviz_layout(G, prog=layout_engine)

    labels = nx.get_node_attributes(G, "label")
    for quantity in quantities:
        quantity_name = quantity["name"]
        quantity_values = quantity["values"]
        if cbar_label==None:
            cbar_label = quantity_name
        
        fig, ax = plt.subplots(figsize=(10,10), facecolor=(1, 1, 1))


        if bicolor:
            cmap = plt.cm.coolwarm
        else:
            cmap = plt.cm.Reds


        if quantity["discrete"]:
            #d_color = {j:i for i,j in enumerate(sorted(set(list(quantity_values.values()))))}
            #colorlist = [COLORS[d_color[quantity_values[i]]%len(COLORS)] for i in nodelist]
            colorlist = [COLORS[quantity_values[i]%len(COLORS)] for i in nodelist]
            nx.draw_networkx_nodes(G, pos,
                nodelist=nodelist,
                node_size=nodesize,
                node_color=colorlist,
                edgecolors='k',
                linewidths=.5)
        else:
            vmin = np.min(list(quantity_values.values()))
            vmax = np.max(list(quantity_values.values()))
            vave = np.mean(list(quantity_values.values()))

            #norm_ = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            #print(norm)

            drawn_nodes = nx.draw_networkx_nodes(G, pos,
                nodelist=nodelist,
                node_size=nodesize,
                #node_color=[norm_(quantity_values[i]) for i in nodelist],
                node_color=[quantity_values[i] for i in nodelist],
                edgecolors='k',
                linewidths=.5,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax)

        nx.draw_networkx_edges(G, pos,
            edgelist=edgelist, alpha=0.3, width=1)
        
        if do_plot_labels:
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=label_size)


        if (quantity["discrete"]==False) and add_cbar:
            
            #sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            try:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
            except:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vave, vmax=vmax))

            sm.set_array([])
            cbar = plt.colorbar(sm, shrink=0.5)#, fraction=0.047, pad=0.04)
            cbar.ax.tick_params(labelsize=16)
            cbar.ax.set_ylabel(cbar_label, rotation=90, fontdict={'fontsize': 16})


        plt.axis('off')

        plt.savefig('network_%s_%s' % (suffix, quantity_name[:50]), dpi=300)
        #plt.close()


def plot_gauges_in_map_dots(df, lat_col, lon_col, column = 'HUC02',suffix='', discrete=True, highlighted_sites=None, values_for_cmap=None, reverse_map=False, title='', filenames=None, in_conus=True, with_states_boundaries=False):

    markers = ["o", "X", "^", "s", "D", "P"]
    if in_conus:
        conus, states = load_map()
    #if filename is not None:
    #    map_ = gpd.read_file(filename)

    #df = copy(df_orig)
    #df[column+'_count'] = df.groupby(column)[column].transform('count')
    #df = df[df[column+'_count']<50]

    idx_to_keep = ~np.isnan(df[column].values)
    lat_values = df[lat_col][idx_to_keep]
    lon_values = df[lon_col][idx_to_keep]
    values = (df[column].values)[idx_to_keep]

    fig, ax = plt.subplots(figsize=(20,10), facecolor=(1, 1, 1))
    if in_conus:
        conus.boundary.plot(ax=ax, color='k')
    if with_states_boundaries:
        states.boundary.plot(ax=ax, color='k')
    if filenames is not None:
        for filename_,c in filenames:
            gpd.read_file(filename_).boundary.plot(ax=ax, color=c, lw=2., zorder=6)
    #gpd.read_file('datasets/NHD_H_14020002_HU8_Shape/Shape/NHDFlowline.shp').plot(ax=ax, color='b', lw=.1, zorder=6)
    if discrete:
        d_color = {j:i for i,j in enumerate(sorted(df[column].unique()))}
        #print(d_color)
        #colorlist=np.array([COLORS[d_color[i]%len(COLORS)] for i in df[column].values])
        #print(colorlist)
        colorlist=np.array([COLORS[i%len(COLORS)] for i in df[column].values])
        for i in sorted(df[column].unique()):
            ax.scatter(lon_values[df[column]==i], lat_values[df[column]==i], s=30, marker=markers[(i//len(COLORS))%len(markers)], color=colorlist[df[column]==i], zorder=7, alpha=.8, lw=0)
        if highlighted_sites is not None: 
            ax.scatter(highlighted_sites["lon"], highlighted_sites["lat"], s=100*np.array(highlighted_sites.get("size",1)), color=[COLORS[i%len(COLORS)] for i in highlighted_sites["community"]], zorder=9)
            ax.scatter(highlighted_sites["lon"], highlighted_sites["lat"], s=200*np.array(highlighted_sites.get("size",1)), c=highlighted_sites["color"], zorder=8)
            ax.text(.05, -.05, "\n".join(highlighted_sites["name"][:20]), transform=ax.transAxes, fontsize=14, verticalalignment='top')


    else:
        if values_for_cmap is not None:

            if reverse_map:
                colors_below_center = plt.cm.coolwarm_r(np.linspace(0, 0.5, 256))
                colors_above_center = plt.cm.coolwarm_r(np.linspace(0.5, 1, 256))
            else:
                colors_below_center = plt.cm.coolwarm(np.linspace(0, 0.5, 256))
                colors_above_center = plt.cm.coolwarm(np.linspace(0.5, 1, 256))
            all_colors = np.vstack((colors_below_center, colors_above_center))
            terrain_map = mcolors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

            # make the norm:  Note the center is offset so that the land has more dynamic range:
            divnorm = mcolors.TwoSlopeNorm(vmin=values_for_cmap[0], vcenter=values_for_cmap[1], vmax=values_for_cmap[2])

            sc = ax.scatter(lon_values, lat_values, c=values, cmap=terrain_map, norm=divnorm, zorder=7)
            cbar = fig.colorbar(sc, shrink=0.5)
            cbar.ax.tick_params(labelsize=24)
        else:
            sc = ax.scatter(lon_values, lat_values, c=values, cmap=plt.cm.coolwarm, zorder=7)
            cbar = fig.colorbar(sc, shrink=0.5)
            cbar.ax.tick_params(labelsize=24)

    ax.axis("off")
    plt.title(title, fontsize=24)
    plt.tight_layout()
    plt.savefig('map_dots_%s' % suffix, dpi=600)
    #plt.close()


def plot_WQ(df_WQ):
    quantities = [x for x in df_WQ.columns.values if x not in ["STAID", "Row_ID", "community", "community_size"]]

    #communities_to_plot = [i for i,j in count_dict.items() if (j>=min_community_size)]
    #df_WQ_filtered = df_WQ[df_WQ["community"].isin(communities_to_plot)]

    for quantity in quantities:
        #X = df_WQ_filtered.groupby('community')[quantity].agg(list).to_dict()
        X = df_WQ.groupby('community')[quantity].agg(list).to_dict()
        sorted_communities = sorted(list(X.keys()))
        #sorted_communities = sorted(communities_to_plot)
        X = [np.array(X[i]) for i in sorted_communities]
        X = [x[~np.isnan(x)] for x in X]
        fig, ax = plt.subplots(figsize=(26,12))
        plt.axhline(y=np.median([x for xs in X for x in xs]), color='k', linestyle='--')
        bplot = plt.boxplot(X, patch_artist=True, showfliers=False, labels =sorted_communities)
        #bplot = plt.violinplot(X)
        for i, boxes in enumerate(bplot['boxes']):
            boxes.set_facecolor(COLORS[sorted_communities[i]%len(COLORS)])
        for i, medians in enumerate(bplot['medians']):
            medians.set_color('k')
        #for i, fliers in enumerate(bplot['fliers']):
        #    fliers.set_mec(COLORS[i%len(COLORS)])
        #ax.set_title(quantity, fontsize=20)
        plt.xlabel('cluster', fontsize=32)
        plt.ylabel(quantity.split('.')[-1].upper(), fontsize=32)
        ax.set_xticklabels(sorted_communities, fontdict={'fontsize': 32}, rotation=0)
        #ax.set_xticklabels(np.arange(len(X)), fontdict={'fontsize': 32})
        ax.tick_params(axis='x', labelsize=32)
        ax.tick_params(axis='y', labelsize=32)
        plt.savefig('boxplot_%s' % quantity.replace('.', ''), facecolor='w', dpi=300)
        #plt.close()



def corrcoef(matrix):
    r = np.corrcoef(matrix)
    rf = r[np.triu_indices(r.shape[0], 1)]
    df = matrix.shape[1] - 2
    ts = rf * rf * (df / (1 - rf * rf))
    pf = betainc(0.5 * df, 0.5, df / (df + ts))
    p = np.zeros(shape=r.shape)
    p[np.triu_indices(p.shape[0], 1)] = pf
    p[np.tril_indices(p.shape[0], -1)] = p.T[np.tril_indices(p.shape[0], -1)]
    p[np.diag_indices(p.shape[0])] = np.ones(p.shape[0])
    return r, p


def make_spatial_analysis(gdf):
    #gdf.set_crs(epsg=4269, inplace=True)
    gdf_top = gdf[gdf["community"]<=33]
    staid_to_keep = []
    #all_points = MultiPoint(gdf["geometry"])
    X = np.empty((len(gdf), 2))
    X[:,0] = gdf.LNG_GAGE.values
    X[:,1] = gdf.LAT_GAGE.values

    all_distances = distance.cdist(X, X, 'euclidean')
    all_communities = gdf["community"].values
    nearest = all_distances.argsort(axis=1)

    homog_random_distr = np.sum(np.array(list(Counter(all_communities).values()))**2)/(len(all_communities)**2)

    dict_nearest = {}
    all_nearest_cumul = 0
    for i in range(len(gdf)):
        dict_nearest.setdefault(all_communities[nearest[i,0]], []).append(all_communities[nearest[i,1]])
        if all_communities[nearest[i,0]] == all_communities[nearest[i,1]]:
            all_nearest_cumul +=1
    real_homog = all_nearest_cumul/len(gdf)

    for community in sorted(gdf_top["community"].unique())[:]:
        gdf_top_in_community = gdf_top.iloc[(gdf_top["community"]==community).values, :]
        centroid = gdf_top_in_community.dissolve().centroid[0]
        distances = gdf_top_in_community["geometry"].distance(centroid)
        perc95 = np.percentile(distances, 95)
        staid_to_keep.extend(gdf_top_in_community["STAID"][distances<perc95].values.tolist())
    gdf_top_no_outliers = gdf_top[gdf_top["STAID"].isin(staid_to_keep)]

    gdf_top_clusters = pd.DataFrame(gdf_top_no_outliers.groupby("community")["geometry"].agg(list))

    gdf_top_clusters["geometry"] = [MultiPoint(cluster).convex_hull for cluster in gdf_top_clusters["geometry"]]
    #gdf_top_clusters["geometry"] = [cluster for cluster in gdf_top_clusters["geometry"]]
    #print(gdf_top_clusters)
    gdf_top_clusters.reset_index(inplace=True)
    gdf_top_clusters=gpd.GeoDataFrame(gdf_top_clusters)

    return gdf_top_clusters, homog_random_distr, real_homog

def plot_homogeneity_by_cluster(gdf, gdf_top_clusters):

    a = []
    b = []
    for community in sorted(gdf_top_clusters["community"].values)[:]:
        gdf_in_community = gpd.sjoin(left_df=gdf[gdf["community"]<=100], right_df=gdf_top_clusters[gdf_top_clusters["community"]==community],  how="inner", predicate="intersects")
        prob = gdf_in_community["community_left"].value_counts()#/(gdf_top["community"].value_counts())
        prob = prob[~np.isnan(prob)]
        prob_norm = prob.values/np.sum(prob)
        spatial_measure = prob[community]/np.sum(prob)#/(len(gdf[gdf["community"]==community])/9067.)
        b.append(len(gdf[gdf["community"]==community])/len(gdf))
        a.append(spatial_measure)
    b = np.array(b)[np.argsort(a)]

    name_clusters = {
        0: "Forested Basin-Ranges",
        1: "Urban Areas",
        2: "Croplands",
        3: "Shrublands Basin-Ranges",
        4: "High Temp Wetlands",
        5: "Mixed Forests",
        6: "High Precip Forests",
        7: "Very High Precip",
        8: "Shrublands",
        9: "Grasslands",
        10: "High Temp Wetlands+Forest",
        11: "Herbaceous Wetlands",
        12: "Low Summer Prec Mixed Forest",
        13: "Croplands",
        14: "Herbaceous Wetlands",
        15: "Low Prec Pastures",
        16: "Shrublands",
        17: "High Prec Decid Forests",
        18: "High Temps Major Dams",
        19: "High Temp Pastures+Grasslands",
        20: "Croplands+Mixed Forests",
        21: "High Elev Decid Forests",
        22: "High Summer Prec Lakes and Reservoirs",
        23: "High Summer Prec Decid Forests",
        24: "Croplands",
        25: "High Temps Low Elev",
        26: "Croplands+Wetlands",
        27: "Summer Prec Decid Forests",
        28: "High Elev Lakes and Reservoirs",
        29: "Croplands",
        30: "High Temp Major Dams",
        31: "Mixed Forests",
        32: "Low Summer Prec Urba Areas",
        33: "High Temps Low Elev",
    }

    color_clusters = {
        0: "green",
        1: "red",
        2: "orange",
        3: "green",
        4: "green",
        5: "green",
        6: "green",
        7: "green",
        8: "green",
        9: "green",
        10: "green",
        11: "green",
        12: "green",
        13: "orange",
        14: "green",
        15: "orange",
        16: "green",
        17: "green",
        18: "red",
        19: "orange",
        20: "orange",
        21: "green",
        22: "red",
        23: "green",
        24: "orange",
        25: "green",
        26: "orange",
        27: "green",
        28: "red",
        29: "orange",
        30: "red",
        31: "green",
        32: "red",
        33: "green",
    }


    name_clusters_inv = {j:i for i,j in name_clusters.items()}
    a = (pd.Series(a)).sort_values(ascending=True)
    a = pd.DataFrame({"cluster_id":a.index.values,"cluster":[name_clusters[x] for x in a.index.values], "spatial measure":a.values})
    fig, ax = plt.subplots(figsize=(15,10), facecolor="white")
    ax.bar(list(range(len(a))), a["spatial measure"].values, color=[color_clusters[name_clusters_inv[x]] for x in a["cluster"]])
    ax.bar(list(range(len(a))), b, edgecolor='black', color='none', lw=2)
    ax.set_xticks(list(range(len(a))))
    ax.set_xticklabels(a["cluster_id"].values.tolist(), rotation=90)
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_xlabel('cluster', fontsize=26)
    ax.set_ylabel('cluster homogeneity measure', fontsize=26)

    plt.savefig("spatial_heterogeneity.png", dpi=300)
    #plt.close()


def plot_exemplar_hulls(gdf, gdf_top_clusters, communities_to_plot=[7,18,26]):
    d_color = {j:i for i,j in enumerate(sorted(gdf["community"].unique()))}
    colorlist=[COLORS[i%len(COLORS)] for i in gdf["community"].values]
    #print(colorlist)

    conus, conus_states = load_map()
    fig, ax = plt.subplots(figsize=(15,10), facecolor="white")
    for c in communities_to_plot:
        gdf_top_clusters[gdf_top_clusters["community"]==c].boundary.plot(ax=ax, color=COLORS[c%20], lw=3)
        inset = gpd.sjoin(left_df=gdf[gdf["community"]<=100], right_df=gdf_top_clusters[gdf_top_clusters["community"]==c],  how="inner", predicate="intersects")
        inset.plot(ax=ax, color=np.array(colorlist)[gdf["STAID"].isin(inset.STAID.values)], markersize=10)

    conus.boundary.plot(ax=ax, color="black")
    conus_states.boundary.plot(ax=ax, color="black")
    ax.axis("off")
    plt.savefig("catchments_hulls", dpi=300)
    #plt.close()
