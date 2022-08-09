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
from collections import Counter
from scipy.spatial import Voronoi, voronoi_plot_2d
from voronoi import wrapper as get_polygons
from shapely.geometry import Polygon
from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
import glob


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


def moving_average(x, window = 3, valid=True, error=None):
    if valid:
        if error is None:
            return np.array([np.nanmean(x[i:i+window]) for i in range(len(x)-window+1)])
        else:
            x_avg = np.array([np.nanmean(x[i:i+window]) for i in range(len(x)-window+1)])
            x_err_avg = np.array([(np.sqrt(np.nansum((error[i:i+window])**2))/np.sum(~np.isnan(error[i:i+window]))) for i in range(len(error)-window+1)])
            return x_avg, x_err_avg

    x_average = []
    step = int(window/2)
    for i in range(len(x)):
        start = np.maximum(0, i-step)
        end = np.minimum(len(x), i+step)
        y = x[start:end+1]
        x_average.append(np.nanmean(y))

    return np.array(x_average)


def standardize_df(df):

    df[df.columns] = standardize_M(df.values)

    return df

def standardize_M(M):

    return StandardScaler().fit_transform(M)


def make_cosine_similarity(M, between_0_1=False):
    
    M = normalize(M, axis=1, norm='l2')

    if between_0_1:
        S = (M.dot(M.T)+1.)/2
    else:
        S = M.dot(M.T)
    
    np.fill_diagonal(S, 0.)

    return S

def make_euclidean_similarity(A, do_norm=False):
    if do_norm:
        A = normalize(A, axis=1, norm='l2')

    # make dot product
    A_dot_A = np.dot(A,A.T)
    
    #Use i,i elements as l2 norm for diagonal
    # and fill matrix with all pairwise combinations
    A_tile = np.tile(np.diagonal(A_dot_A), (A.shape[0],1))

    S = np.sqrt((A_tile + A_tile.T - (2*A_dot_A)))
    print(np.sum(S==0.))
    if do_norm:
        S = (2.-S)/2.
    else:
        S[S!=0.] = 1./S[S!=0.]
    #print(S.shape)
    #print(len(S==0))
    #print(np.max(S))
    if not do_norm:
        S = np.log(1.+S)                
    S/=np.max(S)
    #print(np.min(S[np.triu_indices(S.shape[0], k=1)]), np.median(S[np.triu_indices(S.shape[0], k=1)]), np.max(S[np.triu_indices(S.shape[0], k=1)]))
    np.fill_diagonal(S, 0.)
    return S

def shuffle_along_axis(M, axis = 0):
    if axis == 0:
        M = M.T
    M_shuffled = []
    for v in M:
        np.random.shuffle(v)
        M_shuffled.append(v)

    if axis == 0:
        return np.array(M_shuffled).T
    else:
        return np.array(M_shuffled)




def make_randomized_similarity(M, sim, augmentation=1, suffix=''):

    M_shuffled = np.tile(M,(augmentation,1))
    shuffle_along_axis(M_shuffled)

    random_sim = make_cosine_similarity(M_shuffled)

    iut = np.triu_indices(sim.shape[0], k=1)
    sim_vals = sim[iut]
    random_sim_vals = random_sim[iut]

    bins = np.linspace(np.min(sim_vals), np.max(sim_vals), 31)
    sim_vals_hist, bins = np.histogram(sim_vals, bins=bins, density=True)
    random_sim_vals_hist, bins = np.histogram(random_sim_vals, bins=bins, density=True)

    fig, ax = plt.subplots()
    ax.plot((bins[1:]+bins[:-1])/2., sim_vals_hist, label="similarity")
    ax.plot((bins[1:]+bins[:-1])/2., random_sim_vals_hist, label="random")
    #ax.plot((bins[1:]+bins[:-1])/2., kl)
    #plt.axvline(x=t, color='red', linestyle='--')
    ax.legend(loc=0)
    #ax.set_xscale('log')
    plt.savefig('sim_vals_distribution_%s' % suffix)
    plt.close()


def find_max_curvature(x,y):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    dx = (np.max(x)-np.min(x))
    dy = (np.max(y)-np.min(y))
    xmin = np.min(x)
    ymin = np.min(y)
          
    x = (x-xmin)/dx
    y = (y-ymin)/dy

    f = interp1d(x,y, kind='linear')
    x_fine = np.linspace(np.min(x), np.max(x), 10*len(x))
    y_fine = moving_average(f(x_fine), window=5)

    x_t = np.gradient(x_fine)
    y_t = np.gradient(y_fine)

    vel = np.array([ [x_t[i], y_t[i]] for i in range(x_t.size)])

    #print(vel)
    speed = np.sqrt(x_t * x_t + y_t * y_t)
    tangent = np.array([1/speed] * 2).transpose() * vel

    ss_t = np.gradient(speed)
    xx_t = np.gradient(x_t)
    yy_t = np.gradient(y_t)

    curvature_val = np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t)**1.5
    
    max_curv_idx = np.argmax(curvature_val)

    x_max_curv = xmin+dx*x_fine[max_curv_idx]
    y_max_curv = ymin+dy*y_fine[max_curv_idx]

    ax.scatter(x_max_curv, y_max_curv)
    return x_max_curv

def find_optimal_rank_from_cum_var(S, target_cum_var=0.65, max_k_to_plot=None, reconstruction_error=None, suffix=''):
    
    S2 = S**2/np.sum(S**2)
    D_cumsum = np.cumsum(S2)
    D_cumsum = D_cumsum/D_cumsum[-1]

    #k = int(np.round(find_max_curvature(np.arange(len(S2))+1, S2)))

    # relative_difference
    #rel_diff = (S2[:-1]-S2[1:])/S2[:-1]

    xhat=np.arange(len(S2))+1
    #yhat=moving_average(rel_diff, window=7)

    #k_idx= np.arange(len(yhat))[yhat<0.05][0]
    #k_idx = np.argmin(yhat)
    #k = xhat[k_idx]
    k_idx = np.argmin(np.abs(D_cumsum-target_cum_var))
    k = xhat[k_idx]

    if reconstruction_error !=None:
        fig, (ax1, ax2, ax3) = plt.subplots(3)
    else:
        fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot(np.arange(len(D_cumsum)+1), [0,]+D_cumsum.tolist())

    ax1.axvline(x=k, color='red', linestyle='--', label=str(k))

    ax1.set_ylabel(r"cum($\sigma^2$)")
    ax1.set_xlabel('k')
    ax1.set_ylim([0,1])
    if max_k_to_plot!=None:
        ax1.set_xlim([0, max_k_to_plot])
    ax1.legend(loc=0)

    ax2.plot(np.arange(len(S))+1, S2)

    ax2.axvline(x=k, color='red', linestyle='--', label=str(k))

    ax2.set_ylabel(r"$\sigma^2$")
    ax2.set_xlabel('k')
    if max_k_to_plot!=None:
        ax2.set_xlim([0, max_k_to_plot])
    ax2.legend(loc=0)

    if reconstruction_error !=None:
        ax3.plot(np.arange(len(S))+1, reconstruction_error)
        ax3.set_ylabel("reconstruction error")
        ax3.set_xlabel('k')
        if max_k_to_plot!=None:
            ax3.set_xlim([0, max_k_to_plot])


    plt.savefig('diag_S%s' % suffix)
    plt.close()

    return k



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

def make_network(sim, labels=[], threshold = 0.95, with_weights=True):
    """
    Generate a networkx graph based on cosine similarity values.
    It uses an absolute weight threshold to filter out edges.

    PARAMS:
        sim (np.array): square array containing pairwise cosine similarities
            between pairs of nodes
        labels (np.array): list of strings containing the name of the node
    
    OPTIONAL:
        threshold (float): weight threshold below which edges are discarded.
            Range: [-1, 1]. Default: 0.95
    
    RETURNS:
        G (networkx graph): undirected networkx format graph
    """

    # Generate the undirected graph
    G = nx.Graph()

    # Create nodes and assign labels
    if len(labels)==0:
        labels = range(sim.shape[0])
    #print(labels)
    for i,label in enumerate(labels):
        G.add_node(i, label=label)


    # Select only upper triangle matrix indices
    # because of cosine similarity simmetry
    iut = np.triu_indices(sim.shape[0], k=1)

    # Create edges above the threshold
    if with_weights:
        for i in range(len(iut[0])):
            w = sim[iut[0][i],iut[1][i]]
            if w>threshold:
                G.add_edge(iut[0][i],iut[1][i],weight=w)

    else:
        for i in range(len(iut[0])):
            w = sim[iut[0][i],iut[1][i]]
            if w>threshold:
                G.add_edge(iut[0][i],iut[1][i],weight=1.)

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
        modularity_best_partition = -1


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

def make_weight_distr(M, suffix=''):
    w = [x for x in np.triu(M, k=1).reshape(-1) if x!=0]
    h, bins = np.histogram(w, bins=np.logspace(np.log10(np.min(w)), 0, 51), density=True)
    fig,ax = plt.subplots()
    ax.plot((bins[1:]+bins[:-1])/2., h)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig("weight_distr_%s" % suffix)
    plt.close()

def make_backbone_wrapper(M, alphas, target=.95 ,reciprocated=False, suffix=''):

    np.fill_diagonal(M, 0.)
    n = M.shape[0]
    k = np.tile(np.sum(M.astype(bool), axis=1).reshape(n, 1), (1,n))
    weight_complete = np.sum(M)/2.
    edges_complete = np.sum(M.astype(bool))/2.
    M_norm = normalize(M, axis=1, norm='l1')
    make_weight_distr(M_norm, suffix=suffix)

    Gs = []
    weights = []
    edges = []
    M_alphas = []
    ratio_nodes_in_giant = []
    num_nodes_in_backbone = []
    for alpha in alphas:

        M_alpha = copy(M)

        idx_to_zero = (1-M_norm)**(k-1)>=alpha
        #idx_to_zero = M_norm<alpha
        if reciprocated:
            idx_to_zero = idx_to_zero | idx_to_zero.T
        else:
            idx_to_zero = idx_to_zero & idx_to_zero.T
        M_alpha[idx_to_zero]=0.

        
        #Gs.append(nx.from_edgelist(nx.from_numpy_array(M_alpha).edges))
        Gs.append(nx.from_numpy_array(M_alpha))
        weights.append((np.sum(M_alpha)/2.)/weight_complete)
        edges.append(len(Gs[-1].edges)/edges_complete)
        #ratio_nodes_in_giant.append(1.*len(Gs[-1].nodes)/n)
        ratio_nodes_in_giant.append(float(len(sorted(nx.connected_components(Gs[-1]), key=len, reverse=True)[0]))/n)
        print(alpha, 1.-alpha**(1./(n-2.)), weights[-1], edges[-1], ratio_nodes_in_giant[-1])
        if ratio_nodes_in_giant[-1]>=1.: break

    idx_target = np.argmin(np.abs(np.array(ratio_nodes_in_giant)-target))

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(weights, ratio_nodes_in_giant, '-', label = "weight")
    ax2.plot(edges, ratio_nodes_in_giant, '-', label = "edges")
    ax1.scatter(weights[idx_target], ratio_nodes_in_giant[idx_target], c='r')
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    for i in range(len(weights)):
        if i==idx_target:
            ax1.text(weights[i], ratio_nodes_in_giant[i], '%.3f'%alphas[i], fontsize=20, color='red', zorder=6)
        else:
            ax1.text(weights[i], ratio_nodes_in_giant[i], '%.3f'%alphas[i])

    if reciprocated:
        plt.savefig("backbone_reciprocated_%s" % suffix)
    else:
        plt.savefig("backbone_%s" % suffix)
    plt.close()
 
    return alphas[idx_target]

def make_percolation_wrapper(M, alphas, target=.95, suffix=''):

    np.fill_diagonal(M, 0.)
    
    n = M.shape[0]
    #k = np.tile(np.sum(M.astype(bool), axis=1).reshape(n, 1), (1,n))
    weight_complete = np.sum(M)/2.
    edges_complete = np.sum(M.astype(bool))/2.
    #M_norm = normalize(M, axis=1, norm='l1')


    Gs = []
    weights = []
    edges = []
    M_alphas = []
    ratio_nodes_in_giant = []
    num_nodes_in_backbone = []
    for alpha in alphas:

        M_alpha = copy(M)

        idx_to_zero = M<alpha
        #idx_to_zero = M_norm<alpha
        #idx_to_zero = idx_to_zero & idx_to_zero.T
        M_alpha[idx_to_zero]=0.

        
        #Gs.append(nx.from_edgelist(nx.from_numpy_array(M_alpha).edges))
        Gs.append(nx.from_numpy_array(M_alpha))
        weights.append((np.sum(M_alpha)/2.)/weight_complete)
        edges.append(len(Gs[-1].edges)/edges_complete)
        #ratio_nodes_in_giant.append(1.*len(Gs[-1].nodes)/n)
        ratio_nodes_in_giant.append(float(len(sorted(nx.connected_components(Gs[-1]), key=len, reverse=True)[0]))/n)
        #print(alpha, 1.-alpha**(1./(n-2.)), weights[-1], edges[-1], ratio_nodes_in_giant[-1])
        print(alpha, weights[-1], edges[-1], ratio_nodes_in_giant[-1])
        if ratio_nodes_in_giant[-1]>=.99: break

    idx_target = np.argmin(np.abs(np.array(ratio_nodes_in_giant)-target))

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(weights, ratio_nodes_in_giant, '-', label = "weight")
    ax2.plot(edges, ratio_nodes_in_giant, '-', label = "edges")
    ax1.scatter(weights[idx_target], ratio_nodes_in_giant[idx_target], c='r')
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    for i in range(len(weights)):
        if i==idx_target:
            ax1.text(weights[i], ratio_nodes_in_giant[i], '%.3f'%alphas[i], fontsize=20, color='red', zorder=6)
        else:
            ax1.text(weights[i], ratio_nodes_in_giant[i], '%.3f'%alphas[i])

    plt.savefig("percolation_%s" % suffix)
    plt.close()
 
    return alphas[idx_target]


def make_joined_entropy(A, B):
    C = (A.dot(B.T)).astype(float)/np.sum(A)
    entropy = np.nan_to_num(-C*np.log(C))
    return np.sum(entropy)


def make_community_entropy(A):

    C = (np.sum(A, axis=1).astype(float))/np.sum(A)

    return -np.sum((C)*np.log(C))

def find_elbow(x, y):
    f = interp1d(x,y, kind='linear')

    xhat = np.linspace(min(x), max(x), 9*len(x))
    step = xhat[1]-xhat[0]
    yhat=moving_average(f(xhat), window=9)

    dev1 = yhat[1:]-yhat[:-1]
    dev1 = (dev1[1:]+dev1[:-1])/2.
    dev1 = (1/step)*dev1
    dev2 = yhat[:-2]+yhat[2:]-(2*yhat[1:-1])
    dev2 = ((1/step)**2)*dev2
    curv = np.abs(dev2)/np.power(1+(dev1**2), 3./2.)



    max_curv_idx = np.argmax(curv)+1
    max_curv_x = xhat[max_curv_idx]

    #return np.argmin(np.abs(x-max_curv_x)), xhat, yhat
    return max_curv_x, xhat, yhat

def make_percolation(Ms,suffix='',ws=None):
    if ws == None:
        ws = np.linspace(0,1.,401)[::-1]

    fig, axarr = plt.subplots(1,1,figsize=(16,12))
    for M,k in Ms:
        np.fill_diagonal(M, 0.)
        num_nodes = M.shape[0]
        weight_complete = np.sum(M)/2.
        X=[]; Y=[]; Y_nmi=[]
        for w in ws:
            M_ = copy(M)
            M_[M<w]=0
            weight = np.sum(M_)/2.
            G = nx.from_numpy_array(M_)
            num_nodes_in_giant = len(sorted(nx.connected_components(G), key=len, reverse=True)[0])
            '''
            modularity, communities = make_community(G, do_sort=False)
            community_adj = make_community_adjacency_matrix(communities)
            entropy = make_community_entropy(community_adj)
            if (w != ws[0]) and (community_adj.shape[0]<(num_nodes/10.)):
                joined_entropy = make_joined_entropy(community_adj, community_adj_old)
                nmi = 2.*(entropy + entropy_old-joined_entropy )/(entropy + entropy_old)
                Y_nmi.append(nmi)

            entropy_old = copy(entropy)
            community_adj_old = copy(community_adj)
            '''
            X.append(weight/weight_complete)
            Y.append(num_nodes_in_giant/num_nodes)


            print(k, w, X[-1], Y[-1])
            if Y[-1]>0.995: break
        axarr.plot(X, Y, '-o', label=str(k))
        #axarr[1].plot(X[-len(Y_nmi):], Y_nmi, '-o', label=str(k))

    plt.savefig("percolation_%s" % suffix)
    return 0

def make_modularity(Ms, max_weights = np.linspace(0,1,21), suffix=''):

    fig, ax = plt.subplots(figsize=(16,12))
    for M,k in Ms:
        X=[];Y=[]
        for max_w in max_weights:
            G = make_network(M, threshold = max_w)
            num_nodes = len(G.nodes())
            num_nodes_components = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
            num_nodes_largest_component = num_nodes_components[0]
            num_orphans = np.sum([x for x in num_nodes_components if x<3])
            try:
                modularity_ = community_louvain.modularity(community_louvain.best_partition(G), G)
            except:
                break
            X.append(num_orphans/num_nodes)
            Y.append(modularity_)
        ax.plot(X, Y, '-', label=str(k))
        for i in range(len(X)):
            ax.text(X[i], Y[i], '%.2f'%max_weights[i])
        ax.legend(loc=0)
    plt.savefig("modularity_%s" % suffix)


def plot_gauges(df, column = 'huc_02'):
    conus = load_map()

    fig, ax = plt.subplots(figsize=(16,12))
    conus.boundary.plot(ax=ax, color='k')
    d = {j:i for i,j in enumerate(df[column].unique())}
    colorlist=[COLORS[d[i]%len(COLORS)] for i in df[column].values]
    ax.scatter(df['gauge_lon'], df['gauge_lat'], color=colorlist, zorder=7)
    ax.axis("off")
    plt.savefig('map')
    plt.close()



def make_df_community_anomalies(df):
    df_mean = df.mean(axis=0)
    df_std = df.std(axis=0)

    df_community = df.groupby('community').agg(['mean'])

    for col in [x for x in df_mean.index.values if x != 'community']:
        df_community[(col, 'ttest')] = (df_community[(col, 'mean')] - df_mean[col])/df_std[col]
        df_community.drop(columns=[(col, 'mean')], inplace=True)

    #df_community = df_community[df_community.index.isin([i for i,j in community_counts_dict.items() if j>=min_community_size])]
    
    return df_community

def plot_anomalies(df, count_dict,community_to_states={},min_community_size=3, suffix=''):

    communities_to_plot = [i for i,j in count_dict.items() if j>=min_community_size]
    num_communities = len(communities_to_plot)
    df_columns_names = [x[0] for x in df.columns.values]

    to_write_in_report = []
    fig, axarr = plt.subplots(num_communities, 1, figsize=(20,30), sharex=True, constrained_layout=True)
    for i, community in enumerate(communities_to_plot):
        color= COLORS[community%len(COLORS)]
        print("community: %d\tnum gauges: %d\tcolor: %s" % (community, count_dict[community], color))
        to_write_in_report.append("community: %d\tnum gauges: %d\tcolor: %s" % (community, count_dict[community], color))
        if community_to_states:
            print('\t'.join(["%s:%d" %(j[0],j[1]) for j in community_to_states[community]]))
            to_write_in_report.append('\t'.join(["%s:%d" %(j[0],j[1]) for j in community_to_states[community]]))
        t_test = df.loc[community, :].values
        t_test_abs_rank = np.argsort(np.abs(t_test))[::-1]
        print('\t'+'\n\t'.join(["%s: %.2f" % (df_columns_names[j], t_test[j]) for j in t_test_abs_rank[:20]]))
        to_write_in_report.append('\t'+'\n\t'.join(["%s: %.2f" % (df_columns_names[j], t_test[j]) for j in t_test_abs_rank[:20]]))
        axarr[i].bar(np.arange(len(t_test)), t_test, tick_label=df_columns_names, color=color, label='%d'%(community))
        axarr[i].legend(loc=0)
    #plt.setp(axarr, ylim=(-2,3))
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees

    #plt.tight_layout()
    #plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('anomalies_%s' % suffix)
    plt.close()
    with open('report_anomalies.txt', 'w', encoding='utf-8') as f:
        f.writelines("\n".join(to_write_in_report))
    f.close()



def plot_variance(M, labels, suffix=''):
    fig, ax = plt.subplots(figsize=(16,12))
    var = np.var(M, axis=0)
    plt.bar(np.arange(len(var)), var)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    plt.savefig('variance_%s' % suffix)
    plt.close()

def plot_heatmap(M, labels, communities, suffix=''):
    #print(labels)
    communities_inv = {}
    for i,j in communities.items():
        communities_inv.setdefault(j, []).append(i)
    #print({i: [labels[k] for k in j] for i,j in communities_inv.items()})
    communities_size = {i:len(j) for i,j in communities_inv.items()}
    M_sorted = []
    for community, size in sorted(communities_size.items(), key=lambda item: item[1], reverse=True):
    #    print(community, size, communities_inv[community], [labels[k] for k in communities_inv[community]])
        for node in communities_inv[community]:
            M_sorted.append(M[int(node), :])
    fig, ax = plt.subplots(figsize=(16,12))
    img = ax.imshow(np.array(M_sorted))
    plt.colorbar(img)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    plt.savefig('heatmap_%s' % suffix)
    plt.close()

def plot_heatmap_simple(M, labels=[], suffix=''):
    fig, ax = plt.subplots(figsize=(16,12))
    img = ax.imshow(np.array(M))#, cmap=plt.cm.cool)
    plt.colorbar(img)
    if len(labels)!=0:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    plt.xticks(rotation = 90)
    plt.savefig('heatmap_%s' % suffix)
    plt.close()

def plot_network(G, quantities, nodesize=None, suffix='', do_plot_labels=True, bicolor=False):


    nodelist = G.nodes()
    edgelist = G.edges()

    if len(nodelist)>1000:
        layout_engine="sfdp"
    else:
        layout_engine="fdp"

    if nodesize==None:
        nodesize = 300
    else:
        nodesize = [300*nodesize[i] for i in nodelist]

    pos = nx.nx_agraph.graphviz_layout(G, prog=layout_engine)

    labels = nx.get_node_attributes(G, "label")
    for quantity in quantities:
        quantity_name = quantity["name"]
        quantity_values = quantity["values"]

        
        fig, ax = plt.subplots(figsize=(30,30))


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
                edgecolors='k')
        else:
            vmin = np.min(list(quantity_values.values()))
            vmax = np.max(list(quantity_values.values()))

            drawn_nodes = nx.draw_networkx_nodes(G, pos,
                nodelist=nodelist,
                node_size=nodesize,
                node_color=[quantity_values[i] for i in nodelist],
                edgecolors='k',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax)

        nx.draw_networkx_edges(G, pos,
            edgelist=edgelist)
        
        if do_plot_labels:
            nx.draw_networkx_labels(G, pos, labels=labels)

        if quantity["discrete"]==False:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, shrink=0.5)#, fraction=0.047, pad=0.04)
            cbar.ax.tick_params(labelsize=20)

        plt.axis('off')

        ax.set_title(quantity_name, fontdict={'fontsize': 32})
        plt.savefig('network_%s_%s' % (suffix, quantity_name[:50]))
        plt.close()

def plot_gauges_in_map_dots(df, lat_col, lon_col, column = 'HUC02',suffix=''):
    conus, states = load_map()

    fig, ax = plt.subplots(figsize=(32,24))
    conus.boundary.plot(ax=ax, color='k')
    states.boundary.plot(ax=ax, color='k')
    #d_color = {j:i for i,j in enumerate(sorted(df[column].unique()))}
    #colorlist=[COLORS[d_color[i]%len(COLORS)] for i in df[column].values]
    colorlist=[COLORS[i%len(COLORS)] for i in df[column].values]
    ax.scatter(df[lon_col], df[lat_col], color=colorlist, zorder=7)
    ax.axis("off")
    plt.savefig('map_dots_%s' % suffix)
    plt.close()

def plot_gauges_in_map_voronoi(df, lat_col, lon_col, column = 'HUC02' , suffix=''):

    conus, states = load_map()
    conus_total_bounds = [-124.762578,24.52105162,-66.949831,49.3884167]#conus.total_bounds
    fig, ax = plt.subplots(figsize=(32,24))
    conus.boundary.plot(ax=ax, color='k', zorder=6)
    states.boundary.plot(ax=ax, color='k', zorder=6)

    conus_polygon = conus.geometry
    print(type(conus_polygon))
    
    #xs, ys = conus.geometry.values[0].interiors.xy    
    #ax.fill(xs, ys, alpha=0.5, fc='white', ec='none', zorder=5)
    #colorlist=[COLORS[d[i]%len(COLORS)] for i in df[column].values]
    points = np.vstack((df[lon_col].values, df[lat_col].values)).T
    #d_color = {j:i for i,j in enumerate(sorted(df[column].unique()))}
    colorlist=[COLORS[i%len(COLORS)] for i in df[column].values]
    polygons,bbox = get_polygons(points)
    for i in range(len(polygons)):
        polygon = conus_polygon.intersection(Polygon([polygons[i][j].tolist() for j in range(len(polygons[i]))]))
        #ax.fill(*zip(*polygons[i]), color=colorlist[i], zorder=5)
        polygon.plot(ax=ax, color=colorlist[i], zorder=5)

    #ax.scatter(points[:,0], points[:,1], color=colorlist, zorder=7)
    #plt.xlim([conus_total_bounds[0],conus_total_bounds[2]]), plt.ylim([conus_total_bounds[1],conus_total_bounds[3]])
    #plt.xlim([bbox[0][0],bbox[1][0]]), plt.ylim([bbox[0][1],bbox[2][1]])
    
    ax.axis("off")
    plt.savefig('map_voronoi_%s' % suffix)
    plt.close()

def make_random_forest(X, y):
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X, y)

    return forest

def get_features_importance_from_forest(X, y):

    forest = make_random_forest(X, y)

    feature_importances = forest.feature_importances_

    return feature_importances

def plot_WQ(df_WQ):
    quantities = [x for x in df_WQ.columns.values if x not in ["STAID", "community", "community_size"]]

    df_WQ_filtered = df_WQ[df_WQ["community_size"]>=50]

    for quantity in quantities:
        X = df_WQ_filtered.groupby('community')[quantity].agg(list).to_dict()
        sorted_communities = sorted(list(X.keys()))
        X = [np.array(X[i]) for i in sorted_communities]
        X = [x[~np.isnan(x)] for x in X]
        fig, ax = plt.subplots(figsize=(16,12))
        plt.axhline(y=np.median([x for xs in X for x in xs]), color='k', linestyle='--')
        bplot = plt.boxplot(X, patch_artist=True, showfliers=False, labels =sorted_communities)
        for i, boxes in enumerate(bplot['boxes']):
            boxes.set_facecolor(COLORS[i%len(COLORS)])
        for i, medians in enumerate(bplot['medians']):
            medians.set_color('k')
        #for i, fliers in enumerate(bplot['fliers']):
        #    fliers.set_mec(COLORS[i%len(COLORS)])
        ax.set_title(quantity, fontsize=20)
        plt.xlabel('community', fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        plt.savefig('boxplot_%s' % quantity.replace('.', ''), facecolor='w')
        plt.close()

def plot_WQ_dynamic(df, community_counts_dict):


    files = glob.glob("WQ_dynamic_window_size_1y/*.csv")
    for file in files:
        quantity_name = file.split('/')[-1].split('.')[0]
        df_WQ = pd.read_csv(file, sep=",", encoding = "utf-8", encoding_errors='replace', dtype={"STAID":"str"})
        if len(df_WQ)==0:continue
        df_WQ = pd.merge(df[['STAID', 'community']], df_WQ, on="STAID")
        df_WQ["community_size"] = pd.Series([community_counts_dict[x] for x in df_WQ["community"].values])
        df_WQ = df_WQ[df_WQ["community_size"]>=50]

        #df_WQ = df_WQ.groupby('community').agg(np.mean)
        #df_WQ.drop(["STAID", "community_size"], axis=1, inplace=True)
        sorted_communities = sorted(df_WQ["community"].unique())
        fig, ax = plt.subplots(figsize=(16,12))
        for j in sorted_communities:
            df_WQ_community = df_WQ[df_WQ["community"]==j]
            community_size = df_WQ_community["community_size"].values.tolist()[0]
            df_WQ_community = df_WQ_community.drop(columns=["STAID", "community", "community_size"])
            X = df_WQ_community.columns.values.astype(float)
            df_WQ_community_values = df_WQ_community.values
            Y = np.nanmean(df_WQ_community_values, axis = 0)
            #print(quantity_name, j, np.sum(~np.isnan(df_WQ_community_values), axis=0), community_size)
            if quantity_name.startswith("RDC"): threshold=0.1
            else: threshold=0.05
            not_enough_gages = np.sum(~np.isnan(df_WQ_community_values), axis=0)<(threshold*community_size)
            Y_err = np.nanstd(df_WQ_community_values, ddof=1, axis=0) / np.sqrt(np.sum(~np.isnan(df_WQ_community_values), axis=0))
            Y[not_enough_gages] = np.nan
            Y_err[not_enough_gages] = np.nan

            if len(Y[~np.isnan(Y)])==0:
                continue
            #if (("mean" in quantity_name) | ("median" in quantity_name) | ("perc" in quantity_name)):
            #    Y = Y/np.nanmean(Y)
            Y, Y_err = moving_average(Y, window = 5, error=Y_err)
            X = ((X[:len(Y)]+X[-len(Y):])/2.).astype(str)
            ax.errorbar(X, Y, yerr=Y_err, fmt='-o', label=j, color=COLORS[j%len(COLORS)])
            ax.text(X[~np.isnan(Y)][np.argmax(Y[~np.isnan(Y)])], Y[~np.isnan(Y)][np.argmax(Y[~np.isnan(Y)])], str(j), fontsize=20, color=COLORS[j%len(COLORS)], zorder=6)
            ax.text(X[~np.isnan(Y)][-1], Y[~np.isnan(Y)][-1], str(j), fontsize=20, color=COLORS[j%len(COLORS)], zorder=6)

        ax.set_title(quantity_name, fontsize=20)
        plt.xticks(rotation = 45)
        plt.xlabel('year', fontsize=18)
        #if ('times' not in quantity_name) and ('zero_flow' not in quantity_name):
        #ax.set_yscale('symlog')

        #ax.tick_params(axis='x', labelsize=16)
        #ax.tick_params(axis='y', labelsize=16)
        #ax.legend(loc=0)
        plt.savefig('timeline_%s' % quantity_name.replace('.', ''), facecolor='w')
        plt.close()

def plot_distribution(vals, name_vals, bins_number, suffix_filename='', loglog=False, same_binning=True):
    flat_list = [item for sublist in vals for item in sublist]
    vals_hist = []
    bins = []
    for i, val in enumerate(vals):
        if same_binning:
            if loglog:
                bins_i = np.logspace(np.log10(np.min(flat_list)), np.log10(np.max(flat_list)), bins_number[i])    
            else:
                bins_i = np.linspace(np.min(flat_list), np.max(flat_list), bins_number[i])
        else:
            if loglog:
                bins_i = np.logspace(np.log10(np.min(val)), np.log10(np.max(val)), bins_number[i])
            else:
                bins_i = np.linspace(np.min(val), np.max(val), bins_number[i])
        bins.append(bins_i)
        vals_hist.append(np.histogram(val, bins=bins_i, density=True)[0])

    fig, ax = plt.subplots()
    for i, val_hist in enumerate(vals_hist):
        val_hist[val_hist == 0] = np.nan
        ax.plot((bins[i][1:]+bins[i][:-1])/2., val_hist, 'o--', label=name_vals[i])
    #ax.plot((bins[1:]+bins[:-1])/2., kl)
    #plt.axvline(x=t, color='red', linestyle='--')

    plt.xlabel(suffix_filename.replace('_', ' '), fontsize=16)
    plt.ylabel('PDF', fontsize=16)

    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')
        suffix_filename += "_loglog"

    ax.legend(loc=0)
    plt.savefig('distribution_%s' % suffix_filename)
    plt.close()
