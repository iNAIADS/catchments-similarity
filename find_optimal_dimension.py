import pandas as pd
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utils import moving_average


def make_rank(M, inverse_rank=True):
    print("rank")
    return rankdata(((-1)**int(inverse_rank))*M, axis=1, method='min')

def make_comparison_full_rank(Sims):
    Rs = [make_rank(Sim) for Sim in Sims]
    
    #return np.array([np.mean(2*np.abs(Rs[i]-Rs[i+1])/(Rs[i]+Rs[i+1])) for i in range(len(Rs)-1)])
    return np.array([np.mean(np.abs(Rs[i]-Rs[i+1])/Rs[i+1]) for i in range(len(Rs)-1)])

def make_comparison_nn(Sims, ks, R_tol=15):

    n = len(Sims)
    print(n)
    max_idxs = [Sims[i].argmax(axis=1) for i in range(n)]
    print(max_idxs[0])
    l = np.arange(len(Sims[0]))
    
    Rs = []
    for i in range(n-1):
        A = 1.-(Sims[i+1][l,max_idxs[i]])
        B = 1.-(Sims[i][l,max_idxs[i]])

        #R = np.sum(np.sqrt((A**2 - B**2)/(B**2))>R_tol)
        R = np.mean(A)/np.mean(B) #CAO
        print(ks[i], R, np.mean(A), np.mean(B))
        Rs.append(R)

    return np.array(Rs)

def find_elbow(x, y):

    f = interp1d(x,y, kind='linear')

    xhat = np.linspace(min(x), max(x), 31)
    yhat=moving_average(f(xhat), window=5)

    dev1 = yhat[1:]-yhat[:-1]
    dev1 = (dev1[1:]+dev1[:-1])/2.
    dev2 = yhat[:-2]+yhat[2:]-(2*yhat[1:-1])
    curv = dev2/np.power(1+(dev1**2), 3./2.)
    
    max_curv_idx = np.argmax(curv)
    max_curv_x = xhat[1:-1][max_curv_idx]

    return np.argmin(np.abs(x-max_curv_x)), xhat, yhat

def make_plot(ks, diff, elbow_index, xhat, yhat):
    fig, ax = plt.subplots(figsize = (9,6))
    ax.plot(ks, diff, 'o-')
    plt.axhline(y=1.05, color='r', linestyle='--', label="R = 1.05")
    #print([diff<1.05])
    #print(ks[diff<1.05])
    #print(ks[diff<1.05][0])
    #print(diff[diff<1.05][0])
    #ax.plot([ks[diff<1.05][0]], [diff[diff<1.05][0]], "o", color='red')#, label="k = %d" % ks[ks<1.05][0])
    #ax.plot(xhat, yhat, 'o-')
    #ax.scatter(ks[elbow_index], diff[elbow_index], color='red', zorder=7)
    #ax.set_yscale("log")
    ax.set_xlabel("k")
    ax.set_ylabel("R(k)")
    ax.legend(loc=0)
    plt.savefig('dims')

def similarity_wrapper(M, similarity_funct):
    S = similarity_funct(M)
    S = S[~np.eye(S.shape[0],dtype=bool)].reshape(S.shape[0],-1)

    return S


def wrapper(U, S, VT, similarity_funct, do_plot=True):
    #ks = np.array([3,4,5,6,7,8,9,10,12,15,20,25,30,40,50])
    ks = np.arange(5, np.min([50, U.shape[1]]))
    print(ks)
    Ms = [U[:,:k].dot(np.diag(np.array(S[:k]))).dot(VT[:k,:]) for k in ks]
    #Ms = [U[:,:k].dot(np.diag(S[:k])) for k in ks]
    print("uno")
    print([M.shape for M in Ms])
    Sims = []
    for i,M in enumerate(Ms):
        Sim = similarity_wrapper(M, similarity_funct)
        print(Sim.shape)
        Sims.append(Sim)
    print("due")
    diff = make_comparison_nn(Sims, ks)
    #diff = make_comparison_full_rank(Sims)
    print("tre")
    elbow_index, xhat, yhat = find_elbow(ks[:-1], diff)
    print(elbow_index)
    if do_plot:
        make_plot(ks[:-1], diff, elbow_index, xhat, yhat)

    return ks[elbow_index]

    

