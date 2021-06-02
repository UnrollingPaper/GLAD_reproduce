
import numpy as np
import networkx as nx
import scipy

import multiprocess
from functools import partial

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils import *


#%%

def _generate_BA_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale = False):

    G = nx.barabasi_albert_graph(num_nodes, graph_hyper)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (ee) * np.eye(num_nodes))
    #signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    #z = get_distance_halfvector(signal)

    signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    # z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return signal, W_GT, L_GT + (ee) * np.eye(num_nodes), cov

def generate_BA_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale):
    n_cpu = multiprocess.cpu_count() - 1
    pool = multiprocess.Pool(n_cpu)

    s_multi, W_multi, p_multi, cov_multi = zip(*pool.map(partial(_generate_BA_to_parallel,
                                                        num_nodes = num_nodes,
                                                        num_signals = num_signals,
                                                        graph_hyper = graph_hyper,
                                                        weighted = weighted,
                                                        ee = ee,
                                                        weight_scale = weight_scale),
                                                range(num_samples)))

    result = {
        's': s_multi,
        'W': W_multi,
        'p': p_multi,
        'cov': cov_multi
    }

    return result

#%%

def _generate_ER_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale = False):

    G = nx.erdos_renyi_graph(num_nodes, graph_hyper)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 1e-02, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (ee) * np.eye(num_nodes))
    signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    # z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return signal, W_GT, L_GT + (ee) * np.eye(num_nodes), cov

def generate_ER_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale):
    n_cpu = multiprocess.cpu_count() - 1
    pool = multiprocess.Pool(n_cpu)

    s_multi, W_multi, p_multi, cov_multi = zip(*pool.map(partial(_generate_ER_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             ee = ee,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        's': s_multi,
        'W': W_multi,
        'p': p_multi,
        'cov': cov_multi
    }

    return result

#%%

def _generate_SBM20_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale = False):

    size = [8, 5, 5, 2]

    probs = [[graph_hyper, 0.0, 0.0, 0.0],
             [0.0, graph_hyper, 0.0, 0.0],
             [0.0, 0.0, graph_hyper, 0.0],
             [0.0, 0.0, 0.0, graph_hyper]]

    G = nx.stochastic_block_model(size, probs)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (ee) * np.eye(num_nodes))
    signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return signal, W_GT, L_GT + (ee) * np.eye(num_nodes), cov

def generate_SBM20_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale):
    n_cpu = multiprocess.cpu_count() - 1
    pool = multiprocess.Pool(n_cpu)

    s_multi, W_multi, p_multi, cov_multi = zip(*pool.map(partial(_generate_SBM20_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             ee = ee,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        's': s_multi,
        'W': W_multi,
        'p': p_multi,
        'cov': cov_multi
    }

    return result



#%%
def _generate_SBM20noise_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale = False):

    #cluster = 4
    #size_c = int( num_nodes / cluster)
    #size = [size_c] * cluster
    size = [7, 5, 5, 3]

    probs = [[graph_hyper, 0.05, 0.05, 0.05],
             [0.05, graph_hyper, 0.05, 0.05],
             [0.05, 0.05, graph_hyper, 0.05],
             [0.05, 0.05, 0.05, graph_hyper]]

    G = nx.stochastic_block_model(size, probs)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (ee) * np.eye(num_nodes))
    signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return signal, W_GT, L_GT + (ee) * np.eye(num_nodes), cov

def generate_SBM20noise_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale):
    n_cpu = multiprocess.cpu_count() - 1
    pool = multiprocess.Pool(n_cpu)

    s_multi, W_multi, p_multi, cov_multi = zip(*pool.map(partial(_generate_SBM20noise_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             ee = ee,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        's': s_multi,
        'W': W_multi,
        'p': p_multi,
        'cov': cov_multi
    }

    return result

#%%

def _generate_SBM50_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale = False):


    size = [15, 10, 7, 5, 6, 2, 5]

    probs = [[graph_hyper, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, graph_hyper, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, graph_hyper, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, graph_hyper, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, graph_hyper, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, graph_hyper, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, graph_hyper]]

    G = nx.stochastic_block_model(size, probs)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (ee) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def generate_SBM50_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale):
    n_cpu = multiprocess.cpu_count() - 1
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_SBM50_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             ee = ee,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result

#%%

def _generate_SBM50noise_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale = False):


    size = [15, 13, 10, 8, 4]

    probs = [[graph_hyper, 0.06,  0.06,  0.06,  0.06],
             [0.06, graph_hyper,  0.06,  0.06,  0.06],
             [0.06, 0.06, graph_hyper, 0.06, 0.06],
             [0.06, 0.06, 0.06, graph_hyper, 0.06],
             [0.06, 0.06, 0.06, 0.06, 0.99]]

    G = nx.stochastic_block_model(size, probs)

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (ee) * np.eye(num_nodes))
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return z, W_GT

def generate_SBM50noise_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale):
    n_cpu = multiprocess.cpu_count() - 1
    pool = multiprocess.Pool(n_cpu)

    z_multi, W_multi = zip(*pool.map(partial(_generate_SBM50noise_to_parallel,
                                             num_nodes = num_nodes,
                                             num_signals = num_signals,
                                             graph_hyper = graph_hyper,
                                             weighted = weighted,
                                             ee = ee,
                                             weight_scale = weight_scale),
                                     range(num_samples)))

    result = {
        'z': z_multi,
        'W': W_multi
    }

    return result


#%%


def _generate_WS_to_parallel(i, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale = False):

    G = nx.watts_strogatz_graph(num_nodes, k = graph_hyper['k'], p = graph_hyper['p'])

    W_GT = nx.adjacency_matrix(G).A

    if weighted == 'uniform':
        weights = np.random.uniform(0, 2, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'gaussian':
        weights = np.random.normal(1, 0.05, (num_nodes, num_nodes))
        weights = np.abs(weights)
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights

    if weighted == 'lognormal':
        weights = np.random.lognormal(0, 0.1, (num_nodes, num_nodes))
        weights = (weights + weights.T) / 2
        W_GT = W_GT * weights


    if weight_scale:
        W_GT = W_GT * num_nodes / np.sum(W_GT)

    L_GT = np.diag(W_GT @ np.ones(num_nodes)) - W_GT

    W_GT = scipy.sparse.csr_matrix(W_GT)

    cov = np.linalg.inv(L_GT + (ee) * np.eye(num_nodes))
    #signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    #z = get_distance_halfvector(signal)

    signal = np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals)
    z = get_distance_halfvector(np.random.multivariate_normal(np.zeros(num_nodes), cov, num_signals))

    return signal, W_GT, L_GT + (ee) * np.eye(num_nodes), cov

def generate_WS_parallel(num_samples, num_nodes, num_signals, graph_hyper, weighted, ee, weight_scale):
    n_cpu = multiprocess.cpu_count() - 2
    pool = multiprocess.Pool(n_cpu)

    s_multi, W_multi, p_multi, cov_multi = zip(*pool.map(partial(_generate_WS_to_parallel,
                                                                 num_nodes=num_nodes,
                                                                 num_signals=num_signals,
                                                                 graph_hyper=graph_hyper,
                                                                 weighted=weighted,
                                                                 ee=ee,
                                                                 weight_scale=weight_scale),
                                                         range(num_samples)))

    result = {
        's': s_multi,
        'W': W_multi,
        'p': p_multi,
        'cov': cov_multi
    }

    return result

