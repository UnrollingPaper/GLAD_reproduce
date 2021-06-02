import scipy
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import euclidean_distances
import scipy.sparse as sparse

#%%

def halfvec_to_topo(w, threshold, device = device):
    """
       from half vectorisation to matrix in batch way.
       """
    batch_size, l = w.size()
    m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))

    # extract binary edge {0, 1}:
    bw = (w.clone().detach() >= threshold).float().to(device)
    E = torch.zeros((batch_size, m, m), dtype = w.dtype).to(device)

    for i in range(batch_size):
        E[i, :, :][np.triu_indices(m, 1)] = bw[i].clone().detach()
        E[i, :, :] = E[i, :, :].T + E[i, :, :]

    return E

#%%

def torch_sqaureform_to_matrix(w, device):
    """
    from half vectorisation to matrix in batch way.
    """
    batch_size, l = w.size()
    m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))

    E = torch.zeros((batch_size, m, m), dtype = w.dtype).to(device)

    for i in range(batch_size):
        E[i, :, :][np.triu_indices(m, 1)] = w[i].clone().detach()
        E[i, :, :] = E[i, :, :].T + E[i, :, :]

    return E

#%%


def torch_squareform_to_vector(A, device):
    batch_size, m, _ = A.size()
    l = int(m * (m - 1) / 2)

    w = torch.zeros((batch_size, l), dtype = A.dtype).to(device)

    for i in range(batch_size):
        w[i, :] = A[i,:,:][np.triu_indices(m, 1)].clone().detach()

    return w

#%%

def soft_threshold(w, eta):
    '''
    softthreshold function in a batch way.
    '''
    return (torch.abs(w) >= eta) * torch.sign(w) * (torch.abs(w) - eta)


#%%

def check_tensor(x, device=None):
    if isinstance(x, np.ndarray) or type(x) in [int, float]:
        x = torch.Tensor(x)
    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    return x

#%%

def coo_to_sparseTensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


#%%

def get_degree_operator(m):
    ncols =int(m*(m - 1)/2)

    I = np.zeros(ncols)
    J = np.zeros(ncols)

    k = 0
    for i in np.arange(1, m):
        I[k:(k + m - i)] = np.arange(i, m)
        k = k + (m - i)

    k = 0
    for i in np.arange(1, m):
        J[k: (k + m - i)] = i - 1
        k = k + m - i

    Row = np.tile(np.arange(0, ncols), 2)
    Col = np.append(I, J)
    Data = np.ones(Col.size)
    St = sparse.coo_matrix((Data, (Row, Col)), shape=(ncols, m))
    return St.T

#%%

def get_distance_halfvector(y):
    n, _ = y.shape # m nodes, n observations
    z = (1 / n) * euclidean_distances(y.T, squared=True)
    # z.shape = m, m
    return squareform(z, checks=False)

#%%

def acc_loss(w_list, w, dn=0.9):
    num_unrolls, _ = w_list.size()

    if dn is None:
        # no accumulation, only the last unroll result
        loss = torch.sum((w_list[num_unrolls - 1, :] - w) ** 2) / torch.sum(w ** 2)

    elif dn == 1:
        # all layer matters
        loss = sum(torch.sum((w_list - w) ** 2, dim=1) / torch.sum(w ** 2))

    else:
        # cumulative loss
        factor = torch.tensor([dn ** i for i in range(num_unrolls, 0, -1)]).to(device)
        loss = sum(factor * torch.sum((w_list - w) ** 2, dim=1) / torch.sum(w ** 2))

    return loss

def gmse_loss(w_pred, w):
    loss = torch.sum((w_pred - w) ** 2) / torch.sum(w ** 2)
    return loss

def gmse_loss_batch_mean(w_pred, w):
    loss = torch.sum((w_pred - w) ** 2, dim = -1) / torch.sum(w ** 2, dim = -1)
    return loss.mean()

def gmse_loss_batch(w_pred, w):
    loss = torch.sum((w_pred - w) ** 2, dim = -1) / torch.sum(w ** 2, dim = -1)
    return loss

def layerwise_gmse_loss(w_list, w):
    num_unrolls, _ = w_list.size()
    loss = torch.sum((w_list - w) ** 2, dim=1) / torch.sum(w ** 2)
    return loss

#%%

def theta_to_w(theta, batch_size):
    W = -theta.cpu().detach().numpy()
    for i in range(20):
        W[:, i, i] = 0
    w_pred = []
    for i in range(batch_size):
        W[i] = (W[i] + np.transpose(W[i])) / 2
        # for j in range(20):
        #     for k in range(20):
        #         print(W[i][j][k],  end='\t')
        #     print('\n', end='')
        w_pred.append(scipy.spatial.distance.squareform(W[i]))
    w_pred = torch.Tensor(w_pred).cuda()
    w_pred = F.relu_(w_pred)
    return w_pred

def theta_to_w_train(theta, batch_size):
    W = -theta
    W = F.relu_(W)
    # W = F.leaky_relu(W)
    v = torch.zeros_like(W)
    mask = torch.diag_embed(torch.ones((batch_size, W.shape[2]))).cuda()
    W = mask*v + (1. - mask)*W
    W = (W + torch.transpose(W, 1, 2)) / 2
    return W

def is_spd(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def torch_sqaureform_to_torch_matrix(w, device):
    """
    from half vectorisation to matrix in batch way.
    """
    batch_size, l = w.shape
    m = int((1 / 2) * (1 + math.sqrt(1 + 8 * l)))

    E = torch.zeros((batch_size, m, m), dtype = w.dtype).to(device)

    for i in range(batch_size):
        inds = torch.triu_indices(m, m, 1)
        for j in range(inds.shape[1]):
            E[i, inds[0, j], inds[1, j]] = w[i][j]
        E = torch.transpose(E, 1, 2) + E
    return E