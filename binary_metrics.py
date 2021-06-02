import numpy as np
from sklearn import metrics
import scipy.stats

import torch
from utils import  *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device ='cpu'

# %%

def get_auc(y, scores):
    y = np.array(y).astype(int)
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(y, scores)
    return roc_auc, aupr


def report_metrics(G_true, G, beta=1, batch_size=1):
    FDRs, TPRs, FPRs, SHDs, Ts, Ps, precisions, recalls, F_betas, auprs, aucs = \
        [], [], [], [], [], [], [], [], [], [], []
    for b in range(batch_size):
        G_true = G_true[b].real
        G =G[b].real
        #print('Check report metrics: ', G_true, G)
        # G_true and G are numpy arrays
        # convert all non-zeros in G to 1
        d = G.shape[-1]

        # changing to 1/0 for TP and FP calculations
        G_binary = np.where(G!=0, 1, 0)
        G_true_binary = np.where(G_true!=0, 1, 0)
        # extract the upper diagonal matrix
        indices_triu = np.triu_indices(d, 1)
        edges_true = G_true_binary[indices_triu] #np.triu(G_true_binary, 1)
        edges_pred = G_binary[indices_triu] #np.triu(G_binary, 1)
        # Getting AUROC value
        edges_pred_auc = G[indices_triu] #np.triu(G_true_binary, 1)
        auc, aupr = get_auc(edges_true, np.absolute(edges_pred_auc))
        # Now, we have the edge array for comparison
        # true pos = pred is 1 and true is 1
        TP = np.sum(edges_true * edges_pred) # true_pos
        # False pos = pred is 1 and true is 0
        mismatches = np.logical_xor(edges_true, edges_pred)
        FP = np.sum(mismatches * edges_pred)
        # Find all mismatches with Xor and then just select the ones with pred as 1
        # P = Number of pred edges : nnz_pred
        P = np.sum(edges_pred)
        # T = Number of True edges :  nnz_true
        T = np.sum(edges_true)
        # F = Number of non-edges in true graph
        F = len(edges_true) - T
        # SHD = total number of mismatches
        SHD = np.sum(mismatches)
        # FDR = False discovery rate
        FDR = FP/P
        # TPR = True positive rate
        TPR = TP/T
        # FPR = False positive rate
        FPR = FP/F
        # False negative = pred is 0 and true is 1
        FN = np.sum(mismatches * edges_true)
        # F beta score
        num = (1+beta**2)*TP
        den = ((1+beta**2)*TP + beta**2 * FN + FP)
        F_beta = num/den
        # precision
        precision = TP/(TP+FP)
        # recall
        recall = TP/(TP+FN)

        FDRs.append(FDR)
        TPRs.append(TPR)
        FPRs.append(FPR)
        SHDs.append(SHD)
        Ts.append(T)
        Ps.append(P)
        precisions.append(precision)
        recalls.append(recall)
        F_betas.append(F_beta)
        auprs.append(aupr)
        aucs.append(auc)
#    print('FDR, TPR, FPR, SHD, nnz_true, nnz_pred, F1, auc')
    return np.array(FDRs).mean(), np.array(TPRs).mean(), np.array(FPRs).mean(),np.array(SHDs).mean(),\
           np.array(Ts).mean(), np.array(Ps).mean(), np.array(precisions).mean(), np.array(recalls).mean(),\
           np.array(F_betas).mean(), np.array(auprs).mean(), np.array(aucs).mean()


def mean_confidence_interval(data, confidence=0.95):
    # print(np.array(data))
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n- 1)
    return m, h, m - h, m + h


def binary_metrics_batch(adj_true_batch, w_pred_batch, device, beta=1):
    G_pred_batch = torch_sqaureform_to_matrix(w_pred_batch, device=device).detach().cpu()
    G_true_batch = torch_sqaureform_to_matrix(adj_true_batch, device=device).detach().cpu()

    batch_size = G_pred_batch.size()[0]

    AUC = [report_metrics(G_true_batch[i, :, :], G_pred_batch[i, :, :], beta=1)['auc'] for i in range(batch_size)]
    auc_mean, auc_ci, _, _ = mean_confidence_interval(np.array(AUC), confidence=0.95)

    APS = [report_metrics(G_true_batch[i, :, :], G_pred_batch[i, :, :], beta=1)['aupr'] for i in range(batch_size)]
    aps_mean, aps_ci, _, _ = mean_confidence_interval(np.array(APS), confidence=0.95)

    # FDR = [report_metrics(G_true_batch[i, :, :], G_pred_batch[i, :, :], beta=1)['FDR'] for i in range(batch_size)]
    # TPR = [report_metrics(G_true_batch[i, :, :], G_pred_batch[i, :, :], beta=1)['TPR'] for i in range(batch_size)]
    # FPR = [report_metrics(G_true_batch[i, :, :], G_pred_batch[i, :, :], beta=1)['FPR'] for i in range(batch_size)]

    result = {
        'auc_mean': auc_mean,
        'auc_ci': auc_ci,
        'aps_mean': aps_mean,
        'aps_ci': aps_ci
    }

    print(result)

    return result


#%%