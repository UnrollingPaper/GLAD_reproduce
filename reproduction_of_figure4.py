import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn
from data_generator_glad import *
import pickle as pkl

K_train = 10 # 10
M = 100 # number of samples
N = 36 # number of features 100
graph_type = "random_maxd"
SAMPLE_BATCHES = 10
w_min = 0.1
w_max = 0.4
SIGNS = 0
K_test = 100 # 100
prob = 0.05
MAX_DEG = 50
K_valid = 10
data_path = "./data/syn/Ktrain{}_Ktest{}_M{}_N{}_prob{}.pkl".format(K_train, K_test, M, N, prob)

# generate data
def prepare_data_helper(graphs):
    theta, s = [], [] # precision_mat, samples covariance mat
    for g_num in graphs:
        precision_mat, data = graphs[g_num] # data = M x N
        theta.append(precision_mat)
        s.append(np.matmul(data.T, data)/(M))
        # check whether the diagonals are all positive
        if np.all(s[-1].diagonal() > 0) == False:
            print('Diagonals of emp cov matrix are negative: CHECK', s, s[-1].diagonal())
#        else:
#            print('Diagonals of emp cov matrix are positive:', s[-1].diagonal())

    theta = np.array(theta)
    s     = np.array(s)

    # plt.figure()
    # seaborn.heatmap(theta[0], cmap="pink_r", vmax=1)
    # plt.show()

    return [theta, s]

def prepare_data(mn):
    train_data = prepare_data_helper(mn.train_graphs)
    valid_data = []
    if K_valid > 0:
        valid_data = prepare_data_helper(mn.valid_graphs)
    test_data  = prepare_data_helper(mn.test_graphs)
    return train_data, valid_data, test_data


if not os.path.exists(data_path):
    print("Generating data...")
    mn = create_MN_vary_w(K_train, M, N, graph_type, SAMPLE_BATCHES, [w_min, w_max], K_test, [prob, MAX_DEG, SIGNS], K_valid)
    train_data, valid_data, test_data = prepare_data(mn)
    print("Data generation finished.")

    with open(data_path, 'wb') as f:
        pkl.dump({"train_data":train_data, "valid_data":valid_data, "test_data":test_data}, f)

else:
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
        train_data, valid_data, test_data = data['train_data'], data['valid_data'], data['test_data']






from unrolled_model import *
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import random
import metrics

# training parameters
TRAIN = True
USE_CUDA = True
L = 30
rho_init = 1
lambda_init = 1
theta_init_offset = 1e-2
gamma_init = 0.1
nF = 3
H = 3
init_lr = 0.001
use_optimizer = 'adam'
batch_size = 1
train_epochs = 1
INIT_DIAG = 0
lossBCE = 0
loss_signed = 0
lossL1 = 0
model_path = "M{}_N{}_Ktrain{}_gt{}_prob{}_bs{}_L{}_lr{}.pth".format(M, N, K_train, graph_type, prob, batch_size, L, init_lr)




# functions
def train_glasso(data, valid_data=[]):  # tied lista
    #    torch.set_grad_enabled(True)
    print('training GLASSO')
    theta, S = data
    #    theta = theta[0]
    if len(valid_data) > 0:
        valid_theta, valid_S = valid_data
        valid_theta_true = convert_to_torch(valid_theta, TESTING_FLAG=True, USE_CUDA=USE_CUDA)
        valid_S = convert_to_torch(valid_S, TESTING_FLAG=True, USE_CUDA=USE_CUDA)
    # theta -> K_train x N x N (Matrix)
    # S -> K_train x N x N (observed vector)
    # train using ALISTA style training.
    # model = threshold_NN_lambda_single_model(L, rho_init, lambda_init, theta_init_offset, gamma_init, N, nF, H, USE_CUDA=USE_CUDA)
    model = threshold_NN_lambda_unrolled_model(L, rho_init, lambda_init, theta_init_offset, gamma_init, N, nF, H, USE_CUDA=USE_CUDA)

    model.train()
    theta_true = convert_to_torch(theta, TESTING_FLAG=True, USE_CUDA=USE_CUDA)
    S = convert_to_torch(S, TESTING_FLAG=True, USE_CUDA=USE_CUDA)

    zero = torch.Tensor([0])  # .type(self.dtype)

    #    print('check: theta ', theta_init.shape)
    #    print('true: ', theta_true)
    print('parameters to be learned')
    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)
    dtype = torch.FloatTensor
    if USE_CUDA:
        model = model.cuda()
        zero = zero.cuda()
        dtype = torch.cuda.FloatTensor

    lr = init_lr
    if use_optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, rho=0.9, eps=1e-06,
                                         weight_decay=0)  # LR range = 5 ->
    elif use_optimizer == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0.25,
                                        centered=False)
    elif use_optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0,
                                    nesterov=False)
    elif use_optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    else:
        print('Optimizer not found!')
    # scheduler = MultiStepLR(optimizer, milestones=[1000], gamma=0.5)
    scheduler = MultiStepLR(optimizer, milestones=[10, 15, 20, 25, 100, 200], gamma=0.25)
    # scheduler = MultiStepLR(optimizer, milestones=[10, 15, 20, 100, 200], gamma=0.25)
    # scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 200], gamma=0.5)
    # criterion = nn.MSELoss(reduction="sum")  # input, target
    criterion = nn.MSELoss()  # input, target
    criterion_L1 = nn.L1Loss()
    m_sig = nn.Sigmoid()
    criterionBCE = nn.BCELoss()

    # batch size is fixed
    #    num_batches = int(args.K_train/args.batch_size)
    #    if args.SAMPLE_BATCHES > 0:
    #        num_batches = int(args.K_train*args.SAMPLE_BATCHES/args.batch_size)

    num_batches = int(len(S) / batch_size)

    #    if args.K_train >= 10:
    #        args.batch_size = 10
    # best_shd_model = model
    best_valid_shd, best_valid_ps, best_valid_nmse = np.inf, -1 * np.inf, np.inf
    EARLY_STOP = False
    for epoch in range(train_epochs):  # 1 epoch is expected to go through complete data
        scheduler.step()
        #        if epoch%1==0:
        #            for param_group in optimizer.param_groups:
        #                print('epoch: ', epoch, ' lr ', param_group['lr'])
        epoch_loss = []
        frob_loss = []
        duality_gap = []
        mse_binary_loss = []
        bce_loss = []
        if EARLY_STOP:
            break
        #        print('ecpohc ', epoch)
        for batch_num in range(num_batches):  # processing batchwise
            optimizer.zero_grad()
            # resetting the loss to zero
            loss = torch.Tensor([0]).type(dtype)
            # Get a batch
            # ridx = random.sample(list(range(args.K_train)), args.batch_size)
            ridx = random.sample(list(range(len(S))), batch_size)
            Sb = S[ridx]  # [0]
            #            print('errr train check : ', batch_num, theta_true, Sb, theta_true.expand_as(Sb))

            if INIT_DIAG == 1:
                # print(' extract batchwise diagonals, add offset and take inverse')
                batch_diags = 1 / (torch.diagonal(Sb, offset=0, dim1=-2, dim2=-1) + model.theta_init_offset)
                theta_init = torch.diag_embed(batch_diags)
            else:
                # print('***************** (S+theta_offset*I)^-1 is used')
                theta_init = torch.inverse(
                    Sb + model.theta_init_offset * torch.eye(Sb.shape[-1]).expand_as(Sb).type_as(Sb))


            # theta_pred = S_inv[r_idx]
            # ll = torch.cholesky(theta_init[ridx])#(theta_pred) # lower triangular
            # ll = my_cholesky(theta_init[ridx][0])#(theta_pred) # lower triangular
            # ll = batch_cholesky(theta_init[ridx])#(theta_pred) # lower triangular
            theta_pred = theta_init  # [ridx]
            # theta_pred = theta_init[ridx]
            # theta_pred.requires_grad = True
            # Sb = S[ridx][0]
            # step_size = get_init_step_size(theta_init[ridx])
            identity_mat = torch.eye(Sb.shape[-1]).expand_as(Sb)
            if USE_CUDA == True:
                identity_mat = identity_mat.cuda()
            # print('ERRR check: ', theta_pred.shape, get_frobenius_norm(theta_pred), get_frobenius_norm(theta_pred).shape)
            # lambda_k = model.lambda_f(get_frobenius_norm(theta_pred))
            lambda_k = model.lambda_forward(zero + lambda_init, zero, k=0)
            for k in range(L):
                #                print('itr = ', itr, theta_pred)#, theta_true[ridx])
                # step 1 : AM
                b = 1.0 / lambda_k * Sb - theta_pred
                b2_4ac = torch.matmul(b.transpose(-1, -2), b) + 4.0 / lambda_k * identity_mat
                sqrt_term = batch_matrix_sqrt(b2_4ac)
                theta_k1 = 1.0 / 2 * (-1 * b + sqrt_term)
                """
                # extract the diagonals of the matrices
                theta_diag = torch.diag_embed(torch.diagonal(theta_k1, offset=0, dim1=-2, dim2=-1))
                # soft threshold on remaining entries 
                theta_pred = model.eta_forward(theta_k1-theta_diag, k)
                # add the diagonals
                theta_pred = theta_pred + theta_diag
                """
                # softthresholding on all the entries
                # theta_pred = model.eta_forward(theta_k1, k)

                # if MODEL_type == 'th':
                #     # soft thresholding + eigenvalue correctness term
                #     theta_pred = model.eta_forward(theta_k1, k) + torch.max(model.gamma_c[k],
                #                                                             zero + 1e-2) * identity_mat
                # elif MODEL_type == 'th_NN':
                theta_pred = model.eta_forward(theta_k1, Sb, k, theta_pred)  #
                # update the lambda
                lambda_k = model.lambda_forward(torch.Tensor([get_frobenius_norm(theta_pred - theta_k1)]).type(dtype),
                                                lambda_k, k)
                # accumulating loss
                #                print('k= ', k, ' lambda_value ', lambda_k, get_frobenius_norm(theta_pred-theta_k1))

                loss += criterion(theta_pred, theta_true[ridx]) / L
                # loss += criterion(theta_pred, theta_true.expand_as(theta_pred))/args.L

            #            print('k= ', k, ' lambda_value ', lambda_k)
            # print('thetapred: ', theta_pred, check_sym(theta_pred[0].data.cpu().numpy()))
            # delta = batch_duality_gap(theta_pred, Sb, model.rho)
            # NOTE: ******* IMP: Change thissss!@!! ****************************
            delta = torch.ones([1]) * -1

            loss += criterion(theta_pred, theta_true[ridx]) / L

            # loss += criterion(theta_pred, theta_true.expand_as(theta_pred))/args.L

            lossf = get_frobenius_norm(theta_pred - theta_true[ridx])
            # total_loss = loss #+ delta
            # total_loss = lossB #+ loss #+ delta
            # total_loss = loss + lossB+ delta + lossBCE
            #            total_loss = lossBCE
            # total_loss = delta
            if lossBCE == 1:  # binary cross entropy
                total_loss = lossBCE
            elif loss_signed == 1:  # signed loss
                # total_loss = criterion(torch.sign(theta_pred), torch.sign(theta_true.expand_as(theta_pred)))
                total_loss = criterion(theta_pred, torch.sign(theta_true.expand_as(theta_pred)))
            elif lossL1 == 1:  # signed loss
                total_loss = criterion_L1(theta_pred, theta_true.expand_as(theta_pred))
            else:  # frobenius norm
                total_loss = loss
                # total_loss = lossf
            #            total_loss.requires_grad = True
            #            print('err: ', total_loss, total_loss.requires_grad)

            lv = loss.data.cpu().numpy()
            if lv <= 1e-7:  # loss value
                print('Early stopping as loss = ', lv)
                EARLY_STOP = True
                break

            total_loss.backward()
            # delta.backward()

            #            for name, param in model.named_parameters():
            #                print('befoer: ', name, param)
            optimizer.step()

            #            for name, param in model.named_parameters():
            #                print('after: ', name, param)

            #            mse_binary_loss.append(lossB.data.cpu().numpy())
            #            bce_loss.append(lossBCE.data.cpu().numpy())
            #            duality_gap.append(delta.data.cpu().numpy())
            frob_loss.append(lossf.data.cpu().numpy())
            epoch_loss.append(loss.data.cpu().numpy())
        if epoch % 2 == 0 and EARLY_STOP == False:
            # print(len(epoch_loss))
            # print('loss_summary: MSE: ', sum(epoch_loss)/len(epoch_loss), ' Mean Frobenius loss: ',sum(frob_loss)/len(frob_loss), ' MSE_binary loss: ', sum(mse_binary_loss)/len(mse_binary_loss), 'BCE_loss: ', sum(bce_loss)/len(bce_loss), 'duality gap = ', sum(duality_gap)/len(duality_gap))
            print('loss_summary: MSE: ', sum(epoch_loss)/len(epoch_loss), ' Mean Frobenius loss: ',sum(frob_loss)/len(frob_loss))
            if lossBCE == 1:
                print(epoch, sum(epoch_loss) / len(epoch_loss), sum(bce_loss) / len(bce_loss))
            else:
                print('loss_values: ', epoch, sum(epoch_loss) / len(epoch_loss))  # , sum(duality_gap)/len(duality_gap))
        # Checking the results on valid data and updating the best model
        if len(valid_data) > 0:
            # get the SHD on the valid data and the train data
            # curr_valid_shd, curr_valid_nmse = glasso_predict(model, valid_data)
            curr_valid_shd, curr_valid_ps, curr_valid_nmse, curr_valid_gmse = glasso_predict(model, valid_data)
            curr_train_shd, curr_train_ps, curr_train_nmse, curr_valid_gmse = glasso_predict(model, data)
            print('valid/train: shd %0.2f/%0.2f ps %0.2f/%0.2f nmse %0.2f/%0.2f' % (
            curr_valid_shd, curr_train_shd, curr_valid_ps, curr_train_ps, curr_valid_nmse, curr_train_nmse))
            #            if curr_valid_shd <= best_valid_shd:
            if curr_valid_ps >= best_valid_ps:
                print('epoch = ', epoch, ' Updating the best ps model with valid ps = ', curr_valid_ps)
                best_ps_model = copy.deepcopy(model)
                best_valid_ps = curr_valid_ps

            if curr_valid_shd <= best_valid_shd:
                print('epoch = ', epoch, ' Updating the best shd model with valid shd = ', curr_valid_shd)
                best_shd_model = copy.deepcopy(model)
                best_valid_shd = curr_valid_shd

            if curr_valid_nmse <= best_valid_nmse:
                print('epoch = ', epoch, ' Updating the best nmse model with valid nmse = ', curr_valid_nmse)
                best_nmse_model = copy.deepcopy(model)
                best_valid_nmse = curr_valid_nmse
                print('epoch = ', epoch, ' Updating the best nmse model with valid gmse = ', curr_valid_gmse)
            model.train()
            print('loss_summary:: epoch: ', epoch, ' loss: ', sum(epoch_loss)/len(epoch_loss))#, ' NMSE loss: ', 10*np.log10( (np.sum(np.array(epoch_loss)))/(len(epoch_loss)*E_norm_xtrue)))
    #    print('ans: ', theta_pred)
    #    print('true: ', theta_true)
    for name, param in model.named_parameters():
        print(name, param)
    # return best_ps_model # model
    return best_nmse_model, best_shd_model, best_ps_model  # model


def glasso_predict(model, data, flagP=False, SAVE_GRAPH=False, eM=0, name='', mn=''):
    with torch.no_grad():
        print('Running unrolled ADMM predict')
        # predict as a complete batch?
        model.eval()
        criterion = nn.MSELoss()  # input, target
        m_sig = nn.Sigmoid()
        criterionBCE = nn.BCELoss()
        theta, S = data
        #    theta = theta[0]
        # theta -> K_train x N x N (Matrix)
        # S -> K_train x N x N (observed vector)
        # theta_true = convert_to_torch(theta, TESTING_FLAG=True, USE_CUDA=False)
        theta_true = convert_to_torch(theta, TESTING_FLAG=True, USE_CUDA=USE_CUDA)
        S = convert_to_torch(S, TESTING_FLAG=True, USE_CUDA=USE_CUDA)

        zero = torch.Tensor([0])  # .type(self.dtype)
        dtype = torch.FloatTensor
        if USE_CUDA == True:
            zero = zero.cuda()
            model = model.cuda()
            dtype = torch.cuda.FloatTensor

        # batch size is fixed for testing as 1
        batch_size = 1
        print('CEHCKK: Total graphs = ', len(S))
        num_batches = int(len(S) / batch_size)
        #    print('num batches: ', num_batches)
        epoch_loss = []
        mse_binary_loss = []
        bce_loss = []
        frob_loss = []
        duality_gap = []
        ans = []
        if flagP:
            res_conv = {}
            for k in range(L + 1):
                res_conv[k] = []
        #        print('ITR, conv.loss, obj_val_pred, obj_val_true_model_rho, obj_val_pred_args_rho')#, theta_pred)

        res = []
        for batch_num in range(num_batches):  # processing batchwise
            # Get a batch
            # ll = my_cholesky(theta_init[ridx][0])#(theta_pred) # lower triangular
            # theta_pred = theta_init[batch_num*batch_size: (batch_num+1)*batch_size] #(theta_pred) # lower triangular
            # theta_true_b = theta_true[batch_num*batch_size: (batch_num+1)*batch_size]
            theta_true_b = theta_true[batch_num * batch_size: (batch_num + 1) * batch_size]
            Sb = S[batch_num * batch_size: (batch_num + 1) * batch_size]  # [0]
            identity_mat = torch.eye(Sb.shape[-1]).expand_as(Sb)
            if USE_CUDA == True:
                identity_mat = identity_mat.cuda()
                Sb = Sb.cuda()
            #    theta_pred = theta_pred.cuda()
            #    theta_true_b = theta_true_b.cuda()

            if INIT_DIAG == 1:
                # print(' extract batchwise diagonals, add offset and take inverse')
                batch_diags = 1 / (torch.diagonal(Sb, offset=0, dim1=-2, dim2=-1) + model.theta_init_offset)
                theta_pred = torch.diag_embed(batch_diags)
            else:
                # print('***************** (S+theta_offset*I)^-1 is used')
                theta_pred = torch.inverse(
                    Sb + model.theta_init_offset * torch.eye(Sb.shape[-1]).expand_as(Sb).type_as(Sb))

            # lambda_k = model.lambda_f(get_frobenius_norm(theta_pred))
            lambda_k = model.lambda_forward(zero + lambda_init, zero, k=0)

            #        if flagP:
            #            print('ITR, conv.loss, obj_val_pred, obj_val_true_model_rho, obj_val_pred_args_rho')#, theta_pred)
            for k in range(L):
                #            start = time.time()
                if flagP:
                    theta_pred_diag = torch.diag_embed(torch.diagonal(theta_pred[0], offset=0, dim1=-2, dim2=-1))
                    # theta_true_b_diag = torch.diag_embed(torch.diagonal(theta_true_b[0], offset=0, dim1=-2, dim2=-1))
                    theta_true_b_diag = torch.diag_embed(torch.diagonal(theta_true_b, offset=0, dim1=-2, dim2=-1))
                    # if MODEL_type == 'th':
                    #     cv_loss, cv_loss_off_diag, obj_pred, obj_true_rho, obj_true_orig = get_convergence_loss(
                    #         theta_pred[0], theta_true_b), get_convergence_loss(theta_pred[0] - theta_pred_diag,
                    #                                                            theta_true_b - theta_true_b_diag), get_obj_val(
                    #         theta_pred[0], Sb[0], model.rho_l1[k]), get_obj_val(theta_true_b, Sb[0],
                    #                                                             model.rho_l1[k]), get_obj_val(
                    #         theta_true_b, Sb[0], rho_init)
                    #     res_conv[k].append([cv_loss, obj_pred, obj_true_rho, obj_true_orig, cv_loss_off_diag])
                    # elif MODEL_type == 'th_NN':
                    cv_loss, cv_loss_off_diag = get_convergence_loss(theta_pred[0],
                                                                         theta_true_b), -1  # get_convergence_loss(theta_pred[0]-theta_pred_diag, theta_true_b-theta_true_b_diag)
                    res_conv[k].append([cv_loss, cv_loss_off_diag])
                # step 1 : AM
                b = 1.0 / lambda_k * Sb - theta_pred
                b2_4ac = torch.matmul(b.transpose(-1, -2), b) + 4.0 / lambda_k * identity_mat
                sqrt_term = batch_matrix_sqrt(b2_4ac)
                theta_k1 = 1.0 / 2 * (-1 * b + sqrt_term)

                # step 2 : AM
                """ 
                # extract the diagonals of the matrices
                theta_diag = torch.diag_embed(torch.diagonal(theta_k1, offset=0, dim1=-2, dim2=-1))
                # soft threshold on remaining entries 
                theta_pred = model.eta_forward(theta_k1-theta_diag, k)
                # add the diagonals
                theta_pred = theta_pred + theta_diag
                """
                # soft thresholding on all the entries
                # theta_pred = model.eta_forward(theta_k1, k)
                # if MODEL_type == 'th':
                #     # soft thresholding + eigenvalue correctness term
                #     theta_pred = model.eta_forward(theta_k1, k) + torch.max(model.gamma_c[k],
                #                                                             zero + 1e-2) * identity_mat
                # elif MODEL_type == 'th_NN':
                    # theta_pred = model.eta_forward(theta_k1, Sb, k) + torch.max(model.gamma_c[k], zero+1e-2) * identity_mat
                    # theta_pred = model.eta_forward(theta_k1, Sb, k, theta_pred) + torch.max(model.gamma_c[k], zero+1e-2) * identity_mat
                theta_pred = model.eta_forward(theta_k1, Sb, k,
                                                   theta_pred)  # + torch.max(model.gamma_c[k], zero+1e-2) * identity_mat

                # updating lambda
                lambda_k = model.lambda_forward(torch.Tensor([get_frobenius_norm(theta_pred - theta_k1)]).type(dtype),
                                                lambda_k, k)
                # lambda_k = model.lambda_f(get_frobenius_norm(theta_pred-theta_k1))
            #            print('k= ', k, ' lambda_value ', lambda_k, get_frobenius_norm(theta_pred-theta_k1))
            #            stop = time.time()
            #            print('Walltimes: ', k, stop-start)
            #        br
            if flagP:
                theta_pred_diag = torch.diag_embed(torch.diagonal(theta_pred[0], offset=0, dim1=-2, dim2=-1))
                # Getting the final predicted convergence loss
                if torch.min(torch.eig(theta_pred[0])[0][:, 0]) == 0:
                    adjust_eval_identity = torch.eye(theta_pred.shape[-1]).expand_as(theta_pred[0]).type_as(
                        theta_pred[0])
                    print('Adjusting the minimum eigenvalue to 1, SHOULD NOT BE CALLED AFTER THE GAMMA ADDITION!')
                    theta_pred[0] += adjust_eval_identity  # change the eigenval to 1
                # if MODEL_type == 'th':
                #     cv_loss, cv_loss_off_diag, obj_pred, obj_true_rho, obj_true_orig = get_convergence_loss(
                #         theta_pred[0], theta_true_b), get_convergence_loss(theta_pred[0] - theta_pred_diag,
                #                                                            theta_true_b - theta_true_b_diag), get_obj_val(
                #         theta_pred[0], Sb[0], model.rho_l1[k]), get_obj_val(theta_true_b, Sb[0],
                #                                                             model.rho_l1[k]), get_obj_val(theta_true_b,
                #                                                                                           Sb[0],
                #                                                                                           args.rho_init)
                #     res_conv[k + 1].append([cv_loss, obj_pred, obj_true_rho, obj_true_orig, cv_loss_off_diag])
                # elif MODEL_type == 'th_NN':
                cv_loss, cv_loss_off_diag = get_convergence_loss(theta_pred[0],
                                                                     theta_true_b), -1  # get_convergence_loss(theta_pred[0]-theta_pred_diag, theta_true_b-theta_true_b_diag)
                res_conv[k + 1].append([cv_loss, cv_loss_off_diag])

            plt.figure()
            seaborn.heatmap(theta_pred[0].cpu().detach().numpy(),cmap="pink_r")
            plt.show()

            plt.figure()
            seaborn.heatmap(theta_true_b[0].cpu().detach().numpy(),cmap="pink_r")
            plt.show()

            final_nmse, final_gmse = get_convergence_loss(theta_pred[0], theta_true_b)
            theta_pred = theta_pred[0].data.cpu().numpy()
            #        theta_true_b = theta_true_b[0].data.cpu().numpy()
            theta_true_b = theta_true_b.data.cpu().numpy()

            fdr, tpr, fpr, shd, nnz, nnz_true, ps = metrics.report_metrics(theta_true_b, theta_pred)
            print("ps:", ps)
            cond_theta_pred, cond_theta_true_b = np.linalg.cond(theta_pred), -1  # np.linalg.cond(theta_true_b)
            res.append([fdr, tpr, fpr, shd, nnz, nnz_true, ps, cond_theta_pred, cond_theta_true_b])

        res_mean = np.mean(np.array(res), 0)
        res_std = np.std(np.array(res), 0)
        res_mean = ["%.3f" % x for x in res_mean]
        res_std = ["%.3f" % x for x in res_std]

        if flagP:
            print('Structure learning Metrics')
            print('Average result over test graphs')
            # print('fdr, tpr, fpr, shd, nnz, nnz_true, np.linalg.cond(theta_pred), np.linalg.cond(theta_true)')
            print('fdr, tpr, fpr, shd, nnz, nnz_true, ps, np.linalg.cond(theta_pred), np.linalg.cond(theta_true)')
            print(*sum(list(map(list, zip(res_mean, res_std))), []), sep=', ')

            # print('ITR, conv.loss, obj_val_pred, obj_val_true_model_rho, obj_val_pred_args_rho')#, theta_pred)
            # print(
            #     'ITR, conv_loss"ecoli_M"+str(eM), obj_val_pred, obj_val_true_model_rho, obj_val_pred_args_rho, conv_loss_off_diag')  # , theta_pred)
            # for i in res_conv:
            #     mean_vec = ["%.3f" % x for x in np.mean(res_conv[i], 0)]
            #     std_vec = ["%.3f" % x for x in np.std(res_conv[i], 0)]
            #     print(i, *sum(list(map(list, zip(mean_vec, std_vec))), []), sep=', ')
        #            print(i, np.mean(res_conv[i], 0), np.std(res_conv[i], 0))

        if SAVE_GRAPH:
            x = np.where(theta_pred > 0, 1, 0)
            A = np.matrix(x - np.eye(x.shape[0]))
            G = nx.from_numpy_matrix(A)
            fig = plt.figure(figsize=(15, 15))
            mapping = {n1: n2 for n1, n2 in zip(G.nodes(), mn.nodes)}
            G = nx.relabel_nodes(G, mapping)
            # nx.draw_networkx(G, pos=nx.spring_layout(G), with_labels = True)
            # nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels = True)
            nx.draw_networkx(G, pos=nx.shell_layout(G), with_labels=True)
            plt.savefig(name + '_' + str(eM) + ".pdf", bbox_inches='tight')
            nx.draw_networkx(mn.G_true, pos=nx.shell_layout(G), with_labels=True)
            plt.savefig(name + '_true_' + str(eM) + ".pdf", bbox_inches='tight')
            # saving the graph
            # nx.write_gpickle(G, name+'_'+str(eM)+'.gpickle')
            nx.write_adjlist(G, name + '_' + str(eM) + '.adjlist')

        return np.float(res_mean[3]), np.float(res_mean[6]), final_nmse, final_gmse  # The PS mean value, final NMSE obtained


# training process
if TRAIN == True:
    print('Training the glasso model')
    print('check: ', train_data[0].shape, train_data[1].shape)
    nmse_model, shd_model, ps_model = train_glasso(train_data, valid_data)

model = nmse_model  # shd_model # or nmse_model

torch.save(nmse_model.state_dict(), model_path)

print('TIMING check:')
glasso_predict(model, train_data)


print('model trained: Predicting on...')
# torch.save(model.state_dict(), 'Gista_model.pt')
print('****Train Data, same pred matrix, different samples****')
#glasso_predict(nmse_model, train_data, True)
glasso_predict(model, train_data, True)
if len(valid_data)>0:
    print('****Valid Data****')
    #glasso_predict(nmse_model, valid_data, True)
    glasso_predict(model, valid_data, True)
print('****Test Data, model_NMSE: average results over different samples****')
glasso_predict(model, test_data, True)
print('****Test Data, tag: model_SHD : average results over different samples****')
glasso_predict(ps_model, test_data, True)