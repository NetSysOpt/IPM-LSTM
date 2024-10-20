import os
import sys
import numpy as np
import configargparse
import time
import torch
import torch.optim as optim
import scipy.io as sio

from models.LSTM import LSTM
from problems.QP import QP
from problems.Convex_QCQP import Convex_QCQP
from problems.Nonconvex_Program import Nonconvex_Program
from utils import EarlyStopping, calculate_step
from torch.utils.data import TensorDataset, DataLoader


parser = configargparse.ArgumentParser(description='train')
parser.add_argument('-c', '--config', is_config_file=True, type=str)

#optimizee settings
parser.add_argument('--mat_name', type=str, help='Imported mat file name.')
parser.add_argument('--num_var', type=int, help='Number of decision vars.')
parser.add_argument('--num_eq', type=int, help='Number of equality constraints.')
parser.add_argument('--num_ineq', type=int, help='Number of inequality constraints.')
parser.add_argument('--prob_type', type=str, help='Problem type.')

#model settings
parser.add_argument('--eq_tol', type=float, help='equality tolerance for model saving.')
parser.add_argument('--ineq_tol', type=float, help='inequality tolerance for model saving.')
parser.add_argument('--input_dim', type=int, default=2, help='Input feature dimensions of deep learning optimizer.')
parser.add_argument('--hidden_dim', type=int, help='The hidden dimensions of deep learning optimizer.')
parser.add_argument('--use_line_search', action='store_true', help='Using line search.')
parser.add_argument('--model_name', type=str, help='The deep learning optimizer name.')
parser.add_argument('--precondi', action='store_true', help='Preconditioning.')
parser.add_argument('--sigma', type=float, help='The coefficient of mu.')
parser.add_argument('--tau', type=float, help='The value of frac-and-boundary.')


#training settings
parser.add_argument('--batch_size', type=int, help='training batch size.')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data_size', type=int, help='The number of all instances.')
parser.add_argument('--lr', type=float, help='Learning rate.')
parser.add_argument('--num_epoch', type=int, help='The number of training epochs.')
parser.add_argument('--inner_T', type=int, help='The iterations of deep learning optimizer.')
parser.add_argument('--outer_T', type=int, help='The iterations of IPM.')
parser.add_argument('--patience', type=int, default=100, help='The patience of early stopping.')
parser.add_argument('--save_dir', type=str, default='./results/', help='Save path for the best model.')
parser.add_argument('--save_sol', action='store_true', help='Save the results.')
parser.add_argument('--seed', type=int, default=17, help='random seed.')
parser.add_argument('--test', action='store_true', help='Run in test mode.')
parser.add_argument('--test_solver', type=str, choices=['osqp', 'ipopt'], help='The solver type on the test set.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay rate.')



args, _ = parser.parse_known_args()

#model
if args.model_name == 'LSTM':
    model = LSTM(args.input_dim, args.hidden_dim, args.inner_T, args.device)

#optimizee
if args.prob_type == 'Nonconvex_QP':
    optimizee = QP
    file_path = os.path.join('datasets', 'qp', "{}.mat".format(args.mat_name))
    #model parameter save path
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                               args.mat_name,
                                                                               args.outer_T,
                                                                               args.inner_T))
elif args.prob_type == 'QP':
    optimizee = QP
    # model parameter save path
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                args.num_var,
                                                                                                args.num_ineq,
                                                                                                args.num_eq,
                                                                                                args.outer_T,
                                                                                                args.inner_T))
elif args.prob_type == 'QP_RHS':
    optimizee = QP
    mat_name = "dc3_{}_{}_{}_{}".format(args.num_var, args.num_ineq, args.num_eq, args.data_size)
    file_path = os.path.join('datasets', 'qp', "{}".format(mat_name))
    # model parameter save path
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))
elif args.prob_type == 'Nonconvex_Program':
    optimizee = Nonconvex_Program
    # model parameter save path
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))
elif args.prob_type == 'Nonconvex_Program_RHS':
    optimizee = Nonconvex_Program
    mat_name = "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(args.num_var, args.num_ineq, args.num_eq, args.data_size)
    file_path = os.path.join('datasets', 'nonconvex_program', "{}".format(mat_name))
    # model parameter save path
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))
elif args.prob_type == 'Convex_QCQP_RHS':
    optimizee = Convex_QCQP
    mat_name = "random_convex_qcqp_dataset_var{}_ineq{}_eq{}_ex{}".format(args.num_var,
                                                                         args.num_ineq,
                                                                         args.num_eq,
                                                                         args.data_size)
    file_path = os.path.join('datasets', 'convex_qcqp', "{}.mat".format(mat_name))
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))
elif args.prob_type == 'Convex_QCQP':
    optimizee = Convex_QCQP
    save_path = os.path.join(args.save_dir, model.name(), 'params', '{}_{}_{}_{}_{}_{}.pth'.format(args.prob_type,
                                                                                                   args.num_var,
                                                                                                   args.num_ineq,
                                                                                                   args.num_eq,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))



if not args.test:
    stopper = EarlyStopping(save_path, patience=args.patience)  #Early stopping detector

    #meta optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #training and validation datasets
    if (args.prob_type == 'QP') or (args.prob_type == 'Nonconvex_Program') or (args.prob_type == 'Convex_QCQP'):
        train_data = optimizee(prob_type=args.prob_type, learning_type='train', num_var=args.num_var, num_ineq=args.num_ineq,
                               num_eq=args.num_eq, data_size=args.data_size)
        val_data = optimizee(prob_type=args.prob_type, learning_type='val', num_var=args.num_var, num_ineq=args.num_ineq,
                             num_eq=args.num_eq, data_size=args.data_size)
    else:
        train_data = optimizee(prob_type=args.prob_type, learning_type='train', file_path=file_path)
        val_data = optimizee(prob_type=args.prob_type, learning_type='val', file_path=file_path)

    print('The number of variables: {}.'.format(train_data.num_var))
    if train_data.num_ineq != 0:
        print('The number of inequalities: {}.'.format(train_data.num_ineq))
    if train_data.num_eq != 0:
        print('The number of equalities: {}.'.format(train_data.num_ineq))
    if train_data.num_lb != 0:
        print('The number of lower bounds: {}.'.format(train_data.num_lb))
    if train_data.num_ub != 0:
        print('The number of upper bounds: {}.'.format(train_data.num_lb))


    for epoch in range(args.num_epoch):

        #training
        model.train()

        train_parameters = [train_data.Q, train_data.p]
        if train_data.num_ineq != 0:
            train_parameters.append(train_data.G)
            train_parameters.append(train_data.c)
        if train_data.num_eq != 0:
            train_parameters.append(train_data.A)
            train_parameters.append(train_data.b)
        if train_data.num_lb != 0:
            train_parameters.append(train_data.lb)
        if train_data.num_ub != 0:
            train_parameters.append(train_data.ub)

        train_dataset = TensorDataset(*train_parameters)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        train_start_time = time.time()
        for batch in train_loader:
            #Extract data
            train_Q, train_p, *remaining_parameters0 = batch
            if train_data.num_ineq != 0.0:
                train_G, train_c, *remaining_parameters1 = remaining_parameters0
                if train_data.num_eq != 0:
                    train_A, train_b, *remaining_parameters3 = remaining_parameters1
                    if train_data.num_lb !=0:
                        train_lb, *remaining_parameters5 = remaining_parameters3
                        if train_data.num_ub != 0:
                            train_ub = remaining_parameters5[0]
                        else:
                            train_ub = None
                    else:
                        train_lb = None
                        if train_data.num_ub != 0:
                            train_ub = remaining_parameters3[0]
                        else:
                            train_ub = None

                else:
                    train_A = None
                    train_b = None
                    if train_data.num_lb != 0:
                        train_lb, *remaining_parameters7 = remaining_parameters1
                        if train_data.num_ub != 0:
                            train_ub = remaining_parameters7[0]
                        else:
                            train_ub = None
                    else:
                        train_lb = None
                        if train_data.num_ub != 0:
                            train_ub = remaining_parameters1[0]
                        else:
                            train_ub = None
            else:
                train_G = None
                train_c = None
                if train_data.num_eq != 0:
                    train_A, train_b, *remaining_parameters2 = remaining_parameters0
                    if train_data.num_lb != 0:
                        train_lb, *remaining_parameters4 = remaining_parameters2
                        if train_data.num_ub != 0:
                            train_ub = remaining_parameters4[0]
                        else:
                            train_ub = None
                    else:
                        train_lb = None
                        if train_data.num_ub != 0:
                            train_ub = remaining_parameters2[0]
                        else:
                            train_ub = None
                else:
                    train_A = None
                    train_b = None
                    if train_data.num_lb != 0:
                        train_lb, *remaining_parameters6 = remaining_parameters0
                        if train_data.num_ub != 0:
                            train_ub = remaining_parameters6[0]
                        else:
                            train_ub = None
                    else:
                        train_lb = None
                        if train_data.num_ub != 0:
                            train_ub = remaining_parameters0[0]
                        else:
                            train_ub = None


            #Initialization
            if (train_data.num_lb !=0) and (train_data.num_ub !=0):
                train_x = (train_lb+train_ub)/2
            elif (train_data.num_lb !=0) and (train_data.num_ub ==0):
                train_x = train_lb + torch.ones(size=train_lb.shape, device=args.device)
            elif (train_data.num_lb ==0) and (train_data.num_ub !=0):
                train_x = train_ub - torch.ones(size=train_ub.shape, device=args.device)
            else:
                train_x = torch.zeros((train_Q.shape[0], train_data.num_var, 1), device=args.device)

            if train_data.num_ineq != 0:
                train_eta = torch.ones((train_Q.shape[0], train_data.num_ineq, 1), device=args.device)
                train_s = torch.ones((train_Q.shape[0], train_data.num_ineq, 1), device=args.device)
            else:
                train_eta = None
                train_s = None
            if train_data.num_eq != 0:
                train_lamb = torch.zeros((train_Q.shape[0], train_data.num_eq, 1), device=args.device)
            else:
                train_lamb = None
            if train_data.num_lb != 0:
                train_zl = torch.ones((train_Q.shape[0], train_data.num_lb, 1), device=args.device)
            else:
                train_zl = None
            if train_data.num_ub != 0:
                train_zu = torch.ones((train_Q.shape[0], train_data.num_ub, 1), device=args.device)
            else:
                train_zu = None

            #outer iteration
            for t_out in range(args.outer_T):
                #calculate the kkt information
                train_J, train_F, train_mu = train_data.cal_kkt(train_x, train_eta, train_s, train_lamb, train_zl, train_zu, args.sigma,
                                                                     Q=train_Q, p=train_p, G=train_G, c=train_c, A=train_A, b=train_b,
                                                                     lb=train_lb, ub=train_ub)

                # initialize the updates
                init_y = torch.zeros((train_Q.shape[0], train_data.num_var+2*train_data.num_ineq+train_data.num_eq+train_data.num_lb+train_data.num_ub, 1), device=args.device)

                #preconditioning
                if args.precondi:
                    train_D_values, train_D_id = (torch.bmm(train_J.permute(0, 2, 1), train_J)).max(-1)
                    train_D_inverse = torch.diag_embed(torch.sqrt(1 / train_D_values))
                    train_J_0 = torch.bmm(train_J, train_D_inverse)
                else:
                    train_J_0 = train_J

                train_y, train_loss, _ = model(train_data, init_y, train_J_0, train_F)

                optimizer.zero_grad()
                train_loss.backward(retain_graph=True)
                optimizer.step()

                if args.precondi:
                    train_y = torch.bmm(train_D_inverse, train_y)

                #stepsize
                delta_x = train_y[:, :train_data.num_var, :]
                delta_eta = train_y[:, train_data.num_var:train_data.num_var + train_data.num_ineq, :]
                delta_lamb = train_y[:,train_data.num_var + train_data.num_ineq:train_data.num_var + train_data.num_ineq + train_data.num_eq,:]
                delta_s = train_y[:,train_data.num_var + train_data.num_ineq + train_data.num_eq:train_data.num_var + 2*train_data.num_ineq+train_data.num_eq,:]
                delta_zl = train_y[:,train_data.num_var + 2*train_data.num_ineq+train_data.num_eq:train_data.num_var + 2*train_data.num_ineq+train_data.num_eq+train_data.num_lb,:]
                delta_zu = train_y[:,train_data.num_var+2*train_data.num_ineq+train_data.num_eq+train_data.num_lb:,:]
                alpha_x, alpha_eta, alpha_s, alpha_zl, alpha_zu = calculate_step(train_data, train_x, delta_x, train_eta, delta_eta, train_s,
                                                                                 delta_s, train_zl, delta_zl, train_zu, delta_zu, train_lb,
                                                                                 train_ub, args.tau, args.use_line_search, device='cuda:0')

                #update primal and dual variables
                if (train_data.num_lb != 0) or (train_data.num_ub != 0):
                    train_x = (train_x+alpha_x*delta_x).detach()
                else:
                    if train_data.num_ineq != 0:
                        train_x = (train_x + alpha_s * delta_x).detach()
                    else:
                        train_x = (train_x + delta_x).detach()
                if train_data.num_ineq != 0:
                    train_eta = (train_eta+alpha_eta*delta_eta).detach()
                    train_s = (train_s+alpha_s*delta_s).detach()
                if train_data.num_eq != 0:
                    if (train_data.num_lb != 0) or (train_data.num_ub != 0):
                        train_lamb = (train_lamb+alpha_x*delta_lamb).detach()
                    else:
                        if train_data.num_ineq != 0:
                            train_lamb = (train_lamb + alpha_s * delta_lamb).detach()
                        else:
                            train_lamb = (train_lamb + delta_lamb).detach()
                if train_data.num_lb != 0:
                    train_zl = (train_zl+alpha_zl*delta_zl).detach()
                if train_data.num_ub != 0:
                    train_zu = (train_zu+alpha_zu*delta_zu).detach()

        train_end_time = time.time()

        train_obj = train_data.obj_fn(train_x, Q=train_Q, p=train_p).mean()
        if train_data.num_ineq != 0:
            train_ineq_vio_max = train_data.ineq_dist(train_x, G=train_G, c=train_c).max(dim=1).values.mean()
            train_ineq_vio_mean = train_data.ineq_dist(train_x, G=train_G, c=train_c).mean()
        if train_data.num_eq != 0:
            train_eq_vio_max = train_data.eq_dist(train_x, A=train_A, b=train_b).max(dim=1).values.mean()
            train_eq_vio_mean = train_data.eq_dist(train_x, A=train_A, b=train_b).mean()
        if train_data.num_lb != 0:
            train_lb_vio_max = train_data.lower_bound_dist(train_x, lb=train_lb).max(dim=1).values.mean()
            train_lb_vio_mean = train_data.lower_bound_dist(train_x, lb=train_lb).mean()
        if train_data.num_ub != 0:
            train_ub_vio_max = train_data.upper_bound_dist(train_x, ub=train_ub).max(dim=1).values.mean()
            train_ub_vio_mean = train_data.upper_bound_dist(train_x, ub=train_ub).mean()


        #validation
        model.eval()
        with torch.no_grad():

            if (val_data.num_lb != 0) and (val_data.num_ub != 0):
                val_x = (val_data.lb + val_data.ub) / 2
            elif (val_data.num_lb != 0) and (val_data.num_ub == 0):
                val_x = val_data.lb + torch.ones(size=val_data.lb.shape, device=args.device)
            elif (val_data.num_lb == 0) and (val_data.num_lb != 0):
                val_x = val_data.ub - torch.ones(size=val_data.ub.shape, device=args.device)
            else:
                val_x = torch.zeros((val_data.Q.shape[0], val_data.num_var, 1), device=args.device)

            if val_data.num_ineq != 0:
                val_eta = torch.ones((val_data.Q.shape[0], val_data.num_ineq, 1), device=args.device)
                val_s = torch.ones((val_data.Q.shape[0], val_data.num_ineq, 1), device=args.device)
            else:
                val_eta = None
                val_s = None
            if val_data.num_eq != 0:
                val_lamb = torch.zeros((val_data.Q.shape[0], val_data.num_eq, 1), device=args.device)
            else:
                val_lamb = None
            if val_data.num_lb != 0:
                val_zl = torch.ones((val_data.Q.shape[0], val_data.num_lb, 1), device=args.device)
            else:
                val_zl = None
            if val_data.num_ub != 0:
                val_zu = torch.ones((val_data.Q.shape[0], val_data.num_ub, 1), device=args.device)
            else:
                val_zu = None

            val_start_time = time.time()

            for t_out in range(args.outer_T):
                # kkt
                val_J, val_F, val_mu = val_data.cal_kkt(val_x, val_eta, val_s, val_lamb, val_zl, val_zu, args.sigma)
                # initialization
                init_y = torch.zeros((val_data.val_size, val_data.num_var+2*val_data.num_ineq+val_data.num_eq+val_data.num_lb+val_data.num_ub, 1), device=args.device)

                #preconditioning
                if args.precondi:
                    val_D_values, D_id = (torch.bmm(val_J.permute(0, 2, 1), val_J)).max(-1)
                    val_D_inverse = torch.diag_embed(torch.sqrt(1 / val_D_values))
                    val_J_0 = torch.bmm(val_J, val_D_inverse)
                else:
                    val_J_0 = val_J

                val_y, val_loss, _ = model(val_data, init_y, val_J_0, val_F)

                if args.precondi:
                    val_y = torch.bmm(val_D_inverse, val_y)

                # stepsize
                delta_x = val_y[:, :val_data.num_var, :]
                delta_eta = val_y[:, val_data.num_var:val_data.num_var + val_data.num_ineq, :]
                delta_lamb = val_y[:, val_data.num_var + val_data.num_ineq:val_data.num_var + val_data.num_ineq + val_data.num_eq, :]
                delta_s = val_y[:, val_data.num_var + val_data.num_ineq + val_data.num_eq:val_data.num_var + 2 * val_data.num_ineq + val_data.num_eq, :]
                delta_zl = val_y[:, val_data.num_var + 2 * val_data.num_ineq + val_data.num_eq:val_data.num_var + 2 * val_data.num_ineq + val_data.num_eq + val_data.num_lb, :]
                delta_zu = val_y[:, val_data.num_var + 2 * val_data.num_ineq + val_data.num_eq + val_data.num_lb:, :]
                alpha_x, alpha_eta, alpha_s, alpha_zl, alpha_zu = calculate_step(val_data, val_x, delta_x, val_eta, delta_eta,
                                                                                val_s, delta_s, val_zl, delta_zl, val_zu,
                                                                                delta_zu, val_data.lb, val_data.ub, args.tau,
                                                                                args.use_line_search, device='cuda:0')


                # update primal and dual variables
                if (val_data.num_lb != 0) or (val_data.num_ub != 0):
                    val_x = (val_x + alpha_x * delta_x).detach()
                else:
                    if val_data.num_ineq != 0:
                        val_x = (val_x + alpha_s * delta_x).detach()
                    else:
                        val_x = (val_x + delta_x).detach()
                if val_data.num_ineq != 0:
                    val_eta = (val_eta + alpha_eta * delta_eta).detach()
                    val_s = (val_s + alpha_s * delta_s).detach()
                if val_data.num_eq != 0:
                    if (val_data.num_lb != 0) or (val_data.num_ub != 0):
                        val_lamb = (val_lamb+alpha_x*delta_lamb).detach()
                    else:
                        if val_data.num_ineq != 0:
                            val_lamb = (val_lamb + alpha_s * delta_lamb).detach()
                        else:
                            val_lamb = (val_lamb + delta_lamb).detach()
                if val_data.num_lb != 0:
                    val_zl = (val_zl + alpha_zl * delta_zl).detach()
                if val_data.num_ub != 0:
                    val_zu = (val_zu + alpha_zu * delta_zu).detach()

            val_end_time = time.time()


            val_vios = []
            val_obj = val_data.obj_fn(val_x).mean()
            if val_data.num_ineq != 0:
                val_ineq_vio_max = val_data.ineq_dist(val_x).max(dim=1).values.mean()
                val_ineq_vio_mean = val_data.ineq_dist(val_x).mean()
                val_vios.append(val_ineq_vio_max.data.item())
            if val_data.num_eq != 0:
                val_eq_vio_max = val_data.eq_dist(val_x).max(dim=1).values.mean()
                val_eq_vio_mean = val_data.eq_dist(val_x).mean()
                val_vios.append(val_eq_vio_max.data.item())
            if val_data.num_lb != 0:
                val_lb_vio_max = val_data.lower_bound_dist(val_x).max(dim=1).values.mean()
                val_lb_vio_mean = val_data.lower_bound_dist(val_x).mean()
                val_vios.append(val_lb_vio_max.data.item())
            if val_data.num_ub != 0:
                val_ub_vio_max = val_data.upper_bound_dist(val_x).max(dim=1).values.mean()
                val_ub_vio_mean = val_data.upper_bound_dist(val_x).mean()
                val_vios.append(val_ub_vio_max.data.item())

        early_stop = stopper.step(val_obj.data.item(), model, args.eq_tol, *val_vios)
        print("Epoch : {} | Train_Obj : {:.3f} | Val_Obj : {:.3f} | Train_Time : {:.3f} | Val_Time : {:.3f} |".format(epoch, train_obj, val_obj, train_end_time - train_start_time, val_end_time - val_start_time))
        if val_data.num_ineq != 0:
            print("Epoch : {} | Train_Max_Ineq : {:.3f} | Train_Mean_Ineq : {:.3f} | Val_Max_Ineq : {:.3f} | Val_Mean_Ineq : {:.3f} |".format(epoch, train_ineq_vio_max, train_ineq_vio_mean, val_ineq_vio_max, val_ineq_vio_mean))
        if val_data.num_eq != 0:
            print("Epoch : {} | Train_Max_Eq : {:.3f} | Train_Mean_Eq : {:.3f} | Val_Max_Eq : {:.3f} | Val_Mean_Eq : {:.3f} |".format(epoch, train_eq_vio_max, train_eq_vio_mean, val_eq_vio_max, val_eq_vio_mean))
        if val_data.num_lb != 0:
            print("Epoch : {} | Train_Max_Lb : {:.3f} | Train_Mean_Lb : {:.3f} | Val_Max_Lb : {:.3f} | Val_Mean_Lb : {:.3f} |".format(epoch, train_lb_vio_max, train_lb_vio_mean, val_lb_vio_max, val_lb_vio_mean))
        if val_data.num_ub != 0:
            print("Epoch : {} | Train_Max_Ub : {:.3f} | Train_Mean_Ub : {:.3f} | Val_Max_Ub : {:.3f} | Val_Mean_Ub : {:.3f} |".format(epoch, train_ub_vio_max, train_ub_vio_mean, val_ub_vio_max, val_ub_vio_mean))

        if early_stop:
            break


elif args.test:
    load_path = save_path
    # training and validation datasets
    if args.prob_type == 'QP':
        test_data = optimizee(prob_type=args.prob_type, learning_type='test', num_var=args.num_var, num_ineq=args.num_ineq,
                             num_eq=args.num_eq, data_size=args.data_size)
    else:
        test_data = optimizee(prob_type=args.prob_type, learning_type='test', file_path=file_path)

    model.load_state_dict(torch.load(load_path))
    model.eval()
    with torch.no_grad():
        if (test_data.num_lb != 0) and (test_data.num_ub != 0):
            test_x = (test_data.lb + test_data.ub) / 2
        elif (test_data.num_lb != 0) and (test_data.num_ub == 0):
            test_x = test_data.lb + torch.ones(size=test_data.lb.shape, device=args.device)
        elif (test_data.num_lb == 0) and (test_data.num_lb != 0):
            test_x = test_data.ub - torch.ones(size=test_data.ub.shape, device=args.device)
        else:
            test_x = torch.zeros((test_data.Q.shape[0], test_data.num_var, 1), device=args.device)

        print('The number of variables: {}.'.format(test_data.num_var))
        if test_data.num_ineq != 0:
            test_eta = torch.ones((test_data.Q.shape[0], test_data.num_ineq, 1), device=args.device)
            test_s = torch.ones((test_data.Q.shape[0], test_data.num_ineq, 1), device=args.device)
            print('The number of inequalities: {}.'.format(test_data.num_ineq))
        else:
            test_eta = None
            test_s = None
        if test_data.num_eq != 0:
            test_lamb = torch.zeros((test_data.Q.shape[0], test_data.num_eq, 1), device=args.device)
            print('The number of equalities: {}.'.format(test_data.num_eq))
        else:
            test_lamb = None
        if test_data.num_lb != 0:
            test_zl = torch.ones((test_data.Q.shape[0], test_data.num_lb, 1), device=args.device)
            print('The number of lower bounds: {}.'.format(test_data.num_lb))
        else:
            test_zl = None
        if test_data.num_ub != 0:
            test_zu = torch.ones((test_data.Q.shape[0], test_data.num_ub, 1), device=args.device)
            print('The number of upper bounds: {}.'.format(test_data.num_ub))
        else:
            test_zu = None

        #save results
        test_losses = []
        test_objs = []
        HH_condis = []
        HH_precondis = []
        test_ineq_vio_maxs = []
        test_ineq_vio_means = []
        test_eq_vio_maxs = []
        test_eq_vio_means = []
        test_lb_vio_maxs = []
        test_lb_vio_means = []
        test_ub_vio_maxs = []
        test_ub_vio_means = []
        test_residual = []
        test_y_norms = []
        test_mus = []
        test_F0 = []
        total_time = 0.0
        for t_out in range(args.outer_T):
            start_time = time.time()
            # kkt
            test_J, test_F, test_mu = test_data.cal_kkt(test_x, test_eta, test_s, test_lamb, test_zl, test_zu, args.sigma)

            HH_condis.append(np.linalg.cond(np.array((test_J[0].T @ test_J[0]).cpu().numpy())))
            # initialization
            init_y = torch.zeros((test_data.val_size, test_data.num_var + 2 * test_data.num_ineq + test_data.num_eq + test_data.num_lb + test_data.num_ub, 1), device=args.device)

            #preconditioning
            if args.precondi:
                test_D_values, D_id = (torch.bmm(test_J.permute(0, 2, 1), test_J)).max(-1)
                test_D_inverse = torch.diag_embed(torch.sqrt(1 / test_D_values))
                test_J_0 = torch.bmm(test_J, test_D_inverse)
                HH_precondis.append(np.linalg.cond(np.array((test_J_0[0].T @ test_J_0[0]).cpu().numpy())))
            else:
                test_J_0 = test_J

            test_y, _, losses = model(test_data, init_y, test_J_0, test_F)

            if args.precondi:
                test_y = torch.bmm(test_D_inverse, test_y)

            test_losses.append(losses)

            delta_x = test_y[:, :test_data.num_var, :]
            delta_eta = test_y[:, test_data.num_var:test_data.num_var + test_data.num_ineq, :]
            delta_lamb = test_y[:, test_data.num_var + test_data.num_ineq:test_data.num_var + test_data.num_ineq + test_data.num_eq, :]
            delta_s = test_y[:, test_data.num_var + test_data.num_ineq + test_data.num_eq:test_data.num_var + 2 * test_data.num_ineq + test_data.num_eq, :]
            delta_zl = test_y[:, test_data.num_var + 2 * test_data.num_ineq + test_data.num_eq:test_data.num_var + 2 * test_data.num_ineq + test_data.num_eq + test_data.num_lb, :]
            delta_zu = test_y[:, test_data.num_var + 2 * test_data.num_ineq + test_data.num_eq + test_data.num_lb:, :]
            alpha_x, alpha_eta, alpha_s, alpha_zl, alpha_zu = calculate_step(test_data, test_x, delta_x, test_eta, delta_eta,
                                                                             test_s, delta_s, test_zl, delta_zl, test_zu,
                                                                             delta_zu, test_data.lb, test_data.ub, args.tau,
                                                                             args.use_line_search, device='cuda:0')

            # update primal and dual variables
            if (test_data.num_lb != 0) or (test_data.num_ub != 0):
                test_x = (test_x + alpha_x * delta_x).detach()
            else:
                if test_data.num_ineq != 0:
                    test_x = (test_x + alpha_s * delta_x).detach()
                else:
                    test_x = (test_x + delta_x).detach()
            if test_data.num_ineq != 0:
                test_eta = (test_eta + alpha_eta * delta_eta).detach()
                test_s = (test_s + alpha_s * delta_s).detach()
            if test_data.num_eq != 0:
                if (test_data.num_lb != 0) or (test_data.num_ub != 0):
                    test_lamb = (test_lamb + alpha_x * delta_lamb).detach()
                else:
                    if test_data.num_ineq != 0:
                        test_lamb = (test_lamb + alpha_s * delta_lamb).detach()
                    else:
                        test_lamb = (test_lamb + delta_lamb).detach()
            if test_data.num_lb != 0:
                test_zl = (test_zl + alpha_zl * delta_zl).detach()
            if test_data.num_ub != 0:
                test_zu = (test_zu + alpha_zu * delta_zu).detach()

            end_time = time.time()
            total_time += (end_time-start_time)

            test_residual.append(torch.linalg.vector_norm(torch.bmm(test_J, test_y) + test_F, dim=1).mean().detach().cpu().numpy())
            test_y_norms.append(torch.linalg.vector_norm(test_y, dim=1).mean().detach().cpu().numpy())
            test_mus.append(test_mu.mean().detach().cpu().numpy())
            test_F0.append(torch.linalg.vector_norm(test_data.F0(test_x, test_eta, test_s, test_lamb, test_zl, test_zu, args.sigma), dim=1).mean().detach().cpu().numpy())

            test_obj = test_data.obj_fn(test_x).mean()
            test_objs.append(test_obj.detach().cpu().numpy())
            if test_data.num_ineq != 0:
                test_ineq_vio_max = test_data.ineq_dist(test_x).max(dim=1).values.mean()
                test_ineq_vio_mean = test_data.ineq_dist(test_x).mean()
                test_ineq_vio_maxs.append(test_ineq_vio_max.detach().cpu().numpy())
                test_ineq_vio_means.append(test_ineq_vio_mean.detach().cpu().numpy())
            if test_data.num_eq != 0:
                test_eq_vio_max = test_data.eq_dist(test_x).max(dim=1).values.mean()
                test_eq_vio_mean = test_data.eq_dist(test_x).mean()
                test_eq_vio_maxs.append(test_eq_vio_max.detach().cpu().numpy())
                test_eq_vio_means.append(test_eq_vio_mean.detach().cpu().numpy())
            if test_data.num_lb != 0:
                test_lb_vio_max = test_data.lower_bound_dist(test_x).max(dim=1).values.mean()
                test_lb_vio_mean = test_data.lower_bound_dist(test_x).mean()
                test_lb_vio_maxs.append(test_lb_vio_max.detach().cpu().numpy())
                test_lb_vio_means.append(test_lb_vio_mean.detach().cpu().numpy())
            if test_data.num_ub != 0:
                test_ub_vio_max = test_data.upper_bound_dist(test_x).max(dim=1).values.mean()
                test_ub_vio_mean = test_data.upper_bound_dist(test_x).mean()
                test_ub_vio_maxs.append(test_ub_vio_max.detach().cpu().numpy())
                test_ub_vio_means.append(test_ub_vio_mean.detach().cpu().numpy())

            print("Iter: {} | Test_Obj : {:.3f} | Test_Time : {:.3f} |".format(t_out, test_obj, total_time))
            if test_data.num_ineq != 0:
                print("Test_Max_Ineq : {:.3f} | Test_Mean_Ineq : {:.3f} |".format(test_ineq_vio_max, test_ineq_vio_mean))
            if test_data.num_eq != 0:
                print("Test_Max_Eq : {:.3f} | Test_Mean_Eq : {:.3f} |".format(test_eq_vio_max, test_eq_vio_mean))
            if test_data.num_lb != 0:
                print("Test_Max_Lb : {:.3f} | Test_Mean_Lb : {:.3f} |".format(test_lb_vio_max, test_lb_vio_mean))
            if test_data.num_ub != 0:
                print("Test_Max_Ub : {:.3f} | Test_Mean_Ub : {:.3f} |".format(test_ub_vio_max, test_ub_vio_mean))

        if args.test_solver == 'ipopt':
            if (test_data.num_ineq != 0) and (test_data.num_eq != 0):
                init_g = torch.concat((test_eta, test_lamb), dim=1).squeeze(-1)
            elif (test_data.num_ineq != 0) and (test_data.num_eq == 0):
                init_g = test_eta.squeeze(-1)
            elif (test_data.num_ineq == 0) and (test_data.num_eq != 0):
                init_g = test_lamb.squeeze(-1)
            else:
                init_g = None

            if test_data.num_lb != 0:
                init_zl = test_zl
            else:
                init_zl = None

            if test_data.num_ub != 0:
                init_zu = test_zu
            else:
                init_zu = None

            sols, _, para_times, iters = test_data.opt_solve(solver_type='ipopt', initial_y=test_x)
            best_obj = test_data.obj_fn(torch.tensor(sols, device=args.device).unsqueeze(-1).float()).mean()
            print('Best objective value:', best_obj)
            print('Original Solver Time: {}'.format(para_times))
            print('Original Iters:', iters)

            sols, _, para_times, iters = test_data.opt_solve(solver_type='ipopt', initial_y=test_x, init_mu=test_mu,
                                                             init_g=init_g, init_zl=init_zl, init_zu=init_zu)
            print('Warm Start Solver Time: {}'.format(para_times))
            print('Warm Start Iters:', iters)
            print('Warm Start Model Time: {}'.format((total_time)/test_data.test_size))
            print('Warm Start Total Time: {}'.format((total_time)/test_data.test_size+para_times))
        else:
            sols, _, para_time, iters = test_data.opt_solve(solver_type=args.test_solver, initial_y=test_x)
            best_obj = test_data.obj_fn(torch.tensor(sols, device=args.device).unsqueeze(-1).float()).mean()
            print('Time: {}'.format((total_time) / test_data.test_size))
            print('Solver Time: {}'.format(para_time))

        if args.save_sol:
            if args.prob_type == 'Nonconvex_QP':
                results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                         args.mat_name,
                                                                                                         args.outer_T,
                                                                                                         args.inner_T))
            else:
                results_save_path = os.path.join(args.save_dir, model.name(), '{}_{}_{}_{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                                        args.num_var,
                                                                                                                        args.num_ineq,
                                                                                                                        args.num_eq,
                                                                                                                        args.outer_T,
                                                                                                                        args.inner_T))

            test_dict = {'time': (total_time),
                         'x': np.array(test_x.detach().cpu()),
                         'inner_losses': np.array(test_losses),
                         'HH_condis': np.array(HH_condis),
                         'HH_precondis':np.array(HH_precondis),
                         'objs': np.array(test_objs),
                         'best_obj': np.array(best_obj.detach().cpu()),
                         'ineq_vio_maxs': np.array(test_ineq_vio_maxs),
                         'ineq_vio_means': np.array(test_ineq_vio_means),
                         'eq_vio_maxs': np.array(test_eq_vio_maxs),
                         'eq_vio_means': np.array(test_eq_vio_means),
                         'residual': np.array(test_residual),
                         'y_norm': np.array(test_y_norms),
                         'mu': np.array(test_mus),
                         'F0': np.array(test_F0)}

            #save test results
            sio.savemat(results_save_path, test_dict)