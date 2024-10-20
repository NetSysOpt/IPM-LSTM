import os
import sys
import numpy as np
import configargparse
import matplotlib.pyplot as plt
import scipy.io as sio

from utils import plot_objective_values, plot_inner_losses, plot_condtions

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, type=str)


parser.add_argument('--plot_type', type=str, help='Plot type needed.')
parser.add_argument('--mat_name', type=str, help='Imported mat file name.')
parser.add_argument('--num_var', type=int, help='Number of decision vars.')
parser.add_argument('--num_eq', type=int, help='Number of equality constraints.')
parser.add_argument('--num_ineq', type=int, help='Number of inequality constraints.')
parser.add_argument('--prob_type', type=str, help='Problem type.')
parser.add_argument('--model_name', type=str, help='The deep learning optimizer name.')
parser.add_argument('--inner_T', type=int, help='The iterations of deep learning optimizer.')
parser.add_argument('--outer_T', type=int, help='The iterations of IPM.')
parser.add_argument('--save_dir', type=str, default='./results/', help='Save path for the best model.')
parser.add_argument('--eta', type=float, help='The coefficient of line search.')
parser.add_argument('--sigma', type=float, help='The coefficient of mu.')



args, _ = parser.parse_known_args()

if args.prob_type == 'Nonconvex_QP':
    results_save_path = os.path.join(args.save_dir, args.model_name, '{}_{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                   args.mat_name,
                                                                                                   args.outer_T,
                                                                                                   args.inner_T))
else:
    results_save_path = os.path.join(args.save_dir, args.model_name, '{}_{}_{}_{}_{}_{}_results.mat'.format(args.prob_type,
                                                                                                         args.num_var,
                                                                                                         args.num_ineq,
                                                                                                         args.num_eq,
                                                                                                         args.outer_T,
                                                                                                         args.inner_T))


#load data
data = sio.loadmat(results_save_path)

if args.plot_type == 'Objective_values':
    if args.prob_type == 'Nonconvex_QP':
        obj_file = 'Obj_{}_{}_{}_{}.eps'.format(args.prob_type, args.mat_name, args.outer_T, args.inner_T)
    else:
        obj_file = 'Obj_{}_{}_{}_{}_{}_{}.eps'.format(args.prob_type, args.num_var, args.num_eq, args.num_ineq, args.outer_T, args.inner_T)
    obj_save_path = os.path.join('./results', args.model_name, 'figs', obj_file)
    obj_values = data['objs'].squeeze()
    best_obj = data['best_obj']
    plot_objective_values(obj_values, args.outer_T, obj_save_path, 'IPM-LSTM', best_obj)
elif args.plot_type == 'Residual':
    time_steps = [0, 1, 2, 9, 19, 59, 99]
    if args.prob_type == 'Nonconvex_QP':
        il_file = 'Obj_{}_{}_{}_{}.eps'.format(args.prob_type, args.mat_name, args.outer_T, args.inner_T)
    else:
        il_file = 'Obj_{}_{}_{}_{}_{}_{}.eps'.format(args.prob_type, args.num_var, args.num_eq, args.num_ineq, args.outer_T, args.inner_T)

    il_save_path = os.path.join('./results', args.model_name, 'figs', il_file)
    plot_inner_losses(data, args.inner_T, time_steps, il_save_path)
elif args.plot_type == 'Condition1':
    if args.prob_type == 'Nonconvex_QP':
        cond1_file = 'Cond1_{}_{}_{}_{}.eps'.format(args.prob_type, args.mat_name, args.outer_T, args.inner_T)
    else:
        cond1_file = 'Cond1_{}_{}_{}_{}_{}_{}.eps'.format(args.prob_type, args.num_var, args.num_eq, args.num_ineq, args.outer_T, args.inner_T)
    cond1_save_path = os.path.join('./results', args.model_name, 'figs', cond1_file)
    plot_condtions(data, args.outer_T, cond1_save_path, args.eta, args.sigma, cond="Condition1")
elif args.plot_type == 'Condition2':
    if args.prob_type == 'Nonconvex_QP':
        cond2_file = 'Cond2_{}_{}_{}_{}.eps'.format(args.prob_type, args.mat_name, args.outer_T, args.inner_T)
    else:
        cond2_file = 'Cond2_{}_{}_{}_{}_{}_{}.eps'.format(args.prob_type, args.num_var, args.num_eq, args.num_ineq, args.outer_T, args.inner_T)
    cond2_save_path = os.path.join('./results', args.model_name, 'figs', cond2_file)
    plot_condtions(data, args.outer_T, cond2_save_path, args.eta, args.sigma, cond="Condition2")

