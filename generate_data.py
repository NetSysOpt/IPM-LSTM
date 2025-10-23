import os
import sys
import numpy as np
import configargparse
import scipy.io as sio
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, type=str)

#optimizee settings
parser.add_argument('--mat_name', type=str, help='Imported mat file name.')
parser.add_argument('--num_var', type=int, help='Number of decision vars.')
parser.add_argument('--num_eq', type=int, help='Number of equality constraints.')
parser.add_argument('--num_ineq', type=int, help='Number of inequality constraints.')
parser.add_argument('--prob_type', type=str, help='Problem type.')
parser.add_argument('--data_size', type=int, help='The number of all instances.')



args, _ = parser.parse_known_args()

if args.prob_type == 'Convex_QP_RHS':
    """
    DC3: A LEARNING METHOD FOR OPTIMIZATION WITH HARD CONSTRAINTS
    """
    torch.set_default_dtype(torch.float64)
    num_ineq = args.num_ineq
    num_eqs = [int(10*(args.num_var/100)), int(30*(args.num_var/100)), int(50*(args.num_var/100)),
               int(70*(args.num_var/100)), int(90*(args.num_var/100))]
    for num_eq in num_eqs:
        mat_name = "dc3_{}_{}_{}_{}".format(args.num_var, num_ineq, num_eq, args.data_size)
        file_path = os.path.join('datasets', 'qp', "{}".format(mat_name))
        np.random.seed(17)
        Q = np.diag(np.random.random(args.num_var))
        p = np.random.random(args.num_var)
        A = np.random.normal(loc=0, scale=1., size=(num_eq, args.num_var))
        X = np.random.uniform(-1, 1, size=(args.data_size, num_eq))
        G = np.random.normal(loc=0, scale=1., size=(num_ineq, args.num_var))
        h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)

        sio.savemat(file_path,
            {'Q': np.repeat(np.expand_dims(Q, axis=0), args.data_size, axis=0),
             'p': np.repeat(np.expand_dims(p, axis=0), args.data_size, axis=0),
             'A': np.repeat(np.expand_dims(A, axis=0), args.data_size, axis=0),
             'X': np.expand_dims(X, axis=-1),
             'G': np.repeat(np.expand_dims(G, axis=0), args.data_size, axis=0),
             'h': np.expand_dims(h, axis=-1)})


    num_eq = args.num_eq
    num_ineqs = [int(10*(args.num_var/100)), int(30*(args.num_var/100)),
                 int(70*(args.num_var/100)), int(90*(args.num_var/100))]
    for num_ineq in num_ineqs:
        mat_name = "dc3_{}_{}_{}_{}".format(args.num_var, num_ineq, num_eq, args.data_size)
        file_path = os.path.join('datasets', 'qp', "{}.mat".format(mat_name))
        np.random.seed(17)
        Q = np.diag(np.random.random(args.num_var))
        p = np.random.random(args.num_var)
        A = np.random.normal(loc=0, scale=1., size=(num_eq, args.num_var))
        X = np.random.uniform(-1, 1, size=(args.data_size, num_eq))
        G = np.random.normal(loc=0, scale=1., size=(num_ineq, args.num_var))
        h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)

        sio.savemat(file_path,
                {'Q': np.repeat(np.expand_dims(Q, axis=0), args.data_size, axis=0),
                     'p': np.repeat(np.expand_dims(p, axis=0), args.data_size, axis=0),
                     'A': np.repeat(np.expand_dims(A, axis=0), args.data_size, axis=0),
                     'X': np.expand_dims(X, axis=-1),
                     'G': np.repeat(np.expand_dims(G, axis=0), args.data_size, axis=0),
                     'h': np.expand_dims(h, axis=-1)})

elif args.prob_type == 'Nonconvex_Program_RHS':
    """
    DC3: A LEARNING METHOD FOR OPTIMIZATION WITH HARD CONSTRAINTS
    """
    torch.set_default_dtype(torch.float64)

    num_var = 100
    num_ineq = 50
    num_eq = 50
    num_examples = 10000

    np.random.seed(17)

    Q = np.diag(np.random.random(num_var))
    p = np.random.random(num_var)
    A = np.random.normal(loc=0, scale=1., size=(num_eq, num_var))
    X = np.random.uniform(-1, 1, size=(num_examples, num_eq))
    G = np.random.normal(loc=0, scale=1., size=(num_ineq, num_var))
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)

    mat_name = "random_nonconvex_dataset_var{}_ineq{}_eq{}_ex{}".format(args.num_var, args.num_ineq, args.num_eq, args.data_size)
    file_path = os.path.join('datasets', 'nonconvex_program', "{}.mat".format(mat_name))

    sio.savemat(file_path,
                {'Q': np.repeat(np.expand_dims(Q, axis=0), args.data_size, axis=0),
                 'p': np.repeat(np.expand_dims(p, axis=0), args.data_size, axis=0),
                 'A': np.repeat(np.expand_dims(A, axis=0), args.data_size, axis=0),
                 'b': np.expand_dims(X, axis=-1),
                 'G': np.repeat(np.expand_dims(G, axis=0), args.data_size, axis=0),
                 'c': np.expand_dims(h, axis=-1)})

elif args.prob_type == 'Convex_QCQP_RHS':
    """
    DC3: A LEARNING METHOD FOR OPTIMIZATION WITH HARD CONSTRAINTS
    """
    np.random.seed(2023)
    Q = np.diag(np.random.rand(args.num_var) * 0.5)
    p = np.random.uniform(-1, 1, args.num_var)
    A = np.random.uniform(-1, 1, size=(args.num_eq, args.num_var))
    X = np.random.uniform(-0.5, 0.5, size=(args.data_size, args.num_eq))
    G = np.random.uniform(-1, 1, size=(args.num_ineq, args.num_var))
    h = np.sum(np.abs(G @ np.linalg.pinv(A)), axis=1)
    H = np.random.uniform(0, 0.1, size=(args.num_ineq, args.num_var))
    H = [np.diag(H[i]) for i in range(args.num_ineq)]
    H = np.array(H)
    data = {'Q': Q,
            'p': p,
            'A': A,
            'X': X,
            'G': G,
            'H': H,
            'h': h,
            'Y': []}
    mat_name = "random_convex_qcqp_dataset_var{}_ineq{}_eq{}_ex{}".format(args.num_var,
                                                                          args.num_ineq,
                                                                          args.num_eq,
                                                                          args.data_size)
    file_path = os.path.join('datasets', 'convex_qcqp', "{}.mat".format(mat_name))
    sio.savemat(file_path, data)
elif args.prob_type == 'Nonconvex_QP':
    """
    Globally solving nonconvex quadratic programming problems via completely positive programming
    """
    if (args.mat_name == 'qp1') or (args.mat_name == 'qp2'):
        load_path = os.path.join('datasets', 'qp', 'nonconvex_qp', '{}.mat'.format(args.mat_name))
        data = sio.loadmat(load_path)
        np.random.seed(17)
        Q = []
        p = []
        G = []
        c = []
        A = []
        b = []
        lb = []

        Q0 = data['H'].copy()
        p0 = data['f'].copy()
        G0 = data['A'].copy()
        c0 = data['b'].copy()
        A0 = data['Aeq'].copy()
        b0 = data['beq'].copy()
        lb0 = data['LB'].copy()

        for i in range(args.data_size):
            nonzero_indices = np.nonzero(data['H'])
            nonzero_values = data['H'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            Q0[nonzero_indices] = perturbed_values

            nonzero_indices = np.nonzero(data['A'])
            nonzero_values = data['A'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            G0[nonzero_indices] = perturbed_values

            nonzero_indices = np.nonzero(data['b'])
            nonzero_values = data['b'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            c0[nonzero_indices] = perturbed_values

            Q.append(Q0)
            p.append(p0)
            G.append(G0)
            c.append(c0)
            A.append(A0)
            b.append(b0)
            lb.append(lb0)

        glob_dict = {'Q': np.array(Q), 'p': np.array(p),
                     'G': np.array(G), 'c': np.array(c),
                     'A':np.array(A), 'b':np.array(b),
                     'lb': np.array(lb)}
        file_path = os.path.join('datasets', 'qp', "{}.mat".format(args.mat_name))
        sio.savemat(file_path, glob_dict)

    elif (args.mat_name == 'st_rv1') or (args.mat_name == 'st_rv2') or (args.mat_name == 'st_rv3') or (args.mat_name == 'st_rv7') or (args.mat_name == 'st_rv9'):
        load_path = os.path.join('datasets', 'qp', 'nonconvex_qp', '{}.mat'.format(args.mat_name))
        data = sio.loadmat(load_path)
        np.random.seed(17)
        Q = []
        p = []
        G = []
        c = []
        lb = []

        Q0 = data['H'].copy()
        p0 = data['f'].copy()
        G0 = data['A'].copy()
        c0 = data['b'].copy()
        lb0 = data['LB'].copy()

        for i in range(args.data_size):
            nonzero_indices = np.nonzero(data['H'])
            nonzero_values = data['H'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            Q0[nonzero_indices] = perturbed_values

            nonzero_indices = np.nonzero(data['f'])
            nonzero_values = data['f'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            p0[nonzero_indices] = perturbed_values

            nonzero_indices = np.nonzero(data['A'])
            nonzero_values = data['A'][nonzero_indices]
            perturbed_values = np.round(nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape)))
            G0[nonzero_indices] = perturbed_values

            nonzero_indices = np.nonzero(data['b'])
            nonzero_values = data['b'][nonzero_indices]
            perturbed_values = np.round(nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape)))
            c0[nonzero_indices] = perturbed_values

            Q.append(Q0)
            p.append(p0)
            G.append(G0)
            c.append(c0)
            lb.append(lb0)

        glob_dict = {'Q': np.array(Q), 'p': np.array(p),
                     'G': np.array(G), 'c': np.array(c),
                     'lb': np.array(lb)}
        file_path = os.path.join('datasets', 'qp', "{}.mat".format(args.mat_name))
        sio.savemat(file_path, glob_dict)

    elif args.mat_name == 'qp30_15_1_1':
        load_path = os.path.join('datasets/qp/nonconvex_qp', '{}.mat'.format(args.mat_name))
        data = sio.loadmat(load_path)
        np.random.seed(17)
        Q = []
        p = []
        G = []
        c = []
        A = []
        b = []
        lb = []
        ub = []

        Q0 = data['H'].copy()
        p0 = data['f'].copy()
        G0 = data['A'].copy()
        c0 = data['b'].copy()
        A0 = data['Aeq'].copy()
        b0 = data['beq'].copy()
        lb0 = data['LB'].copy()
        ub0 = data['UB'].copy()

        for i in range(args.data_size):
            nonzero_indices = np.nonzero(data['H'])
            nonzero_values = data['H'][nonzero_indices]
            perturbed_values = np.round(nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape)))
            Q0[nonzero_indices] = perturbed_values
            Q0 = (Q0 + Q0.T) / 2

            nonzero_indices = np.nonzero(data['f'])
            nonzero_values = data['f'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            p0[nonzero_indices] = perturbed_values

            nonzero_indices = np.nonzero(data['A'])
            nonzero_values = data['A'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            G0[nonzero_indices] = perturbed_values

            nonzero_indices = np.nonzero(data['b'])
            nonzero_values = data['b'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            c0[nonzero_indices] = perturbed_values

            nonzero_indices = np.nonzero(data['Aeq'])
            nonzero_values = data['Aeq'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            A0[nonzero_indices] = perturbed_values

            nonzero_indices = np.nonzero(data['beq'])
            nonzero_values = data['beq'][nonzero_indices]
            perturbed_values = nonzero_values * (1 + np.random.uniform(-0.2, 0.2, size=nonzero_values.shape))
            b0[nonzero_indices] = perturbed_values

            Q.append(Q0)
            p.append(p0)
            G.append(G0)
            c.append(c0)
            A.append(A0)
            b.append(b0)
            lb.append(lb0)
            ub.append(ub0)

        rand_dict = {'Q': np.array(Q), 'p': np.array(p),
                     'G': np.array(G), 'c': np.array(c),
                     'A': np.array(A), 'b': np.array(b),
                     'lb': np.array(lb)}
        file_path = os.path.join('datasets', 'qp', "{}.mat".format(args.mat_name))
        sio.savemat(file_path, rand_dict)















