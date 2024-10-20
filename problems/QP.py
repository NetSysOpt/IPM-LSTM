import os
import osqp
import time
import torch
import numpy as np
import scipy.io as sio
import cyipopt as ipopt

from qpth.qp import QPFunction
from scipy.sparse import csc_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class QP(object):
    """
        minimize_x 0.5*x^T Q x + p^Tx
        s.t.       Gx <= c
                   Ax = b

        Q: [batch_size, num_var, num_var]
        p: [batch_size, num_var, 1]
        G: [batch_size, num_ineq, num_var]
        c: [batch_size, num_ineq, 1]
        A: [batch_size, num_eq, num_var]
        b: [batch_size, num_eq, 1]
    """
    def __init__(self, prob_type, learning_type, val_frac=0.0833, test_frac=0.0833, device='cuda:0', seed=17, **kwargs):
        super().__init__()

        self.device = device
        self.seed = seed
        self.learning_type = learning_type
        self.train_frac = 1 - val_frac - test_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.prob_type = prob_type
        torch.manual_seed(self.seed)

        if prob_type == 'QP_RHS':
            file_path = kwargs['file_path']
            data = sio.loadmat(file_path)
            self.data_size = data['X'].shape[0]

            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size

            self.num_var = data['Q'].shape[0]
            self.num_ineq = data['G'].shape[0]
            self.num_eq = data['A'].shape[0]
            self.num_lb = 0
            self.num_ub = 0

            if learning_type == 'train':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.train_size, 1).unsqueeze(-1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[:self.train_size].unsqueeze(-1)
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.train_size, 1).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.val_size, 1).unsqueeze(-1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size:self.train_size + self.val_size].unsqueeze(-1)
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.val_size, 1).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.test_size, 1).unsqueeze(-1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size + self.val_size:].unsqueeze(-1)
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.test_size, 1).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf

        elif prob_type == 'QP':
            self.data_size = kwargs['data_size']
            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size

            self.num_var = kwargs['num_var']
            self.num_ineq = kwargs['num_ineq']
            self.num_eq = kwargs['num_eq']
            self.num_lb = 0
            self.num_ub = 0

            if learning_type == 'train':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[:self.train_size]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[:self.train_size].unsqueeze(-1)
                self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[:self.train_size]
                self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[:self.train_size].unsqueeze(-1) - 1  # [-1, 1]
                self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[:self.train_size]
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size:self.train_size + self.val_size]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size:self.train_size + self.val_size].unsqueeze(-1)
                self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
                self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[self.train_size:self.train_size + self.val_size].unsqueeze(-1) - 1  # [-1, 1]
                self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size + self.val_size:]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size + self.val_size:].unsqueeze(-1)
                self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[self.train_size + self.val_size:]
                self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[self.train_size + self.val_size:].unsqueeze(-1) - 1  # [-1, 1]
                self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size + self.val_size:]
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf

        else:
            file_path = kwargs['file_path']
            data = sio.loadmat(file_path)
            self.data_size = data['Q'].shape[0]

            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size

            self.num_var = data['Q'].shape[1]
            try:
                self.num_ineq = data['G'].shape[1]
            except KeyError:
                self.num_ineq = 0

            try:
                self.num_eq = data['A'].shape[1]
            except KeyError:
                self.num_eq = 0

            try:
                self.num_lb = data['lb'].shape[1]
            except KeyError:
                self.num_lb = 0

            try:
                self.num_ub = data['ub'].shape[1]
            except KeyError:
                self.num_ub = 0

            if learning_type == 'train':
                self.Q = torch.tensor(data['Q'], device=self.device).float()[:self.train_size]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[:self.train_size]
                if self.num_eq != 0:
                    self.A = torch.tensor(data['A'], device=self.device).float()[:self.train_size]
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[:self.train_size]
                if self.num_ineq != 0:
                    self.G = torch.tensor(data['G'], device=self.device).float()[:self.train_size]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[:self.train_size]
                if self.num_lb != 0:
                    self.lb = torch.tensor(data['lb'], device=self.device).float()[:self.train_size]
                else:
                    self.lb = -torch.inf
                if self.num_ub != 0:
                    self.ub = torch.tensor(data['ub'], device=self.device).float()[:self.train_size]
                else:
                    self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.tensor(data['Q'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
                if self.num_eq != 0:
                    self.A = torch.tensor(data['A'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
                if self.num_ineq != 0:
                    self.G = torch.tensor(data['G'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
                if self.num_lb != 0:
                    self.lb = torch.tensor(data['lb'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                else:
                    self.lb = -torch.inf
                if self.num_ub != 0:
                    self.ub = torch.tensor(data['ub'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                else:
                    self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.tensor(data['Q'], device=self.device).float()[self.train_size + self.val_size:]
                self.p = torch.tensor(data['p'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:]
                if self.num_eq != 0:
                    self.A = torch.tensor(data['A'], device=self.device).float()[self.train_size + self.val_size:]
                    self.b = torch.tensor(data['b'].astype(np.float32), device=self.device).float()[self.train_size + self.val_size:]
                if self.num_ineq != 0:
                    self.G = torch.tensor(data['G'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                    self.c = torch.tensor(data['c'].astype(np.float32), device=self.device).float()[self.train_size:self.train_size + self.val_size]
                if self.num_lb != 0:
                    self.lb = torch.tensor(data['lb'], device=self.device).float()[self.train_size + self.val_size:]
                else:
                    self.lb = -torch.inf
                if self.num_ub != 0:
                    self.ub = torch.tensor(data['ub'], device=self.device).float()[self.train_size + self.val_size:]
                else:
                    self.ub = torch.inf


    def name(self):
        str = '{}_{}_{}_{}_{}'.format(self.prob_type, self.num_ineq, self.num_eq, self.num_lb, self.num_ub)
        return str


    def obj_fn(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return 0.5 * torch.bmm(x.permute(0, 2, 1), torch.bmm(Q, x)) + torch.bmm(p.permute(0, 2, 1), x)

    def obj_grad(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p

    def ineq_resid(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.bmm(G, x) - c

    def ineq_dist(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.clamp(self.ineq_resid(x, G=G, c=c), 0)

    def eq_resid(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.bmm(A, x) - b

    def eq_dist(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.abs(self.eq_resid(x, A=A, b=b))

    def lower_bound_dist(self, x, **kwargs):
        lb = kwargs.get('lb', self.lb)
        return torch.clamp(lb - x, 0)

    def upper_bound_dist(self, x, **kwargs):
        ub = kwargs.get('ub', self.ub)
        return torch.clamp(x-ub, 0)

    def F0(self, x, eta, s, lamb, zl, zu, sigma, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        if self.num_ineq != 0:
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)

        # residual
        F_list = []
        F1 = torch.bmm(0.5 * (Q + Q.permute(0, 2, 1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(G.permute(0, 2, 1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(A.permute(0, 2, 1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = torch.bmm(G, x) - c + s
            F3 = eta * s
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = torch.bmm(A, x) - b
            F_list.append(F4)

        if self.num_lb != 0:
            F5 = zl * (x - lb)
            F_list.append(F5)

        if self.num_ub != 0:
            F6 = zu * (ub - x)
            F_list.append(F6)

        F = torch.concat(F_list, dim=1)
        return F


    def cal_kkt(self, x, eta, s, lamb, zl, zu, sigma, **kwargs):
        """
        x: [batch_size, num_var, 1]
        eta: [batch_size, num_ineq, 1]
        lamb: [batch_size, num_eq, 1]
        s: [batch_size, num_ineq, 1]
        zl: [batch_size, num_lb, 1]
        zu: [batch_size, num_ub, 1]

        return:
        J: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        F: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        mu: [batch_size, 1, 1]
        """
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        mu = 0
        if self.num_ineq != 0:
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
            mu += sigma * ((eta * s).sum(1).unsqueeze(-1))
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
            mu += sigma * ((zl * (x-lb)).sum(1).unsqueeze(-1))
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
            mu += sigma * ((zu * (ub-x)).sum(1).unsqueeze(-1))
        batch_size = Q.shape[0]
        # mu
        mu = mu/(self.num_ineq+self.num_lb+self.num_ub)

        # residual
        F_list = []
        F1 = torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(G.permute(0, 2, 1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(A.permute(0, 2, 1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = torch.bmm(G, x) - c + s
            F3 = eta * s - mu
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = torch.bmm(A, x) - b
            F_list.append(F4)

        if self.num_lb != 0:
            F5 = zl*(x-lb) - mu
            F_list.append(F5)

        if self.num_ub != 0:
            F6 = zu*(ub-x) - mu
            F_list.append(F6)

        F = torch.concat(F_list, dim=1)

        # jacobian of residual
        J_list = []
        J1 = 0.5*(Q+Q.permute(0,2,1))
        if self.num_ineq != 0:
            J1 = torch.concat((J1, G.permute(0,2,1)), dim=2)
        if self.num_eq != 0:
            J1 = torch.concat((J1, A.permute(0,2,1)), dim=2)
        if self.num_ineq != 0:
            J1 = torch.concat((J1, torch.zeros(size=(batch_size, self.num_var, self.num_ineq), device=self.device)), dim=2)
        if self.num_lb != 0:
            J1 = torch.concat((J1, -torch.diag_embed(torch.ones(size=(batch_size, self.num_lb), device=self.device))), dim=2)
        if self.num_ub != 0:
            J1 = torch.concat((J1, torch.diag_embed(torch.ones(size=(batch_size, self.num_ub), device=self.device))), dim=2)
        J_list.append(J1)

        if self.num_ineq != 0:
            J2 = torch.concat((G, torch.zeros(size=(batch_size, self.num_ineq, self.num_ineq), device=self.device)), dim=2)
            if self.num_eq != 0:
                J2 = torch.concat((J2, torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device)), dim=2)
            J2 = torch.concat((J2, torch.diag_embed(torch.ones(size=(batch_size, self.num_ineq), device=self.device))), dim=2)
            if self.num_lb != 0:
                J2 = torch.concat((J2, torch.zeros(size=(batch_size, self.num_ineq, self.num_lb), device=self.device)), dim=2)
            if self.num_ub != 0:
                J2 = torch.concat((J2, torch.zeros(size=(batch_size, self.num_ineq, self.num_ub), device=self.device)), dim=2)
            J_list.append(J2)

            J3 = torch.zeros(size=(batch_size, self.num_ineq, self.num_var), device=self.device)
            J3 = torch.concat((J3, torch.diag_embed(s.squeeze(-1))), dim=2)
            if self.num_eq != 0:
                J3 = torch.concat((J3, torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device)), dim=2)
            J3 = torch.concat((J3, torch.diag_embed(eta.squeeze(-1))), dim=2)
            if self.num_lb != 0:
                J3 = torch.concat((J3, torch.zeros(size=(batch_size, self.num_ineq, self.num_lb), device=self.device)), dim=2)
            if self.num_ub != 0:
                J3 = torch.concat((J3, torch.zeros(size=(batch_size, self.num_ineq, self.num_ub), device=self.device)), dim=2)
            J_list.append(J3)

        if self.num_eq != 0:
            J4 = A
            if self.num_ineq != 0:
                J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_ineq), device=self.device)), dim=2)
            J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_eq), device=self.device)), dim=2)
            if self.num_ineq != 0:
                J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_ineq), device=self.device)), dim=2)
            if self.num_lb != 0:
                J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_lb), device=self.device)), dim=2)
            if self.num_ub != 0:
                J4 = torch.concat((J4, torch.zeros(size=(batch_size, self.num_eq, self.num_ub), device=self.device)), dim=2)
            J_list.append(J4)

        if self.num_lb != 0:
            J5 = torch.diag_embed(zl.squeeze(-1))
            if self.num_ineq != 0:
                J5 = torch.concat((J5, torch.zeros(size=(batch_size, self.num_lb, self.num_ineq), device=self.device)), dim=2)
            if self.num_eq != 0:
                J5 = torch.concat((J5, torch.zeros(size=(batch_size, self.num_lb, self.num_eq), device=self.device)), dim=2)
            if self.num_ineq != 0:
                J5 = torch.concat((J5, torch.zeros(size=(batch_size, self.num_lb, self.num_ineq), device=self.device)), dim=2)
            J5 = torch.concat((J5, torch.diag_embed((x-lb).squeeze(-1))), dim=2)
            if self.num_ub != 0:
                J5 = torch.concat((J5, torch.zeros(size=(batch_size, self.num_lb, self.num_ub), device=self.device)), dim=2)
            J_list.append(J5)

        if self.num_ub != 0:
            J6 = -torch.diag_embed(zu.squeeze(-1))
            if self.num_ineq != 0:
                J6 = torch.concat((J6, torch.zeros(size=(batch_size, self.num_ub, self.num_ineq), device=self.device)), dim=2)
            if self.num_eq != 0:
                J6 = torch.concat((J6, torch.zeros(size=(batch_size, self.num_ub, self.num_eq), device=self.device)), dim=2)
            if self.num_ineq != 0:
                J6 = torch.concat((J6, torch.zeros(size=(batch_size, self.num_ub, self.num_ineq), device=self.device)), dim=2)
            if self.num_lb != 0:
                J6 = torch.concat((J6, torch.zeros(size=(batch_size, self.num_ub, self.num_lb), device=self.device)), dim=2)
            J6 = torch.concat((J6, torch.diag_embed((ub-x).squeeze(-1))), dim=2)
            J_list.append(J6)

        J = torch.concat(J_list, dim=1)
        return J, F, mu

    def sub_objective(self, y, J, F):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        J: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        F: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        1/2||J@y-F||_2^2 = 1/2(y^T@J^T@Jy)-y^TJ^TF+1/2(F^TF)
        """
        obj0 = 0.5 * torch.bmm(torch.bmm(y.permute(0, 2, 1), J.permute(0, 2, 1)), torch.bmm(J, y))
        obj1 = torch.bmm(torch.bmm(y.permute(0, 2, 1), J.permute(0, 2, 1)), F)
        obj2 = 0.5 * (torch.bmm(F.permute(0, 2, 1), F))
        return obj0+obj1+obj2

    def sub_smooth_grad(self, y, J, F):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        return H^T@H@delta_r+H^T@r
        """
        grad = torch.bmm(torch.bmm(J.permute(0, 2, 1), J), y) + torch.bmm(J.permute(0, 2, 1), F)
        return grad

    def opt_solve(self, solver_type='osqp', tol=1e-4, initial_y = None, init_mu=None, init_g=None, init_zl=None, init_zu=None):
        if solver_type == 'osqp':
            print('running osqp')
            Q, p = self.Q.detach().cpu().numpy(), self.p.detach().cpu().numpy()
            if self.num_ineq != 0:
                G, c = self.G.detach().cpu().numpy(), self.c.detach().cpu().numpy()
            if self.num_eq != 0:
                A, b = self.A.detach().cpu().numpy(), self.b.detach().cpu().numpy()
            if self.num_lb != 0:
                lb = self.lb.detach().cpu().numpy()
            if self.num_ub != 0:
                ub = self.ub.detach().cpu().numpy()

            s = []
            iters = 0
            total_time = 0
            for i in range(Q.shape[0]):
                solver = osqp.OSQP()
                A0 = []
                zl = []
                zu = []
                if self.num_ineq != 0:
                    A0.append(G[i, :, :])
                    zl.append(-np.ones(c.shape[1]) * np.inf)
                    zu.append(c[i, :])
                if self.num_eq != 0:
                    A0.append(A[i, :, :])
                    zl.append(b[i, :])
                    zu.append(b[i, :])
                if self.num_lb != 0:
                    A0.append(np.eye(p.shape[1]))
                    zl.append(lb[i, :])
                else:
                    A0.append(np.eye(p.shape[1]))
                    zl.append(-np.ones(p.shape[1])*np.inf)
                if self.num_ub != 0:
                    zu.append(ub[i, :])
                else:
                    zu.append(ub[i, :])

                my_A = np.vstack(A0)
                my_l = np.hstack(zl)
                my_u = np.hstack(zu)
                solver.setup(P=csc_matrix(Q[i, :, :]), q=p[i, :], A=csc_matrix(my_A),
                             l=my_l, u=my_u, verbose=False, eps_prim_inf=tol)
                start_time = time.time()
                results = solver.solve()
                end_time = time.time()

                total_time += (end_time - start_time)
                if results.info.status == 'solved':
                    s.append(results.x)
                else:
                    s.append(np.ones(self.num_var) * np.nan)
                    print('Batch {} optimization failed.'.format(i))

            sols = np.array(s)
            parallel_time = total_time / Q.shape[0]

        elif solver_type == 'ipopt':
            Q, p = self.Q.detach().cpu().numpy(), self.p.detach().cpu().numpy()
            if self.num_ineq != 0:
                G, c = self.G.detach().cpu().numpy(), self.c.detach().cpu().numpy()
            if self.num_eq != 0:
                A, b = self.A.detach().cpu().numpy(), self.b.detach().cpu().numpy()
            if self.num_lb != 0:
                lb = self.lb.detach().cpu().numpy()
            else:
                lb = -np.infty * np.ones(shape=(Q.shape[0], Q.shape[1], 1))
            if self.num_ub != 0:
                ub = self.ub.detach().cpu().numpy()
            else:
                ub = np.infty * np.ones(shape=(Q.shape[0], Q.shape[1], 1))

            Y = []
            iters = []
            total_time = 0
            for i in range(Q.shape[0]):
                if initial_y is None:
                    # y0 = np.linalg.pinv(A[i]) @ b[i]  # feasible initial point
                    if (self.num_lb != 0) and (self.num_ub != 0):
                        y0 = ((lb[i]+ub[i])/2).squeeze(-1)
                    elif (self.num_lb != 0) and (self.num_ub == 0):
                        y0 = (lb[i] + np.ones(shape=lb[i].shape)).squeeze(-1)
                    elif (self.num_lb == 0) and (self.num_lb != 0):
                        y0 = (ub[i] - np.ones(shape=ub[i].shape)).squeeze(-1)
                    else:
                        y0 = np.zeros(self.num_var)
                else:
                    y0 = initial_y[i].cpu().numpy()

                # upper and lower bounds on constraints
                cls = []
                cus = []
                if self.num_ineq != 0:
                    cls.append(-np.inf * np.ones(G[i].shape[0]))
                    cus.append(c[i].squeeze(-1))
                if self.num_eq != 0:
                    cls.append(b[i].squeeze(-1))
                    cus.append(b[i].squeeze(-1))
                if (self.num_ineq == 0) and (self.num_eq == 0):
                    cl = []
                    cu = []
                else:
                    cl = np.hstack(cls)
                    cu = np.hstack(cus)

                if (self.num_ineq != 0) and (self.num_eq != 0):
                    G0, A0 = G[i], A[i]
                elif (self.num_ineq != 0) and (self.num_eq == 0):
                    G0, A0 = G[i], np.array(0.0)
                elif (self.num_ineq == 0) and (self.num_eq != 0):
                    G0, A0 = np.array(0.0), A[i]
                else:
                    G0, A0 = np.array(0.0), np.array(0.0)

                nlp = convex_ipopt(
                    Q[i],
                    p[i].squeeze(-1),
                    G0,
                    A0,
                    n=len(y0),
                    m=len(cl),
                    lb=lb[i],
                    ub=ub[i],
                    cl=cl,
                    cu=cu
                )

                nlp.add_option('tol', tol)
                nlp.add_option('print_level', 0)  # 3)
                if init_mu is not None:
                    nlp.add_option('warm_start_init_point', 'yes')
                    nlp.add_option('warm_start_bound_push', 1e-20)
                    nlp.add_option('warm_start_bound_frac', 1e-20)
                    nlp.add_option('warm_start_slack_bound_push', 1e-20)
                    nlp.add_option('warm_start_slack_bound_frac', 1e-20)
                    nlp.add_option('warm_start_mult_bound_push', 1e-20)
                    nlp.add_option('mu_strategy', 'monotone')
                    nlp.add_option('mu_init', init_mu[i].squeeze().cpu().item())

                start_time = time.time()
                if init_g is not None:
                    g = [x.item() for x in init_g[i].cpu()]
                else:
                    g = []

                if init_zl is not None:
                    zl = [x.item() for x in init_zl[i].cpu()]
                else:
                    zl = []

                if init_zu is not None:
                    zu = [x.item() for x in init_zu[i].cpu()]
                else:
                    zu = []

                y, info = nlp.solve(y0, lagrange=g, zl=zl, zu=zu)

                end_time = time.time()
                Y.append(y)
                iters.append(len(nlp.objectives))
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / Q.shape[0]
        else:
            raise NotImplementedError

        return sols, total_time, parallel_time, np.array(iters).mean()


class convex_ipopt(ipopt.Problem):
    def __init__(self, Q, p, G, A, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = Q
        self.p = p
        self.G = G
        self.A = A
        if (self.G == 0.0).all():
            self.num_ineq = 0
        else:
            self.num_ineq = self.G.shape[0]
        if (self.A == 0.0).all():
            self.num_eq = 0
        else:
            self.num_eq = self.A.shape[0]

        self.objectives = []
        self.mus = []
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p@y

    def gradient(self, y):
        return 0.5*(self.Q+self.Q.T)@y + self.p

    def constraints(self, y):
        const_values = []
        if self.num_ineq != 0:
            const_values.append(self.G@y)
        if self.num_eq != 0:
            const_values.append(self.A@y)
        return np.hstack(const_values)

    def jacobian(self, y):
        const_jacob = []
        if self.num_ineq != 0:
            const_jacob.append(self.G.flatten())
        if self.num_eq != 0:
            const_jacob.append(self.A.flatten())
        return np.concatenate(const_jacob)

    def intermediate(self, alg_mod, iter_count, obj_value,
            inf_pr, inf_du, mu, d_norm, regularization_size,
            alpha_du, alpha_pr, ls_trials):
        self.objectives.append(obj_value)
        self.mus.append(mu)