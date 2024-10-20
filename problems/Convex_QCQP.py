import os
import sys
import time
import torch
import numpy as np
import scipy.io as sio
import cyipopt as ipopt

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Convex_QCQP(object):

    def __init__(self, prob_type, learning_type, val_frac=0.0833, test_frac=0.0833, device='cuda:0', seed=17, **kwargs):
        super().__init__()

        self.device = device
        self.seed = seed
        self.learning_type = learning_type
        self.train_frac = 1 - val_frac - test_frac
        self.val_frac = val_frac
        self.test_frac = test_frac


        if prob_type == 'Convex_QCQP_RHS':
            file_path = kwargs['file_path']
            data = sio.loadmat(file_path)
            self.data_size = data['X'].shape[0]

            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size
            torch.manual_seed(self.seed)

            self.num_var = data['Q'].shape[0]
            self.num_ineq = data['H'].shape[0]
            self.num_eq = data['A'].shape[0]
            self.num_lb = 0
            self.num_ub = 0

            if learning_type == 'train':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.train_size, 1).unsqueeze(-1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[:self.train_size].unsqueeze(-1)
                self.Q_ineq = torch.tensor(data['H'], device=self.device).float()
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.train_size, 1).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.val_size, 1).unsqueeze(-1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size:self.train_size + self.val_size].unsqueeze(-1)
                self.Q_ineq = torch.tensor(data['H'], device=self.device).float()
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.val_size, 1).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.test_size, 1).unsqueeze(-1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size + self.val_size:].unsqueeze(-1)
                self.Q_ineq = torch.tensor(data['H'], device=self.device).float()
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.test_size, 1).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf

        elif prob_type == 'Convex_QCQP':
            self.data_size = kwargs['data_size']
            self.train_size = int(self.data_size * self.train_frac)
            self.val_size = int(self.data_size * val_frac)
            self.test_size = self.data_size - self.train_size - self.val_size
            torch.manual_seed(self.seed)

            self.num_var = kwargs['num_var']
            self.num_ineq = kwargs['num_ineq']
            self.num_eq = kwargs['num_eq']
            self.num_lb = 0
            self.num_ub = 0

            if learning_type == 'train':
                self.Q = torch.diag_embed(0.5 * torch.rand(size=(self.data_size, self.num_var), device=device))[:self.train_size]
                self.p = 2 * torch.rand(size=(self.data_size, self.num_var), device=device)[:self.train_size].unsqueeze(-1) - 1
                self.A = 2 * torch.rand(size=(self.data_size, self.num_eq, self.num_var), device=self.device)[:self.train_size] - 1
                self.b = torch.rand(size=(self.data_size, self.num_eq, 1), device=self.device)[:self.train_size] - 0.5
                self.Q_ineq = torch.diag_embed(0.1 * torch.rand(size=(self.num_ineq, self.num_var), device=self.device))
                self.G = 2 * torch.rand(size=(self.data_size, self.num_ineq, self.num_var), device=device)[:self.train_size] - 1
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.diag_embed(0.5 * torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size:self.train_size + self.val_size]
                self.p = 2 * torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size:self.train_size + self.val_size].unsqueeze(-1) - 1
                self.A = 2 * torch.rand(size=(self.data_size, self.num_eq, self.num_var), device=self.device)[self.train_size:self.train_size + self.val_size] - 1
                self.b = torch.rand(size=(self.data_size, self.num_eq, 1), device=self.device)[self.train_size:self.train_size + self.val_size] - 0.5
                self.Q_ineq = torch.diag_embed(0.1 * torch.rand(size=(self.num_ineq, self.num_var), device=self.device))
                self.G = 2 * torch.rand(size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size] - 1
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.diag_embed(0.5 * torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size + self.val_size:]
                self.p = 2 * torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size + self.val_size:].unsqueeze(-1) - 1
                self.A = 2 * torch.rand(size=(self.data_size, self.num_eq, self.num_var), device=self.device)[self.train_size + self.val_size:] - 1
                self.b = torch.rand(size=(self.data_size, self.num_eq, 1), device=self.device)[self.train_size + self.val_size:] - 0.5
                self.Q_ineq = torch.diag_embed(0.1 * torch.rand(size=(self.num_ineq, self.num_var), device=self.device))
                self.G = 2 * torch.rand(size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size + self.val_size:] - 1
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2).unsqueeze(-1)
                self.lb = -torch.inf
                self.ub = torch.inf


    def obj_fn(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return 0.5 * torch.bmm(x.permute(0, 2, 1), torch.bmm(Q, x)) + torch.bmm(p.permute(0, 2, 1), x)

    def obj_grad(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p

    def ineq_resid(self, x, **kwargs):
        Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)

        res = []
        for i in range(self.num_ineq):
            resi = torch.bmm(x.permute(0, 2, 1) @ Q_ineq[i, :, :], x) + torch.bmm(G[:, i, :].unsqueeze(-1).permute(0, 2, 1), x) - c[:, i, :].unsqueeze(-1)
            res.append(resi)
        return torch.concat(res, dim=1)

    def ineq_dist(self, x, **kwargs):
        Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.clamp(self.ineq_resid(x, Q_ineq=Q_ineq, G=G, c=c), 0)

    def ineq_grad(self, x, **kwargs):
        Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
        G = kwargs.get('G', self.G)

        grad_list = []
        for i in range(self.num_ineq):
            grad_list.append(x.permute(0,2,1)@(Q_ineq[i,:,:].T+Q_ineq[i,:,:])+G[:,i,:].unsqueeze(1))
        return torch.concat(grad_list, dim=1)


    def eq_resid(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.bmm(A, x) - b

    def eq_dist(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.abs(self.eq_resid(x, A=A, b=b))

    def eq_grad(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        return A

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
            Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
        batch_size = Q.shape[0]

        # residual
        F_list = []
        F1 = torch.bmm(0.5 * (Q + Q.permute(0, 2, 1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0, 2, 1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(self.eq_grad(x, A=A, b=b).permute(0, 2, 1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = self.ineq_resid(x, Q_ineq=Q_ineq, G=G, c=c) + s
            F3 = eta * s
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = self.eq_resid(x, A=A, b=b)
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
            Q_ineq = kwargs.get('Q_ineq', self.Q_ineq)
            G = kwargs.get('G', self.G)
            c = kwargs.get('c', self.c)
            mu += sigma * ((eta * s).sum(1).unsqueeze(-1))
        if self.num_eq != 0:
            A = kwargs.get('A', self.A)
            b = kwargs.get('b', self.b)
        if self.num_lb != 0:
            lb = kwargs.get('lb', self.lb)
            mu += sigma * ((zl * (x - lb)).sum(1).unsqueeze(-1))
        if self.num_ub != 0:
            ub = kwargs.get('ub', self.ub)
            mu += sigma * ((zu * (ub - x)).sum(1).unsqueeze(-1))
        batch_size = Q.shape[0]
        # mu
        mu = mu / (self.num_ineq + self.num_lb + self.num_ub)

        # residual
        F_list = []
        F1 = torch.bmm(0.5*(Q+Q.permute(0,2,1)), x) + p
        if self.num_ineq != 0:
            F1 += torch.bmm(self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0,2,1), eta)
        if self.num_eq != 0:
            F1 += torch.bmm(self.eq_grad(x, A=A, b=b).permute(0,2,1), lamb)
        if self.num_lb != 0:
            F1 += -zl
        if self.num_ub != 0:
            F1 += zu
        F_list.append(F1)

        if self.num_ineq != 0:
            F2 = self.ineq_resid(x, Q_ineq=Q_ineq, G=G, c=c) + s
            F3 = eta * s - mu
            F_list.append(F2)
            F_list.append(F3)

        if self.num_eq != 0:
            F4 = self.eq_resid(x, A=A, b=b)
            F_list.append(F4)

        if self.num_lb != 0:
            F5 = zl * (x - lb) - mu
            F_list.append(F5)

        if self.num_ub != 0:
            F6 = zu * (ub - x) - mu
            F_list.append(F6)

        F = torch.concat(F_list, dim=1)

        # jacobian of residual
        J_list = []
        J1 = 0.5*(Q+Q.permute(0,2,1))
        if self.num_ineq != 0:
            J1 = torch.concat((J1, self.ineq_grad(x, Q_ineq=Q_ineq, G=G).permute(0,2,1)), dim=2)
        if self.num_eq != 0:
            J1 = torch.concat((J1, self.eq_grad(x, A=A, b=b).permute(0,2,1)), dim=2)
        if self.num_ineq != 0:
            J1 = torch.concat((J1, torch.zeros(size=(batch_size, self.num_var, self.num_ineq), device=self.device)), dim=2)
        if self.num_lb != 0:
            J1 = torch.concat((J1, -torch.diag_embed(torch.ones(size=(batch_size, self.num_lb), device=self.device))), dim=2)
        if self.num_ub != 0:
            J1 = torch.concat((J1, torch.diag_embed(torch.ones(size=(batch_size, self.num_ub), device=self.device))), dim=2)
        J_list.append(J1)

        if self.num_ineq != 0:
            J2 = torch.concat((self.ineq_grad(x, Q_ineq=Q_ineq, G=G), torch.zeros(size=(batch_size, self.num_ineq, self.num_ineq), device=self.device)), dim=2)
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
            J4 = self.eq_grad(x, A=A, b=b)
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
            J5 = torch.concat((J5, torch.diag_embed((x - lb).squeeze(-1))), dim=2)
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
            J6 = torch.concat((J6, torch.diag_embed((ub - x).squeeze(-1))), dim=2)
            J_list.append(J6)

        J = torch.concat(J_list, dim=1)
        return J, F, mu

    def sub_objective(self, y, J, F):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        J: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        F: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        1/2||J@y-F||_2^2 = 1/2(y^T@J^T@y)-y^TJ^TF+1/2(F^TF)
        """
        obj0 = 0.5 * torch.bmm(torch.bmm(y.permute(0, 2, 1), J.permute(0, 2, 1)), torch.bmm(J, y))
        obj1 = torch.bmm(torch.bmm(y.permute(0, 2, 1), J.permute(0, 2, 1)), F)
        obj2 = 0.5 * (torch.bmm(J.permute(0, 2, 1), F))
        return obj0 + obj1 + obj2

    def sub_smooth_grad(self, y, J, F):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        return J^T@J@y+J^T@F
        """
        grad = torch.bmm(torch.bmm(J.permute(0, 2, 1), J), y) + torch.bmm(J.permute(0, 2, 1), F)
        return grad

    def opt_solve(self, solver_type='iopopt', tol=1e-4, initial_y = None, init_mu=None, init_g=None, init_zl=None, init_zu=None):
        if solver_type == 'ipopt':
            Q, p = self.Q.detach().cpu().numpy(), self.p.detach().cpu().numpy()
            if self.num_ineq != 0:
                Q_ineq, G, c = self.Q_ineq.detach().cpu().numpy(), self.G.detach().cpu().numpy(), self.c.detach().cpu().numpy()
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
                cl = np.hstack(cls)
                cu = np.hstack(cus)

                if (self.num_ineq != 0) and (self.num_eq != 0):
                    Q_ineq, G0, A0 = Q_ineq, G[i], A[i]
                elif (self.num_ineq != 0) and (self.num_eq == 0):
                    Q_ineq, G0, A0 = Q_ineq, G[i], np.array([0.0])
                elif (self.num_ineq == 0) and (self.num_eq != 0):
                    Q_ineq, G0, A0 = np.array([0.0]), np.array([0.0]), A[i]

                nlp = convex_ipopt(
                    Q[i],
                    p[i].squeeze(-1),
                    Q_ineq,
                    G0,
                    A0,
                    n=len(y0),
                    m=len(cl),
                    # problem_obj=prob_obj,
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
    def __init__(self, Q, p, Q_ineq, G, A, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = Q
        self.p = p
        self.Q_ineq = Q_ineq
        self.G = G
        self.A = A
        if ((self.G == 0.0).all()) and (len(self.G)==1):
            self.num_ineq = 0
        else:
            self.num_ineq = self.G.shape[0]
        if ((self.A == 0.0).all()) and (len(self.A)==1):
            self.num_eq = 0
        else:
            self.num_eq = self.A.shape[0]

        self.objectives = []
        self.mus = []
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p@y

    def gradient(self, y):
        return self.Q@y + self.p

    def constraints(self, y):
        const_values = []
        if self.num_ineq != 0:
            if (self.Q_ineq == 0).all():
                const_values.append(self.G @ y)
            else:
                ineq_const = []
                for i in range(self.num_ineq):
                    ineq_const.append(y.T@self.Q_ineq[i]@y+self.G[i,:].T@y)
                const_values.append(np.array(ineq_const))
        if self.num_eq != 0:
            const_values.append(self.A @ y)
        return np.hstack(const_values)


    def jacobian(self, y):
        const_jacob = []
        if self.num_ineq != 0:
            if (self.Q_ineq == 0).all():
                const_jacob.append(self.G.flatten())
            else:
                ineq_grad = []
                for i in range(self.num_ineq):
                    ineq_grad.append((self.Q_ineq[i,:,:].T+self.Q_ineq[i,:,:])@y+self.G[i,:])
                const_jacob.append(np.concatenate(ineq_grad, axis=-1).T.flatten())
        if self.num_eq != 0:
            const_jacob.append(self.A.flatten())
        return np.concatenate(const_jacob)

    def intermediate(self, alg_mod, iter_count, obj_value,
            inf_pr, inf_du, mu, d_norm, regularization_size,
            alpha_du, alpha_pr, ls_trials):
        self.objectives.append(obj_value)
        self.mus.append(mu)