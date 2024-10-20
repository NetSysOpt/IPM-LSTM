import os
import osqp
import time
import torch
import cyipopt as ipopt
import numpy as np
import scipy.io as sio



os.environ['KMP_DUPLICATE_LIB_OK']='True'


class Nonconvex_Program(object):
    """
        minimize_x 0.5*x^T Q x + p^Tsin(x)
        s.t.       Ax =  b
                   Gx <= c

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

        if prob_type == 'Nonconvex_Program_RHS':
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
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.train_size, 1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[:self.train_size]
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.train_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.train_size, 1)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.val_size, 1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size:self.train_size + self.val_size]
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.val_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.val_size, 1)
                self.batch_size = self.val_size
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.tensor(data['Q'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.p = torch.tensor(data['p'], device=self.device).float().repeat(self.test_size, 1)
                self.A = torch.tensor(data['A'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.b = torch.tensor(data['X'], device=self.device).float()[self.train_size + self.val_size:]
                self.G = torch.tensor(data['G'], device=self.device).float().unsqueeze(0).repeat(self.test_size, 1, 1)
                self.c = torch.tensor(data['h'], device=self.device).float().repeat(self.test_size, 1)
                self.lb = -torch.inf
                self.ub = torch.inf

        elif prob_type == 'Nonconvex_Program':
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
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[:self.train_size]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[:self.train_size]
                self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[:self.train_size]
                self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[:self.train_size] - 1  # [-1, 1]
                self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[:self.train_size]
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'val':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size:self.train_size + self.val_size]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
                self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
                self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[self.train_size:self.train_size + self.val_size] - 1  # [-1, 1]
                self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size:self.train_size + self.val_size]
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2)
                self.lb = -torch.inf
                self.ub = torch.inf
            elif learning_type == 'test':
                self.Q = torch.diag_embed(torch.rand(size=(self.data_size, self.num_var), device=device))[self.train_size + self.val_size:]
                self.p = torch.rand(size=(self.data_size, self.num_var), device=device)[self.train_size + self.val_size:]
                self.A = torch.normal(mean=0, std=1, size=(self.data_size, self.num_eq, self.num_var), device=device)[self.train_size + self.val_size:]
                self.b = 2 * torch.rand(size=(self.data_size, self.num_eq), device=device)[self.train_size + self.val_size:] - 1  # [-1, 1]
                self.G = torch.normal(mean=0, std=1, size=(self.data_size, self.num_ineq, self.num_var), device=device)[self.train_size + self.val_size:]
                self.c = torch.sum(torch.abs(torch.bmm(self.G, torch.pinverse(self.A))), dim=2)
                self.lb = -torch.inf
                self.ub = torch.inf

    def name(self):
        str = '{}_{}_{}_{}_{}'.format(self.prob_type, self.num_ineq, self.num_eq, self.num_lb, self.num_ub)
        return str

    def obj_fn(self, x, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        return 0.5*torch.bmm(x.permute(0, 2, 1), torch.bmm(Q, x))+torch.bmm(p.unsqueeze(-1).permute(0,2,1), torch.sin(x))

    def eq_resid(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.bmm(A, x) - b.unsqueeze(-1)

    def eq_dist(self, x, **kwargs):
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        return torch.abs(self.eq_resid(x, A=A, b=b))

    def ineq_resid(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.bmm(G, x) - c.unsqueeze(-1)

    def ineq_dist(self, x, **kwargs):
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        return torch.clamp(self.ineq_resid(x, G=G, c=c), 0)


    def F0(self, x, eta, lamb, s, **kwargs):
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        batch_size = Q.shape[0]

        r1 = torch.bmm(Q, x) + torch.bmm(torch.diag_embed(p), torch.cos(x)) + \
             torch.bmm(A.permute(0, 2, 1), lamb) + torch.bmm(G.permute(0, 2, 1), eta)
        r2 = torch.bmm(G, x) - c.unsqueeze(-1) + s
        r3 = eta * s
        r4 = torch.bmm(A, x) - b.unsqueeze(-1)
        r = torch.concat((r1, r2, r3, r4), dim=1)
        return r

    def cal_kkt(self, x, eta, s, lamb, zl, zu, sigma, **kwargs):
        """
        x: [batch_size, num_var, 1]
        eta: [batch_size, num_ineq, 1]
        lamb: [batch_size, num_eq, 1]
        s: [batch_size, num_ineq, 1]
        mu: [batch_size, 1]
        b: [batch_size, num_eq]

        return:
        r: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        H: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        """
        Q = kwargs.get('Q', self.Q)
        p = kwargs.get('p', self.p)
        G = kwargs.get('G', self.G)
        c = kwargs.get('c', self.c)
        A = kwargs.get('A', self.A)
        b = kwargs.get('b', self.b)
        batch_size = Q.shape[0]
        # mu
        mu = sigma * ((eta * s).sum(1).unsqueeze(-1)) / self.num_ineq

        # calculate KKT linear system residual vector
        r1 = torch.bmm(Q, x) + torch.bmm(torch.diag_embed(p), torch.cos(x)) + \
             torch.bmm(A.permute(0, 2, 1), lamb) + torch.bmm(G.permute(0, 2, 1), eta)
        r2 = torch.bmm(G, x) - c.unsqueeze(-1) + s
        r3 = eta * s - mu
        r4 = torch.bmm(A, x) - b.unsqueeze(-1)
        r = torch.concat((r1, r2, r3, r4), dim=1)

        # calculate KKT linear system matrix
        H1 = torch.concat((Q - torch.bmm(torch.diag_embed(p), torch.diag_embed(torch.sin(x).squeeze())),
                           G.permute(0, 2, 1), A.permute(0, 2, 1),
                           torch.zeros(size=(batch_size, self.num_var, self.num_ineq), device=self.device)), dim=2)
        H2 = torch.concat((G, torch.zeros(size=(batch_size, self.num_ineq, self.num_ineq), device=self.device),
                           torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device),
                           torch.diag_embed(torch.ones(size=(batch_size, self.num_ineq), device=self.device))), dim=2)
        H3 = torch.concat((torch.zeros(size=(batch_size, self.num_ineq, self.num_var), device=self.device),
                           torch.diag_embed(s.squeeze()),
                           torch.zeros(size=(batch_size, self.num_ineq, self.num_eq), device=self.device),
                           torch.diag_embed(eta.squeeze())), dim=2)
        H4 = torch.concat((A, torch.zeros(size=(batch_size, self.num_eq, self.num_ineq + self.num_ineq + self.num_eq), device=self.device)), dim=2)
        H = torch.concat((H1, H2, H3, H4), dim=1)
        return H, r, mu

    def sub_objective(self, y, H, r):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        H: [batch_size, num_var+num_ineq+num_ineq+num_eq, num_var+num_ineq+num_ineq+num_eq]
        r: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        1/2||H@delta_r-r||_2^2 = 1/2(deta_r^T@H^T@Hdelta_r)-deta_r^TH^Tr+1/2(r^Tr)
        """
        obj0 = 0.5 * torch.bmm(torch.bmm(y.permute(0, 2, 1), H.permute(0, 2, 1)), torch.bmm(H, y))
        obj1 = torch.bmm(torch.bmm(y.permute(0, 2, 1), H.permute(0, 2, 1)), r)
        obj2 = 0.5 * (torch.bmm(r.permute(0, 2, 1), r))
        return obj0+obj1+obj2

    def sub_smooth_grad(self, y, H, r):
        """
        y: [batch_size, num_var+num_ineq+num_ineq+num_eq, 1]
        return H^T@H@delta_r+H^T@r
        """
        grad = torch.bmm(torch.bmm(H.permute(0, 2, 1), H), y) + torch.bmm(H.permute(0, 2, 1), r)
        return grad

    def opt_solve(self, solver_type='ipopt', tol=1e-4, initial_y=None, init_mu=None, init_g=None):
        Q = self.Q.detach().cpu().numpy()
        p = self.p.detach().cpu().numpy()
        G = self.G.detach().cpu().numpy()
        c = self.c.detach().cpu().numpy()
        A = self.A.detach().cpu().numpy()
        b = self.b.detach().cpu().numpy()

        Y = []
        iters = []
        total_time = 0
        if solver_type == 'ipopt':
            for i in range(Q.shape[0]):
                if initial_y is None:
                    # y0 = np.linalg.pinv(A[i]) @ b[i]  # feasible initial point
                    y0 = np.zeros(self.num_var)
                else:
                    y0 = initial_y[i].cpu().numpy()

                # upper and lower bounds on variables
                lb = -np.infty * np.ones(y0.shape)
                ub = np.infty * np.ones(y0.shape)

                # upper and lower bounds on constraints
                cl = np.hstack([-np.inf * np.ones(G[i].shape[0]), b[i]])
                cu = np.hstack([c[i], b[i]])

                nlp = nonconvex_ipopt(
                    Q[i],
                    p[i],
                    G[i],
                    A[i],
                    n=len(y0),
                    m=len(cl),
                    # problem_obj=prob_obj,
                    lb=lb,
                    ub=ub,
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
                    y, info = nlp.solve(y0, lagrange=[x.item() for x in init_g[i].cpu()])
                else:
                    y, info = nlp.solve(y0)
                end_time = time.time()
                Y.append(y)
                iters.append(len(nlp.objectives))
                total_time += (end_time - start_time)

            sols = np.array(Y)
            parallel_time = total_time / Q.shape[0]
        else:
            raise NotImplementedError

        return sols, total_time, parallel_time, np.array(iters).mean()

class nonconvex_ipopt(ipopt.Problem):
    def __init__(self, Q, p, G, A, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = Q
        self.p = p
        self.G = G
        self.A = A
        self.objectives = []
        self.mus = []
        self.tril_indices = np.tril_indices(Q.shape[0])

    def objective(self, y):
        return 0.5 * (y @ self.Q @ y) + self.p @ np.sin(y)

    def gradient(self, y):
        return self.Q @ y + (self.p * np.cos(y))

    def constraints(self, y):
        return np.hstack([self.G @ y, self.A @ y])

    def jacobian(self, y):
        return np.concatenate([self.G.flatten(), self.A.flatten()])

    # # Don't use: In general, more efficient with numerical approx
    # def hessian(self, y, lagrange, obj_factor):
    #     H = obj_factor * (self.Q - np.diag(self.p * np.sin(y)) )
    #     return H[self.tril_indices]

    def intermediate(self, alg_mod, iter_count, obj_value,
                    inf_pr, inf_du, mu, d_norm, regularization_size,
                    alpha_du, alpha_pr, ls_trials):
        self.objectives.append(obj_value)
        self.mus.append(mu)