import torch
import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


class EarlyStopping(object):
    def __init__(self, save_path, patience=10):
        dt = datetime.datetime.now()
        self.filename = save_path
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, model, tol, *args):
        if all(vio <= tol for vio in args):
            if self.best_loss is None:
                self.best_loss = loss
                self.save_checkpoint(model)
                self.counter = 0
            elif (loss <= self.best_loss):
                self.save_checkpoint(model)
                self.best_loss = np.min((loss, self.best_loss))
                self.counter = 0
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))



def calculate_step(data, x, delta_x, eta, delta_eta, s, delta_s, zl, delta_zl, zu, delta_zu,
                             lb, ub, tau, use_line_search, device='cuda:0'):
    """
    x: [batch_size, num_var, 1]
    delta_x: [batch_size, num_var, 1]
    eta: [batch_size, num_ineq, 1]
    delta_eta: [batch_size, num_ineq, 1]
    s: [batch_size, num_ineq, 1]
    delta_s: [batch_size, num_ineq, 1]
    zl: [batch_size, num_lb, 1]
    delta_zl: [batch_size, num_lb, 1]
    zu: [batch_size, num_ub, 1]
    delta_zu: [batch_size, num_ub, 1]
    return: min(1,min((-eta/delta_eta)|delta_eta<0))
    """
    if (data.num_lb != 0) and (data.num_ub != 0):
        alpha_x1 = ((lb - x) / delta_x)
        alpha_x2 = ((ub - x) / delta_x)
        masked_matrix_1 = torch.where(alpha_x1 >= 0, alpha_x1, torch.tensor(float('inf'), device=device))
        masked_matrix_2 = torch.where(alpha_x2 >= 0, alpha_x2, torch.tensor(float('inf'), device=device))
        masked_matrix_x = torch.concat((tau * masked_matrix_1, tau * masked_matrix_2, torch.ones(size=(x.shape[0], 1, 1), device=device)), dim=1)
        alpha_x = masked_matrix_x.min(dim=1).values.unsqueeze(-1)

        alpha_zl = -(zl / delta_zl)
        masked_matrix_4 = torch.where(alpha_zl >= 0, alpha_zl, torch.tensor(float('inf'), device=device))
        masked_matrix_zl = torch.concat((tau * masked_matrix_4, torch.ones(size=(zl.shape[0], 1, 1), device=device)), dim=1)
        alpha_zl = masked_matrix_zl.min(dim=1).values.unsqueeze(-1)

        alpha_zu = -(zu / delta_zu)
        masked_matrix_5 = torch.where(alpha_zu >= 0, alpha_zu, torch.tensor(float('inf'), device=device))
        masked_matrix_zu = torch.concat((tau * masked_matrix_5, torch.ones(size=(zu.shape[0], 1, 1), device=device)), dim=1)
        alpha_zu = masked_matrix_zu.min(dim=1).values.unsqueeze(-1)  # [batch_size, 1, 1]
    elif (data.num_lb != 0) and (data.num_ub == 0):
        alpha_x = ((lb-x) / delta_x)
        masked_matrix_1 = torch.where(alpha_x >= 0, alpha_x, torch.tensor(float('inf'), device=device))
        masked_matrix_x = torch.concat((tau*masked_matrix_1, torch.ones(size=(x.shape[0], 1, 1), device=device)), dim=1)
        alpha_x = masked_matrix_x.min(dim=1).values.unsqueeze(-1)

        alpha_zl = -(zl / delta_zl)
        masked_matrix_4 = torch.where(alpha_zl >= 0, alpha_zl, torch.tensor(float('inf'), device=device))
        masked_matrix_zl = torch.concat((tau * masked_matrix_4, torch.ones(size=(zl.shape[0], 1, 1), device=device)), dim=1)
        alpha_zl = masked_matrix_zl.min(dim=1).values.unsqueeze(-1)

        alpha_zu = 0.0
    elif (data.num_lb == 0) and (data.num_ub != 0):
        alpha_x = ((ub - x) / delta_x)
        masked_matrix_1 = torch.where(alpha_x >= 0, alpha_x, torch.tensor(float('inf'), device=device))
        masked_matrix_x = torch.concat((tau*masked_matrix_1, torch.ones(size=(x.shape[0], 1, 1), device=device)), dim=1)
        alpha_x = masked_matrix_x.min(dim=1).values.unsqueeze(-1)

        alpha_zl = 0.0

        alpha_zu = -(zu / delta_zu)
        masked_matrix_4 = torch.where(alpha_zu >= 0, alpha_zu, torch.tensor(float('inf'), device=device))
        masked_matrix_zu = torch.concat((tau * masked_matrix_4, torch.ones(size=(zu.shape[0], 1, 1), device=device)), dim=1)
        alpha_zu = masked_matrix_zu.min(dim=1).values.unsqueeze(-1)  # [batch_size, 1, 1]
    else:
        alpha_x = 0.0
        alpha_zl = 0.0
        alpha_zu = 0.0


    if data.num_ineq != 0:
        alpha_eta = -(eta / delta_eta)
        masked_matrix_2 = torch.where(alpha_eta >= 0, alpha_eta, torch.tensor(float('inf'), device=device))
        masked_matrix_eta = torch.concat((tau * masked_matrix_2, torch.ones(size=(eta.shape[0], 1, 1), device=device)), dim=1)
        alpha_eta = masked_matrix_eta.min(dim=1).values.unsqueeze(-1)  # [batch_size, 1, 1]

        alpha_s = -(s / delta_s)
        masked_matrix_3 = torch.where(alpha_s >= 0, alpha_s, torch.tensor(float('inf'), device=device))
        masked_matrix_s = torch.concat((tau * masked_matrix_3, torch.ones(size=(s.shape[0], 1, 1), device=device)),dim=1)
        alpha_s = masked_matrix_s.min(dim=1).values.unsqueeze(-1)  # [batch_size, 1, 1]
    else:
        alpha_eta = 0.0
        alpha_s = 0.0

    if use_line_search:
        pass

    return alpha_x, alpha_eta, alpha_s, alpha_zl, alpha_zu



def plot_objective_values(objs, outer_T, save_path, label, best_obj=None):
    # plot settings
    font = {
        'family': 'Times New Roman',
        'size': 12
    }
    plt.rc('font', **font)

    # IPM-LSTM objective values
    x = np.arange(outer_T)
    sns.lineplot(x=x, y=objs[:outer_T], color='cornflowerblue', linewidth=2.0, marker="o",
                 markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                 label=label)
    if best_obj is not None:
        sns.lineplot(x=x, y=np.repeat(best_obj, len(x)), color="lightcoral", linewidth=2.0, linestyle='--',
                     markersize=8, markeredgecolor="lightcoral", markeredgewidth=1,
                     label='Optimal value')

    plt.xlabel("IPM iteration", fontsize=12)
    plt.ylabel("Objective value", fontsize=12)
    plt.legend(frameon=True, fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.5)

    plt.grid(True)
    plt.savefig(save_path, format='eps')
    plt.close()

def plot_inner_losses(data, inner_T, time_steps, save_path):
    # plot settings
    colors = ['dimgrey', 'paleturquoise', 'orange', 'yellowgreen', 'c', 'dodgerblue', 'mediumpurple', 'pink']
    markers = ["o", "s", "*", "v", "1", "2", "_"]
    font = {
        'family': 'Times New Roman',
        'size': 12
    }
    plt.rc('font', **font)
    plt.rcParams["text.usetex"] = True

    # inner iterations analysis
    x = np.arange(inner_T)
    for i in range(len(time_steps)):
        sns.lineplot(x=x, y=data['inner_losses'][time_steps[i]], color=colors[i], linewidth=2.0, marker=markers[i],
                     markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                     label='IPM iteration {}'.format(time_steps[i]+1))

    plt.xlabel("LSTM time step", fontsize=12)
    plt.ylabel(r'$\frac{1}{2}\left\|J^ky^k + F^k \right\|^{2}$', fontsize=12)

    plt.legend(frameon=True, fontsize=15)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.5)

    plt.grid(True)
    plt.savefig(save_path, format='eps')
    plt.close()


def plot_condtions(data, outer_T, eta, sigma, save_path, cond="Condition1"):
    font = {
        'family': 'Times New Roman',
        'size': 12
    }
    plt.rc('font', **font)
    plt.rcParams["text.usetex"] = True

    # plot
    x = np.arange(outer_T) + 1
    if cond == "Condition1":
        sns.lineplot(x=x, y=data['residual'].squeeze(), color='red', linewidth=2.0, marker="o",
                    markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                    label=r'$\left\|J^ky^k+F^k\right\|$')
        sns.lineplot(x=x, y=eta*(data['mu'].squeeze()/sigma), color='dodgerblue', linewidth=2.0, marker="s",
                    markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                    label=r'$\eta \left[(z^k)^{\top}x^k \right] / n$')
    elif cond == "Condition2":
        sns.lineplot(x=x, y=data['y_norm'].squeeze(), color='orange', linewidth=2.0, marker="*",
                     markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                     label=r'$\left\|y^k\right\|$')
        sns.lineplot(x=x, y=(1+sigma+eta) * data['F0'].squeeze(), color='yellowgreen', linewidth=2.0, marker="v",
                     markersize=8, markeredgecolor="white", markeredgewidth=1.5,
                     label=r'$\left\|F_0(x^k,\lambda^k,z^k)\right\|$')


    plt.xlabel("IPM iteration", fontsize=12)
    plt.legend(frameon=True, fontsize=15)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor("#CCCCCC")
        spine.set_linewidth(1.5)
    plt.grid(True)
    plt.savefig(save_path, format='jpg')
    plt.close()






