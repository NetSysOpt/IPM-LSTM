import torch
import torch.nn as nn


"""
model inputs: optimizees
model outputs: [X_1, X_2, ..., X_T]
"""
class LSTM(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 iter_step,
                 device):
        #input_dim: the dimension of model inputs, for example, [X_t, ▽f(X_t)]
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.iter_step = iter_step
        self.device = device

        self.W_i = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim),device=self.device), requires_grad=True)
        self.U_i = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim),device=self.device), requires_grad=True)
        self.b_i = nn.Parameter(torch.zeros((hidden_dim), device=device, dtype=torch.float32), requires_grad=True)

        self.W_f = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim),device=self.device), requires_grad=True)
        self.U_f = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim),device=self.device), requires_grad=True)
        self.b_f = nn.Parameter(torch.zeros((hidden_dim), device=device, dtype=torch.float32), requires_grad=True)

        self.W_o = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim),device=self.device), requires_grad=True)
        self.U_o = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim),device=self.device), requires_grad=True)
        self.b_o = nn.Parameter(torch.zeros((hidden_dim), device=device, dtype=torch.float32), requires_grad=True)

        self.W_u = nn.Parameter(torch.normal(mean=0, std=0.01, size=(input_dim, hidden_dim),device=self.device), requires_grad=True)
        self.U_u = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, hidden_dim),device=self.device), requires_grad=True)
        self.b_u = nn.Parameter(torch.zeros((hidden_dim), device=device, dtype=torch.float32), requires_grad=True)

        self.W_h = nn.Parameter(torch.normal(mean=0, std=0.01, size=(hidden_dim, 1),device=self.device), requires_grad=True)
        self.b_h = nn.Parameter(torch.zeros((1), device=device, dtype=torch.float32), requires_grad=True)

    def name(self):
        return 'lstm'

    def forward(self, data, y, J, F, states=(None, None)):
        """
        X: [batch_size, num_var, 1]
        """

        if states[0] is None:
            H_t = torch.zeros((y.shape[0], y.shape[1], self.hidden_dim), device=self.device)
        else:
            H_t = states[0]

        if states[1] is None:
            C_t = torch.zeros((y.shape[0], y.shape[1], self.hidden_dim), device=self.device)
        else:
            C_t = states[1]

        final_y = 0.0
        final_loss = 0.0
        losses = []
        loss0 = 10000

        for iter in range(self.iter_step):
            grad = data.sub_smooth_grad(y, J, F)

            inputs = torch.concat((y, grad), dim=-1)
            I_t = torch.sigmoid(inputs@self.W_i+H_t@self.U_i+self.b_i)
            F_t = torch.sigmoid(inputs@self.W_f+H_t@self.U_f+self.b_f)
            O_t = torch.sigmoid(inputs@self.W_o+H_t@self.U_o+self.b_o)
            U_t = torch.tanh(inputs@self.W_u+H_t@self.U_u+self.b_u)
            C_t = I_t*U_t+F_t*C_t
            H_t = O_t*torch.tanh(C_t)
            grad = H_t@self.W_h+self.b_h
            y = y - grad

            loss = (data.sub_objective(y, J, F).mean())/self.iter_step
            final_loss += loss
            losses.append(loss.detach().cpu())
            if (loss < loss0) or (iter == 0):
                loss0 = loss
                final_y = y

        return final_y, final_loss, losses