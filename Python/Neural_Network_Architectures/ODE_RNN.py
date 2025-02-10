def init_network_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.constant_(m.bias, val=0)

class GRU_Unit(nn.Module):
    def __init__(self, latent_dim, input_dim, n_units=100):
        super(GRU_Unit, self).__init__()

        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim  + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim ))
        init_network_weights(self.new_state_net)


    def forward(self, x, y_mean, mask):
        y_concat = torch.cat((y_mean, x), dim=-1)
        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat((y_mean * reset_gate, x), dim=-1)
        new_state = self.new_state_net(concat)

        output_y = (1 - update_gate) * new_state + update_gate * y_mean
      
        mask = (torch.sum(mask, -1, keepdim=True) > 0).float()
        mask = mask.unsqueeze(-1).float()
        new_y = mask * output_y + (1 - mask) * y_mean

        return new_y

def create_net(n_inputs, n_outputs, n_layers, n_units, nl=nn.Tanh):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nl())
        layers.append(nn.Linear(n_units, n_units))
    
    layers.append(nl())
    layers.append(nn.Linear(n_units, n_outputs))
    
    return nn.Sequential(*layers)


class DiffeqSolver(nn.Module):
    def __init__(self, ode_func, method, odeint_rtol, odeint_atol):
        super(DiffeqSolver, self).__init__()
        
        self.ode_method = method
        self.ode_func = ode_func
        
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        
    def forward(self, first_point, time_steps_to_predict):
        
        pred_y = odeint(self.ode_func, first_point, time_steps_to_predict,
                        rtol=self.odeint_rtol, atol=self.odeint_atol,
                        method=self.ode_method)
        
        pred_y = pred_y.permute(1,2,0,3)
        
        return pred_y
        


class ODEFunc(nn.Module):
    def __init__(self, ode_func_net):
        super(ODEFunc, self).__init__()
        
        self.gradient_net = ode_func_net
        
    def forward(self, t_local, y):
        return self.get_ode_gradient_nn(t_local, y)
    
    def get_ode_gradient_nn(self, t_local, y):
        return self.gradient_net(y)
        


class ODE_RNN(nn.Module):
    def __init__(self, hp):
        super(ODE_RNN, self).__init__()
        
        # network parameters
        self.hp = hp
        self.device = hp.device
        self.nhidden = hp.n_hidden

        self.input_size = hp.input_size
        self.rnn_unit = hp.rnn_unit
     
        hidden_reg = int(self.nhidden)   
     
        self.score_T10 = nn.Sequential(nn.Linear(hidden_reg, 1, bias=False),
                                        nn.Softmax(dim=1))
        
        self.score_M0 = nn.Sequential(nn.Linear(hidden_reg, 1, bias=False),
                                        nn.Softmax(dim=1))

        self.reg_T10 = nn.Sequential(nn.Linear(hidden_reg, 300),
                                     nn.ELU(),
                                     nn.Linear(300, 300),
                                     nn.ELU(),
                                     nn.Linear(300, 1))

        self.reg_M0 = nn.Sequential(nn.Linear(hidden_reg, 300),
                                    nn.ELU(),
                                    nn.Linear(300, 300),
                                    nn.ELU(),
                                    nn.Linear(300, 1))

        latent_dim = self.nhidden
        layers = 3
        units = 124
        ode_func_net = create_net(latent_dim, latent_dim,
                                  n_layers=layers,
                                  n_units=units,
                                  nl=nn.Tanh)
        
        self.x0 = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                nn.Tanh())

        ode_method = 'rk4'
        rec_ode_func = ODEFunc(ode_func_net=ode_func_net)
        self.ode_solver = DiffeqSolver(rec_ode_func, ode_method, odeint_rtol=1e-3, odeint_atol=1e-4)

        if self.rnn_unit == 'RNN':
            self.rnn_cell = nn.RNNCell(1, self.nhidden)
        elif self.rnn_unit == 'GRU':
            self.rnn_cell = nn.GRUCell(1, self.nhidden)
            

    def forward(self, X_fa_in, fa_vals, fa_mask, fa_len, TR_vals, fa_union):
        X_fa = X_fa_in
        
        X_fa = X_fa[:, :len(fa_union)]
        fa_mask = fa_mask[:, :len(fa_union)]
        
        prev_hidden = torch.zeros((X_fa.size(0), self.nhidden)).to(self.hp.device)

        ode_trajectory = prev_hidden.unsqueeze(dim=1).clone().detach()
        
        num_steps = 4
        
        """ first ODE step """
        time_points = torch.linspace(0, fa_union[0], num_steps).to(self.hp.device)
        ode_sol_steps = self.ode_solver(prev_hidden.unsqueeze(0), time_points)
             
        prev_hidden = ode_sol_steps[0][:, -1]
        
        """ first RNN step """
        prev_hidden_rnn = self.rnn_cell(X_fa[:, 0].unsqueeze(-1),
                                        prev_hidden,
                                        mask=fa_mask[:, 0]
                                        )

        mask = fa_mask[:, 0].unsqueeze(-1).float()
        prev_hidden = prev_hidden_rnn * mask + (1 - mask) * prev_hidden
 
        hidden_ode = prev_hidden.unsqueeze(1)
        
        ode_trajectory = ode_sol_steps[0].clone()
        ode_trajectory[:, -1] = prev_hidden.clone().detach()
        
        trajectory = ode_trajectory

        for i in range(1, len(fa_union)):
            time_points = torch.linspace(fa_union[i-1], fa_union[i], num_steps).to(self.hp.device)
            ode_sol_steps = self.ode_solver(prev_hidden.unsqueeze(0), time_points)
            prev_hidden = ode_sol_steps[0][:, -1]
            
            prev_hidden_rnn = self.rnn_cell(X_fa[:, i].unsqueeze(-1),
                                            prev_hidden,
                                            mask=fa_mask[:, i]
                                            )

            mask = fa_mask[:, i].unsqueeze(-1).float()
            prev_hidden = prev_hidden_rnn * mask + (1 - mask) * prev_hidden

            hidden_ode = torch.cat((hidden_ode,
                                    prev_hidden.unsqueeze(1)), dim=1)
            
            ode_trajectory = ode_sol_steps[0].clone().detach()
            ode_trajectory[:, -1] = prev_hidden.clone().detach()
            
            trajectory = torch.cat((trajectory, 
                                    ode_trajectory), dim=1)

        hidden_T10 = torch.sum(self.score_T10(hidden_ode)*hidden_ode, dim=1)
        hidden_M0 = torch.sum(self.score_M0(hidden_ode)*hidden_ode, dim=1)
        
        T10 = self.reg_T10(hidden_T10).squeeze()
        M0 = self.reg_M0(hidden_M0).squeeze()        

        # # update T10
        T10_diff = self.hp.simulations.T10bounds[1] - self.hp.simulations.T10bounds[0]
        T10 = self.hp.simulations.T10bounds[0] + torch.sigmoid(T10.unsqueeze(1)) * T10_diff
        
        M0_diff = self.hp.simulations.M0bounds[1] - self.hp.simulations.M0bounds[0]
        M0 = self.hp.simulations.M0bounds[0] + torch.sigmoid(M0.unsqueeze(1)) * M0_diff
    
        R1 = 1 / T10
        X_out = torch.mul(
            (1 - torch.exp(torch.mul(-TR_vals, R1))), torch.sin(fa_vals)) / (
            1 - torch.mul(torch.cos(fa_vals), torch.exp(torch.mul(-TR_vals, R1))))

        X_out *= M0

        return X_out, T10, M0, trajectory
