class CDE_Func(nn.Module):
    def __init__(self, hp, input_channels=2, hidden_channels=2):
        super(CDE_Func, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.device = hp.device
        
        self.linear1 = torch.nn.Linear(hidden_channels, 256).to(self.device)
        self.linear2 = torch.nn.Linear(256, input_channels*hidden_channels).to(self.device)

    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z.to(self.device))
        z = z.relu()
        z = self.linear2(z.to(self.device))
    
        z = z.tanh()

        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        
        return z.to(self.device)


# class T10_transformer(nn.Module):
class T10_transformer(nn.Module):
    def __init__(self, hp, input_channels=2, hidden_channels=8, output_channels=4):
        super(T10_transformer, self).__init__()


        self.device = hp.device
        self.interpolation = "cubic"
        self.hp = hp
        # only the signals
        self.features = 1
        self.hidden_size = hidden_channels
        
        self.len_concat = hp.len_concat
        self.fa_concat = hp.fa_concat
        self.learn_emb = hp.learn_emb
        self.embed_time = hp.embed_time
        self.dim = hp.input_dim
        self.device = hp.device
        self.nhidden = hp.n_hidden
        self.num_heads = hp.n_heads
        self.gru_layers = hp.gru_layers
        self.dropout_p = hp.dropout_p
        self.gru_bi = hp.gru_bi
        self.mh_att = hp.mh_attention
        self.one_reg = hp.one_reg
        self.ref_angle = hp.ref_angle
        self.layer_bool = hp.layer_bool
        self.T10_under = hp.simulations.T10bounds[0]
        self.T10_upper = hp.simulations.T10bounds[1]

        self.initial = torch.nn.Linear(input_channels, hidden_channels).to(hp.device)
        
        # function to give to the ODE solver
        self.ode_net = CDE_Func(hp, input_channels, hidden_channels)
        

        hidden_reg = self.hidden_size

        if self.len_concat:
            hidden_reg += 1
        
        if self.fa_concat:
            if self.hp.xFA[1][1] is None:
                hidden_reg += self.hp.xFA[1][0]
            else:
                hidden_reg += self.hp.xFA[1][1]

        self.reg_T10 = nn.Sequential(nn.Dropout(self.dropout_p),
                                     nn.Linear(int(hidden_reg), 200),
                                     nn.GELU(),
                                     nn.Dropout(self.dropout_p),
                                     nn.Linear(200, 200),
                                     nn.GELU(),
                                     nn.Linear(200, 1))
    
    # @jit.script_method
    def forward(self, X_fa_in, fa_vals_in, fa_mask, fa_len, TR_vals):

        mean = torch.sum(X_fa_in, dim=1) / fa_len.squeeze()
        X_fa_in = X_fa_in / mean.unsqueeze(dim=1)
        
        X_fa = push_zeros(X_fa_in.clone(), device=self.device)
        fa_vals = push_zeros(fa_vals_in.clone(), device=self.device)

        last_X = X_fa[
            torch.arange(X_fa.size(0)),
            fa_len.squeeze().long()-1].unsqueeze(-1).expand(*X_fa.size())
        
        last_fa = fa_vals_in[
            torch.arange(X_fa_in.size(0)),
            fa_len.squeeze().long()-1].unsqueeze(-1).expand(*X_fa.size())
        
        X_fa = torch.where(X_fa == 0., last_X, X_fa)
        fa_vals = torch.where(fa_vals == 0., last_fa, fa_vals)
        
        x = torch.cat((
            fa_vals.unsqueeze(dim=-1),
            X_fa.unsqueeze(dim=-1)),
            dim=-1)
        
        # waarom moet dit ?????
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x)

        # ik snap ook niet wat hier gebeurd? 
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        X0 = X.evaluate(X.interval[0]).to(self.device)
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.ode_net,
                              t=X.interval)

        z_T = z_T[:, 1]
        
        # regress T10 value
        T10 = self.reg_T10(z_T).squeeze()

        # # update T10
        T10_diff = self.T10_upper - self.T10_under
        T10 = self.T10_under + torch.sigmoid(T10.unsqueeze(1)) * T10_diff

        R1 = 1 / T10
        X_out = torch.mul(
            (1 - torch.exp(torch.mul(-TR_vals, R1))), torch.sin(fa_vals_in)) / (
            1 - torch.mul(torch.cos(fa_vals_in),
                          torch.exp(torch.mul(-TR_vals, R1))))
       
        mean = torch.sum(X_out, dim=1) / fa_len.squeeze()
        X_out = X_out / mean.unsqueeze(dim=1)

        M0 = torch.ones_like(T10)
        return X_out, T10, M0

