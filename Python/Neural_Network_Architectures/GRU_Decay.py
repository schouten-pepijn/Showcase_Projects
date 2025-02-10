class GRUDcell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):        
        super(GRUDcell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x_combined = nn.Linear(input_size, 3 * hidden_size, bias=True)
        
        self.h_n = nn.Linear(hidden_size, hidden_size, bias=False)
        self.h_combined = nn.Linear(hidden_size, 2 * hidden_size, bias=False)

        self.m_combined = nn.Linear(input_size, 3 * hidden_size, bias=False)
        
        self.gamma_x = nn.Linear(input_size, input_size, bias=True)
        self.gamma_h = nn.Linear(input_size, hidden_size, bias=True)

        self.relu = nn.ReLU()
        
        self.zeros_in = torch.zeros(input_size)
        self.zeros_hidden = torch.zeros(hidden_size)

        self.reset_parameters()
      
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    @jit.script_method
    def forward(self, X_in, X_mask, X_delta, X_last, X_mean, hidden):
        
        X_in = X_in.view(-1, X_in.size(1))
        X_mask = X_mask.view(-1, X_mask.size(1))
        X_delta = X_delta.view(-1, X_delta.size(1))
        X_last = X_last.view(-1, X_last.size(1))
        X_mean = X_mean.view(-1, X_mean.size(1))

        gamma_x = torch.exp(-1.*torch.maximum(self.zeros_in, self.gamma_x(X_delta)))

        X = (X_mask * X_in) + (1 - X_mask) * gamma_x * X_last + (1 - X_mask) * ( 1 - gamma_x) * X_mean
        
        gamma_h = torch.exp(-1.*torch.maximum(self.zeros_hidden, self.gamma_h(X_delta)))
        
        hidden = torch.squeeze(gamma_h * hidden)
        
        x_combined = self.x_combined(X)
        x_z, x_r, x_n = x_combined.chunk(3, 1)
        
        h_combined = self.h_combined(hidden)
        h_z, h_r = h_combined.chunk(2, 1)
        
        m_combined = self.m_combined(X_mask)
        m_z, m_r, m_n = m_combined.chunk(3, 1)
        
        z_gate = torch.sigmoid(x_z + h_z + m_z)
        r_gate = torch.sigmoid(x_r + h_r + m_r)
        
        h_n = self.h_n(r_gate * hidden)
        
        h_tilde = torch.tanh(x_n + h_n + m_n)
        
        h_t = (1 - z_gate) * hidden + z_gate * h_tilde

        return h_t, X
    

class GRUDmodel(jit.ScriptModule):
    def __init__(self, input_dim, hidden_dim, device, reverse=False, bias=True):
        super(GRUDmodel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.reverse = reverse
        
        self.gru_cell = GRUDcell(input_dim, hidden_dim)
        
    @jit.script_method 
    def forward(self, X_in, X_mask, X_delta, X_last, X_mean):
        hn = torch.zeros(X_in.size(0), self.hidden_dim).to(self.device)

        hn_out = []
        X_out = []
        for i in range(X_in.size(1)):
            hn, X = self.gru_cell(
                        X_in=X_in[:, i],
                        X_mask=X_mask[:, i],
                        X_delta=X_delta[:, i],
                        X_last=X_last[:, i],
                    X_mean=X_mean[:, i],
                    hidden=hn)
            hn_out.append(hn.unsqueeze(1))
            X_out.append(X)
            
        hn_out = torch.cat(hn_out, dim=1)
        X_out = torch.cat(X_out, dim=-1)
        return hn_out, X_out
