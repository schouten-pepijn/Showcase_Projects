class mTAN(nn.Module):
    def __init__(self, hp):
        super(mTAN, self).__init__()
        
        # network parameters
        self.hp = hp
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

        # full flip angle range
        if self.mh_att:
            print('using multihead attention')
            
            self.query = torch.linspace(0, 1, 128).to(self.hp.device)

            # attention layer
            self.attention = MultiTimeAttention(d_input=self.dim,
                                                d_hidden=self.nhidden,
                                                d_time=self.embed_time,
                                                num_heads=self.num_heads,
                                                dropout_p=0.)
    
            # rnn layer
            self.rnn = nn.GRU(input_size=self.nhidden, 
                              hidden_size=self.nhidden,
                              num_layers=1,
                              dropout=0,
                              bidirectional=self.gru_bi,
                              batch_first=True) 
            
            self.time_embedding = TimeEmbedding(hp=self.hp, 
                                                in_features=1,
                                                out_features=self.embed_time,
                                                dropout_p=self.dropout_p)

        # uni/bi directional
        if self.gru_bi:
            hidden_reg = 2*self.nhidden
        else:
            hidden_reg = self.nhidden
        
        self.score_T10 = nn.Sequential(nn.Linear(hidden_reg, 1,
                                                 bias=False),
                                        nn.Softmax(dim=1))
        self.score_M0 = nn.Sequential(nn.Linear(hidden_reg, 1,
                                                 bias=False),
                                        nn.Softmax(dim=1))
        
        self.reg_T10 = nn.Sequential(nn.Dropout(self.dropout_p),
                                     nn.Linear(int(hidden_reg), 200),
                                     nn.GELU(),
                                     nn.Dropout(self.dropout_p),
                                     nn.Linear(200, 200),
                                     nn.GELU(),
                                     nn.Linear(200, 1))
        self.reg_M0 = nn.Sequential(nn.Dropout(self.dropout_p),
                                     nn.Linear(int(hidden_reg), 200),
                                     nn.GELU(),
                                     nn.Dropout(self.dropout_p),
                                     nn.Linear(200, 200),
                                     nn.GELU(),
                                     nn.Linear(200, 1))


    def forward(self, X_fa_in, fa_vals, fa_mask, fa_len, TR_vals):

        mean = torch.sum(X_fa_in, dim=1) / fa_len.squeeze()
        X_fa = X_fa_in


        key_input = fa_vals / math.radians(self.hp.xFA[2][1] - 1)
        
        X_fa = torch.cat((X_fa.unsqueeze(dim=-1),
                          fa_mask.unsqueeze(dim=-1).float()), dim=-1)

        key = self.time_embedding(key_input).to(self.hp.device)
        query = self.time_embedding(self.query.unsqueeze(dim=0))

        out = self.attention(query, key, X_fa, mask=fa_mask) 
        
        out, _ = self.rnn(out)

        hidden_T10 = torch.sum(out*self.score_T10(out), dim=1)
        hidden_M0 = torch.sum(out*self.score_M0(out), dim=1)
            
        # regress T10 value
        T10 = self.reg_T10(hidden_T10).squeeze()
        M0 = self.reg_M0(hidden_M0).squeeze()

        # # update T10
        T10_diff = self.hp.simulations.T10bounds[1] - self.hp.simulations.T10bounds[0]
        T10 = self.hp.simulations.T10bounds[0] + torch.sigmoid(T10.unsqueeze(1)) * T10_diff
        M0_diff = self.hp.simulations.M0bounds[1] - self.hp.simulations.M0bounds[0]
        M0 = self.hp.simulations.M0bounds[0] + torch.sigmoid(M0.unsqueeze(1)) *  M0_diff

        R1 = 1 / T10
        X_out = torch.mul(
            (1 - torch.exp(torch.mul(-TR_vals, R1))), torch.sin(fa_vals)) / (
            1 - torch.mul(torch.cos(fa_vals), torch.exp(torch.mul(-TR_vals, R1))))

        X_out = X_out * M0
        return X_out, T10, M0



class MultiTimeAttention(nn.Module):
    def __init__(self, d_input, d_hidden=16, d_time=4,
                 num_heads=1, dropout_p=None, layer_bool=None):
        super(MultiTimeAttention, self).__init__()
        assert d_time % num_heads == 0
        
        # network parameters
        self.d = d_time
        self.d_k = d_time // num_heads
        self.h = num_heads
        self.dim = d_input
        self.d_h = d_hidden
        self.dropout_p = dropout_p
        self.layer_bool = layer_bool

        # linear layers
        self.k_linear = nn.Linear(d_time, d_time)
        self.q_linear = nn.Linear(d_time, d_time)
        self.out = nn.Linear(d_input*num_heads, d_hidden)
        
        if dropout_p is not None:
            self.dropout = nn.Dropout(dropout_p)


    def create_mask(self, mask, dim_in):
        mask = mask.unsqueeze(dim=-1)
        mask = torch.repeat_interleave(mask, repeats=dim_in, dim=-1)
        mask = mask.unsqueeze(dim=1)
        mask = ~mask
        return mask


    def attention(self, q, k, v, mask=None):
        # value/query size
        d_v, d_q = v.size(dim=-1), q.size(dim=-1)

        # scaled dot product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_q)
        scores = scores.unsqueeze(dim=-1).repeat_interleave(d_v, dim=-1)

        # update scores with padding mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(dim=-3), -1e9)

        # normalize scoring to probabilities
        p_scores = F.softmax(scores, dim=-2)

        # apply dropout
        if self.dropout_p is not None:
            p_scores = self.dropout(p_scores)

        output = torch.sum(p_scores*v.unsqueeze(dim=-3), dim=-2)
        return output


    def forward(self, q, k, v, mask=None):
        bs, seq_len, dim_v = v.size()
        v = v.unsqueeze(1)

        if mask is not None:
            mask = self.create_mask(mask, dim_v)

        q = self.q_linear(q).view(q.size(dim=0), -1, self.h, self.d_k)
        k = self.k_linear(k).view(k.size(dim=0), -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        scores = self.attention(q, k, v, mask)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.h*dim_v)

        return self.out(concat)


class TimeEmbedding(nn.Module):
    def __init__(self, hp, in_features, out_features, dropout_p=0.1):
        super(TimeEmbedding, self).__init__()

        self.hp = hp
        self.periodic = nn.Linear(in_features, out_features-1)
        self.linear = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, t):
        t = t.unsqueeze(dim=-1).to(self.hp.device)
        out_linear = self.linear(t)
        out_periodic = self.periodic(t).sin()
        out = torch.cat((out_linear, out_periodic), dim=-1)
        return self.dropout(out)
    return a
