import torch
import torch.nn as nn

from retention import MultiScaleRetention

class RetNet_shared(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False):
        super(RetNet_shared, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])

        self.ffns_code = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_nl = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])

        self.ffns_nl = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_code = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
    def forward(self, X, modal):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        assert modal in ['code','nl']
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
           
            if modal=='code':
                X = self.ffns_code[i](self.layer_norms_code[i](Y)) + Y
            elif modal == "nl":
                X = self.ffns_nl[i](self.layer_norms_nl[i](Y)) + Y
        return X


