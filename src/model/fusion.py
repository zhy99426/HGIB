import torch
import torch.nn as nn

class Fusion(nn.Module):
    def __init__(self, emb_dim, bsg_types, trbg_types):
        super(Fusion, self).__init__()
        self.emb_dim = emb_dim
        self.bsg_types = bsg_types
        self.trbg_types = trbg_types
        self.proj_layers = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim, bias=True),
            nn.Linear(emb_dim, emb_dim, bias=True)
        ])
        
    def reset_parameters(self):
        for layer in self.proj_layers:
            nn.init.xavier_uniform_(layer.weight)
    def forward(self, emb_dict):
        key = emb_dict['buy'] # [N, D]
        
        for i, behavior_types in enumerate([self.bsg_types, self.trbg_types]):
            proj_key = self.proj_layers[i](key)  # [N, D]
            
            query_embs = torch.stack([emb_dict[bt] for bt in behavior_types], dim=1)  # [N, B, D]
            
            scores = torch.einsum('nd,nbd->nb', proj_key, query_embs) / (self.emb_dim ** 0.5)
            attention = scores.softmax(dim=1).unsqueeze(-1)  # [N, B, 1]
            
            key = (attention * query_embs).sum(dim=1)  # [N, D]

        return key
