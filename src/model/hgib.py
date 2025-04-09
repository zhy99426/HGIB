import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_conv import GraphConvLayer
from .fusion import Fusion

def gumbel_sigmoid(logits, tau = 1, hard = True, threshold = 0.05):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0]] = 1.0
        ret = y_hard * y_soft - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def kernel_matrix(x, sigma):
    return torch.exp((torch.matmul(x, x.transpose(0,1)) - 1) / sigma)    ### real_kernel

def hsic(Kx, Ky, m):
    Kxy = torch.mm(Kx, Ky)
    h = torch.trace(Kxy) / m ** 2 + torch.mean(Kx) * torch.mean(Ky) - \
        2 * torch.mean(Kxy) / m
    return h * (m / (m - 1)) ** 2


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss

class HGIB(nn.Module):
    def __init__(self, data, emb_dim, threshold, beta, sigma, alpha):
        super(HGIB, self).__init__()
        self.edge_dict = data['edge_dict']
        
        self.n_users = data['n_users']
        self.n_items = data['n_items']
        self.emb_dim = emb_dim
        self.temperature = 1
        self.dropout = nn.Dropout(0.1)
        
        self.bsg_types = data['bsg_types']
        self.tcb_types = data['tcb_types'] # target-complemented behaviors
        self.tib_types = data['tib_types'] # target-intersected behaviors
        self.trbg_types = self.tcb_types + self.tib_types
        self.total_behaviors = ['ubg'] + self.bsg_types + self.trbg_types
        
        self.activate = nn.ReLU()
        self.lin = {b: nn.ModuleList([nn.Linear(emb_dim*2, emb_dim),
                                  nn.Linear(emb_dim, 1)]).cuda() for b in self.total_behaviors}
        
        self.beta = beta
        self.alpha = alpha
        
        self.fusion = Fusion(emb_dim, self.bsg_types, self.trbg_types)
        self.ce_loss = nn.CrossEntropyLoss()
        self.reg_loss = EmbLoss()

        self.eps = 0.1
        self.edge_bias = 0.5
        self.sigma = sigma
        self.hsic_loss = torch.tensor(0.0).cuda()
        self.threshold = threshold

        self.user_embedding = nn.Embedding(self.n_users+1, emb_dim, padding_idx=0) # index 0 is padding
        self.item_embedding = nn.Embedding(self.n_items+1, emb_dim, padding_idx=0) # index 0 is padding
        
        self.convs = nn.ModuleDict()
        for behavior_type in self.total_behaviors:
            if behavior_type in self.tcb_types:
                self.convs[behavior_type] = nn.ModuleList([GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(3)])
            else:
                self.convs[behavior_type] = nn.ModuleList([GraphConvLayer(emb_dim, emb_dim, 'gcn') for _ in range(1)])
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        for behavior_type in self.total_behaviors:
            nn.init.xavier_uniform_(self.lin[behavior_type][0].weight)
            nn.init.xavier_uniform_(self.lin[behavior_type][1].weight)
        self.fusion.reset_parameters()
            
    def propagate(self, x, edge_index, behavior_type, target_emb=None, weight=None):
        result = [x]
        for i, conv in enumerate(self.convs[behavior_type]):
            x = conv(x, edge_index, target_emb, weight=weight)
            x = F.normalize(x, dim=1)
            result.append(x)
        result = torch.stack(result, dim=1)
        x = result.sum(dim=1)
        return x
    
    def graph_learner(self, behavior_type, adj_matrix, emb_table):

        row, col = adj_matrix[0], adj_matrix[1]
        row_emb = emb_table[row]
        col_emb = emb_table[col]

        logit = torch.sum(row_emb * col_emb, -1)
        logit = logit.view(-1)
        weights = gumbel_sigmoid(logit, tau=1, threshold=self.threshold)+1e-7

        return weights
    
    def forward(self):
        edge_dict = self.edge_dict
        emb_dict = dict()

        ## Unified behavior graph aggregation ##
        init_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        emb_dict['init'] = init_emb
        weight = self.graph_learner('ubg', edge_dict['ubg'], init_emb)
        ubg_emb = self.propagate(init_emb, edge_dict['ubg'], 'ubg', weight=weight)
        emb_dict['ubg'] = ubg_emb
        
        ## Behavior-speicfic graph aggregation ##
        for behavior_type in self.bsg_types:
            previous_emb = ubg_emb
            weight = self.graph_learner(behavior_type, edge_dict[behavior_type], emb_dict["ubg"])
            bsg_emb = self.propagate(self.dropout(previous_emb), edge_dict[behavior_type], behavior_type, weight=weight)
            emb_dict[behavior_type] = bsg_emb
        
        ## Target-related behavior graph aggregation ##
        # Target-intersected behavior graph aggregation
        for behavior_type in self.tib_types:
            if 'buy' in behavior_type:
                previous_behavior = behavior_type.split('_')[0] # view or cart or collect
                weight = self.graph_learner(behavior_type, edge_dict[behavior_type], emb_dict[previous_behavior])
                # import pdb;pdb.set_trace()
                previous_emb = emb_dict[previous_behavior]
                tib_emb = self.propagate(self.dropout(previous_emb), edge_dict[behavior_type], behavior_type, weight=weight)
                emb_dict[behavior_type] = tib_emb

        for behavior_type in self.tcb_types:
            if 'buy' in behavior_type:
                previous_behavior = behavior_type.split('_')[0] # view or cart or collect
                weight = self.graph_learner(behavior_type, edge_dict[behavior_type], emb_dict[previous_behavior])
                previous_emb = emb_dict[previous_behavior]
                target_emb = emb_dict['buy']
                tcb_emb = self.propagate(self.dropout(previous_emb), edge_dict[behavior_type], behavior_type, target_emb, weight=weight)
                emb_dict[behavior_type] = tcb_emb
                
        final_emb = self.fusion(emb_dict)

        emb_dict['final'] = final_emb

        return emb_dict
    
    def hsic_graph(self, users_emb1, items_emb1, users_emb2, items_emb2):
        ### user part ###
        input_x = users_emb1
        input_y = users_emb2
        input_x = F.normalize(input_x, p=2, dim=1)
        input_y = F.normalize(input_y, p=2, dim=1)
        Kx = kernel_matrix(input_x, self.sigma)
        Ky = kernel_matrix(input_y, self.sigma)
        loss_user = hsic(Kx, Ky, users_emb1.shape[0])
        ### item part ###
        input_i = items_emb1
        input_j = items_emb2
        input_i = F.normalize(input_i, p=2, dim=1)
        input_j = F.normalize(input_j, p=2, dim=1)
        Ki = kernel_matrix(input_i, self.sigma)
        Kj = kernel_matrix(input_j, self.sigma)
        loss_item = hsic(Ki, Kj, users_emb1.shape[0])
        loss = loss_user + loss_item
        return loss
    

    def loss(self, users, pos_idx, neg_idx):
        emb_dict = self.forward()
        user_emb, item_emb = torch.split(emb_dict['final'], [self.n_users+1, self.n_items+1], dim=0)
        self.pt_loss = torch.tensor(0.0).cuda()
        for behavior in self.total_behaviors+["init"]:
            user, item = torch.split(emb_dict[behavior], 
                                    [self.n_users+1, self.n_items+1], dim=0)

            emb_dict[behavior] = {'user': user, 'item': item}

        self.pt_loss += self.beta * self.hsic_graph(emb_dict["ubg"]["user"][users], emb_dict["ubg"]["item"][pos_idx], emb_dict["init"]["user"][users], emb_dict["init"]["item"][pos_idx])
        
        for behavior in self.bsg_types:
            self.pt_loss += self.beta *  self.hsic_graph(emb_dict[behavior]["user"][users], emb_dict[behavior]["item"][pos_idx], emb_dict["ubg"]["user"][users], emb_dict["ubg"]["item"][pos_idx])
            
        for behavior in self.trbg_types:
            self.pt_loss += self.beta * self.hsic_graph(emb_dict[behavior]["user"][users], emb_dict[behavior]["item"][pos_idx], emb_dict[behavior.split('_')[0]]["user"][users], emb_dict[behavior.split('_')[0]]["item"][pos_idx])

        for behavior in self.bsg_types+["ubg"]:
            self.pt_loss += self.alpha * self.cl_loss(user_emb[users], emb_dict[behavior]["user"][users])
            self.pt_loss += self.alpha * self.cl_loss(item_emb[pos_idx], emb_dict[behavior]["item"][pos_idx])
        
        
        reg_loss = self.reg_loss(user_emb, item_emb, require_pow=True)
        logits = torch.matmul(user_emb[users], item_emb.transpose(0, 1))

        return self.ce_loss(logits, pos_idx)+self.pt_loss+0.1*reg_loss
    
    
    def cl_loss(self, x1, x2):
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).mean()
    

    def predict(self, users):
        final_embeddings = self.forward()['final']
        final_user_emb, final_item_emb = torch.split(final_embeddings, [self.n_users + 1, self.n_items + 1])

        user_emb = final_user_emb[users.long()]
        scores = torch.matmul(user_emb, final_item_emb.transpose(0, 1))
        return scores