import torch
import numpy as np
import torch.nn.functional as F
import math
import gc
import dgl


class InfoNCE(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        # TODO: add temperature
        self.register_buffer("temperature", torch.tensor(temperature))

    def forward(self, graph, feat):
        """Neighborhood contrastive loss"""
        
        # gets connnected graph
        isolated_nodes =((graph.in_degrees() == 0) & (graph.out_degrees() == 0)).nonzero().squeeze(1)
        contr_graph = dgl.remove_nodes(graph, isolated_nodes)

        pos_pairs = torch.stack(
            dgl.sampling.sample_neighbors(
                contr_graph, list(contr_graph.nodes()), 15, replace=True
            ).edges()
        ).T
        
        nodes_samp = pos_pairs[:,1].unique()[:100]
        pos_pairs = pos_pairs[torch.isin(pos_pairs[:,1], nodes_samp)]
        contr_feat =  feat

        norm_feat = (contr_feat - contr_feat.min(0).values) / (
            contr_feat.max(0).values - contr_feat.min(0).values
        )
        
        # CHUNK matrix multiplicationn        
        pos_pairs_expand = pos_pairs[:,0].unsqueeze(1).reshape(nodes_samp.shape[0], 15)
        pos_feats = norm_feat[pos_pairs_expand]
        
        numerator = torch.exp(torch.matmul(pos_feats, norm_feat[nodes_samp].unsqueeze(2)))

        denominator = torch.sum(torch.exp(torch.matmul(norm_feat[pos_pairs_expand.unique()], norm_feat[pos_pairs_expand.unique()].unsqueeze(-1))))

        loss = -torch.log(numerator/denominator).sum(1).mean()

        return loss
