import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import gc

import dgl
import dgl.nn as dglnn


class graph_encoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(graph_encoder, self).__init__()
        self.backbone = dglnn.GATConv(
            x_dim, 128, 8, feat_drop=0.0, attn_drop=0.1, residual=True
        )
        self.dropout = nn.Dropout(p=0.1)
        self.act = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.mu = nn.Linear(128, z_dim)
        torch.nn.init.xavier_uniform_(self.mu.weight)
        self.log_sigma = nn.Linear(128, z_dim)
        torch.nn.init.xavier_uniform_(self.log_sigma.weight)

    def forward(self, graph, x):
        h = self.dropout(
            self.bn1(self.act(self.backbone(dgl.add_self_loop(graph), x).sum(1)))
        )

        mu = self.mu(h)
        log_sigma = self.log_sigma(h).clamp(-10, 10)

        return mu, log_sigma


class encoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(encoder, self).__init__()

        self.f1 = nn.Linear(x_dim, 128)
        # torch.nn.init.xavier_uniform_(self.f1.weight)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout(p=0.1)

        self.mu = nn.Linear(128, z_dim)
        self.log_sigma = nn.Linear(128, z_dim)

    def forward(self, x):
        h = self.dropout(self.bn1(self.act(self.f1(x))))
        mu = self.mu(h)
        log_sigma = self.log_sigma(h).clamp(-10, 10)
        return mu, log_sigma


class decoder_patchwork(nn.Module):
    def __init__(self, mod_dims, z_dim, emb_dim, num_batch, domains):
        super(decoder_patchwork, self).__init__()

        self.domains = domains
        self.alpha = nn.Parameter(torch.randn(z_dim, emb_dim))

        self.rhos = nn.ParameterList(
            [nn.Parameter(torch.randn(mod_dim, emb_dim)) for mod_dim in mod_dims]
        )

        self.betas = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(z_dim, emb_dim))
                for i in range(len(self.domains))
            ]
        )

        self.mod_batch_bias = nn.ModuleList([])

        for domain in self.domains:
            self.mod_batch_bias.append(
                nn.ParameterList(
                    [nn.Parameter(torch.zeros(1, mod_dim)) for mod_dim in mod_dims]
                )
            )

        self.Topic_mod1, self.Topic_mod2 = None, None
        self.shared_embedding = self.alpha

    def add_ft_params(self, domains_add):
        """Add domain-specific parameters for fine-tuning"""

        self.domains = torch.cat((self.domains, domains_add))

        for domain in domains_add:
            self.mod_batch_bias.append(
                nn.ParameterList(
                    [
                        nn.Parameter(
                            nn.init.xavier_uniform(
                                torch.randn(1, self.rhos[i].shape[0])
                            )
                        ).cuda()
                        for i in range(len(self.rhos))
                    ]
                )
            )

            self.betas.append(nn.Parameter(torch.zeros(self.betas[0].shape)).cuda())

    def forward(
        self, theta, batch_list, domains=None, cross_prediction=False, impute=False
    ):
        domains = torch.cat([i for i in batch_list]).unique()

        # reconstruction for each modality
        recons = [[] for i in range(len(batch_list) - 1)]

        for i in np.array([len(j) > 0 for j in batch_list[1:]]).nonzero()[0]:
            recon_mod_domains, all_batches = [], []
            mod_domains = batch_list[i + 1].unique()
            for domain in mod_domains:
                Topic_mod = torch.mm(
                    self.alpha + self.betas[domain], self.rhos[i].t()
                ).t()

                # assumes that first batch list contains full batch info
                recon_mod_domain = torch.mm(
                    theta[(batch_list[0] == domain)],
                    Topic_mod.t(),
                )

                if impute is False:
                    recon_mod_domain += self.mod_batch_bias[domain][i]

                if cross_prediction == False:
                    recon_mod_domain = F.log_softmax(recon_mod_domain, dim=-1)
                else:
                    recon_mod_domain = F.softmax(recon_mod_domain, dim=-1)

                all_batches.append(((batch_list[0] == domain)).nonzero().T[0])
                recon_mod_domains.append(recon_mod_domain)
                del recon_mod_domain, Topic_mod

            all_batches = torch.cat(all_batches, dim=0)
            batches_sort = torch.argsort(all_batches)
            recon_mod = torch.cat(recon_mod_domains, dim=0)

            recons[i] = recon_mod[batches_sort]

        return recons


class decoder(nn.Module):
    def __init__(self, mod_dims, z_dim, emb_dim, num_batch, domains):
        super(decoder, self).__init__()
        self.domains = domains

        self.alpha_mods = nn.ParameterList(
            [nn.Parameter(torch.randn(mod_dim, emb_dim)) for mod_dim in mod_dims]
        )

        self.beta = nn.Parameter(torch.randn(z_dim, emb_dim))

        # domain x modality
        self.mod_batch_bias = nn.ModuleList([])

        for domain in self.domains:
            self.mod_batch_bias.append(
                nn.ParameterList(
                    [nn.Parameter(torch.randn(1, mod_dim)) for mod_dim in mod_dims]
                )
            )

        self.Topic_mods = [None for i in range(len(mod_dims))]
        self.shared_embedding = self.beta

    def add_ft_params(self, domains_add):
        """Add domain-specific parameters for fine-tuning"""

        self.domains = torch.cat((self.domains, domains_add))

        for domain in domains_add:
            self.mod_batch_bias.append(
                nn.ParameterList(
                    [
                        nn.Parameter(torch.randn(1, self.alpha_mods[i].shape[0])).cuda()
                        for i in range(len(self.alpha_mods))
                    ]
                )
            )

    def forward(
        self, theta, batch_list, domains=None, cross_prediction=False, impute=False
    ):
        domains = torch.cat([i for i in batch_list]).unique()

        # reconstruction for each modality
        recons = [[] for i in range(len(batch_list) - 1)]

        for i in np.array([len(j) > 0 for j in batch_list[1:]]).nonzero()[0]:
            Topic_mod = torch.mm(self.alpha_mods[i], self.beta.t()).t()
            recon_mod_domains, all_batches = [], []
            mod_domains = batch_list[i + 1].unique()
            for domain in mod_domains:
                # assumes that first batch list contains full batch info
                recon_mod_domain = torch.mm(
                    theta[(batch_list[0] == domain)],
                    Topic_mod,
                )
                if impute is False:
                    recon_mod_domain += self.mod_batch_bias[domain][i]

                if cross_prediction == False:
                    recon_mod_domain = F.log_softmax(recon_mod_domain, dim=-1)
                else:
                    recon_mod_domain = F.softmax(recon_mod_domain, dim=-1)

                all_batches.append(((batch_list[0] == domain)).nonzero().T[0])
                recon_mod_domains.append(recon_mod_domain)
                del recon_mod_domain

            del Topic_mod
            all_batches = torch.cat(all_batches, dim=0)
            batches_sort = torch.argsort(all_batches)
            recon_mod = torch.cat(recon_mod_domains, dim=0)

            recons[i] = recon_mod[batches_sort]
            del batches_sort, all_batches

        return recons
