import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable
import time
from mo_utils import calc_weight
from eval_utils import evaluate
import scipy
import scipy.io as sio
import pandas as pd
import os
import gc
from contrastive import InfoNCE
import dgl
import itertools
from torch.utils.tensorboard import SummaryWriter
from trainer_moetm import Trainer_moETM
from layers import encoder, decoder, graph_encoder, decoder_patchwork
import gc
import sklearn

import scanpy as sc
import anndata as ad


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class Trainer_moETM_patchwork(Trainer_moETM):
    def __init__(
        self,
        num_cells,
        encoders,
        decoder,
        optimizer,
        exp="",
        run_mode="",
        lr=1e-3,
        logging_interval=500,
    ):
        super().__init__(
            num_cells, encoders, decoder, optimizer, exp, run_mode, lr, logging_interval
        )
        self.label_binarizer = sklearn.preprocessing.LabelBinarizer()

    def init_priors(self, domains):
        self.mu_priors = nn.ParameterList([])
        self.logsigma_priors = nn.ParameterList([])

        for d in domains:
            mu_prior, logsigma_prior = self.prior_expert(
                (1, 1, 100), use_cuda=True, requires_grad=True
            )
            self.mu_priors.append(mu_prior)
            self.logsigma_priors.append(logsigma_prior)

    def add_ft_params(self, domains_add, num_ft_cells):
        """
        Adds fine-tuning parameters to the trainer.

        Args:
            domains_add (Tensor): The domains to add.
        """
        self.train_domains = self.domains
        self.test_domains = domains_add
        self.domains = torch.cat((self.domains, domains_add))
        assert torch.equal(
            torch.unique(self.domains), self.domains
        ), "Fine-tuning domains overlap with training domains"
        self.decoder.add_ft_params(domains_add)

        for d in self.test_domains:
            prior_add = self.prior_expert(
                (1, 1, 100), use_cuda=True, requires_grad=True
            )
            self.mu_priors.append(prior_add[0])
            self.logsigma_priors.append(prior_add[1])

        for module in self.decoder.mod_batch_bias[len(self.train_domains) :]:
            toogle_grad(module, requires_grad=True)
            self.optimizer.add_param_group({"params": module.parameters()})

        for d in self.test_domains:
            self.optimizer.add_param_group({"params": self.mu_priors[d]})
            self.optimizer.add_param_group({"params": self.logsigma_priors[d]})
            self.optimizer.add_param_group({"params": self.decoder.betas[d]})

        LIST = np.arange(0, num_ft_cells)
        np.random.shuffle(LIST)
        self.TRAIN_LIST = LIST[: int(0.8 * LIST.shape[0])]
        self.VAL_LIST = LIST[int(0.8 * LIST.shape[0]) :]

        self.decoder.cuda()

    def set_mode(self, mode):
        """Toggles training/eval"""
        if mode == "train":
            self.encoders.train()
            self.decoder.train()
            self.mu_priors.train()
            self.logsigma_priors.train()

        elif mode == "eval":
            self.encoders.eval()
            self.decoder.eval()
            self.mu_priors.eval()
            self.logsigma_priors.eval()

    def get_NLL_recons(self, xs, preds, batch_index, batch_list):
        domains = batch_list[0].unique()

        nll_mods = []

        for i, pred in enumerate(preds):
            datasets = torch.tensor(
                self.label_binarizer.fit_transform(
                    self.dataloader.dataset_idx[i].tolist()
                ).T[0][batch_index[i + 1].detach().cpu().numpy()]
            ).cuda()

            nlls_dataset = [[] for j in range(len(datasets.unique()))]
            for domain in domains:
                if domain not in batch_list[i + 1]:
                    continue
                domain_indices = (batch_list[i + 1] == domain).nonzero().T[0]
                nlls_dataset[datasets[domain_indices].unique()[0]].append(
                    (-pred[domain_indices] * xs[i][domain_indices]).sum(-1).mean()
                    if xs[i][domain_indices].shape[0] != 0
                    else torch.tensor(0.0).cuda()
                )

            # sum over domains
            nlls = torch.stack(
                [torch.stack(nlls_dataset[j]).sum() for j in range(len(nlls_dataset))]
            )

            # average over datasets
            nll_mods.append(
                (nlls * datasets.unique(return_counts=True)[1].float()).sum()
                / datasets.shape[0]
            )

        return nll_mods

    def train(self, xs, batch_index, batch_list, hyperparams, graphs=[None], log=False):
        KL_weight, beta_weight, ncl_weight = (
            hyperparams["kl"],
            hyperparams["beta"],
            hyperparams["ncl"],
        )

        # topic embeddings
        if graphs[0] is None:
            mu, log_sigma, gcontr_losses = self.get_embed(
                xs, batch_index, batch_list, log=log
            )
        else:
            mu, log_sigma, gcontr_losses = self.get_embed(
                xs, batch_index, batch_list, graphs=graphs, log=log
            )

        Theta = F.softmax(
            self.reparameterize(mu, log_sigma), dim=-1
        )  # log-normal distribution

        # decoding
        recon_log_mods = self.decoder(Theta, batch_list)

        # loss formulation
        nll_mods = self.get_NLL_recons(
            xs,
            recon_log_mods,
            batch_index,
            batch_list,
        )

        print("Theta max:", Theta.max())

        KL = self.get_kl(mu, log_sigma, batch_index, batch_list[0])

        gcontr_loss = (
            0 if gcontr_losses[0] is None else torch.stack(gcontr_losses).sum()
        )

        Loss = KL_weight * KL

        for i in range(len(nll_mods)):
            Loss += nll_mods[i].mean()

        Loss += gcontr_loss * ncl_weight

        for i in range(len(self.decoder.betas)):
            beta_loss = (
                torch.norm(self.decoder.betas[i], 2)
                if i == 0
                else beta_loss + torch.norm(self.decoder.betas[i], 2)
            )

        Loss += beta_loss * beta_weight

        return (
            Loss,
            [nll_mod for nll_mod in nll_mods],
            KL_weight * KL.item(),
            gcontr_losses,
            (beta_loss * beta_weight),
        )

    def reset_optimizer(self):
        PARA = [
            {"params": self.encoders.parameters()},
            {"params": self.decoder.parameters()},
        ]

        for d in self.domains:
            PARA.append({"params": self.mu_priors[d]})
            PARA.append({"params": self.logsigma_priors[d]})

        self.optimizer = optim.Adam(PARA, lr=self.lr)

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma, batch_index, batch_list):
        """Calculate KL(q||p) where q = Normal(mu, sigma) and p = Normal(mu_p, sigma_p)
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma) and p = Normal(mu_p, sigma_p)
        """

        # calculate kl between posterior distributions per domain and domain-specific priors
        datasets = torch.tensor(
            self.label_binarizer.fit_transform(
                self.dataloader.dataset_idx[0].tolist()
            ).T[0][batch_index[0].detach().cpu().numpy()]
        ).cuda()

        mus = [mu[batch_list == domain] for domain in self.domains]
        logsigmas = [logsigma[batch_list == domain] for domain in self.domains]

        kl_losses = 0

        for d_idx, d in enumerate(batch_list.unique()):
            kl_loss = self.logsigma_priors[d] - logsigmas[d]
            kl_loss += (
                ((2 * logsigmas[d]).exp() + (mus[d] - self.mu_priors[d]).pow(2))
                / (2 * (2 * self.logsigma_priors[d]).exp())
            ) - (1 / 2)
            kl_losses += kl_loss.sum(-1).mean()

        if self.kl_setting == "new":
            return kl_losses / len(batch_list.unique())
        else:
            return kl_losses

    def get_embed(
        self,
        xs,
        batch_index,
        batch_list,
        domains=None,
        graphs=[None],
        eval=False,
        embed_type="topic",
        log=False,
    ):
        """Calculate topic embeddings"""
        gcontr_loss = [None]
        mu, log_sigma, gcontr_loss = self.encode(
            xs, batch_index, batch_list, graphs=graphs, log=log
        )
        if embed_type == "topic":
            return mu, log_sigma, gcontr_loss
        elif embed_type == "shared":
            Theta = F.softmax(
                self.reparameterize(mu, log_sigma), dim=-1
            )  # log-normal distribution
            domain_embed = (self.decoder.alpha).detach().cpu().numpy()
            domains = batch_list[0].unique()
            topic_embed = Theta.detach().cpu().numpy()
            embed = np.zeros((topic_embed.shape[0], domain_embed.shape[-1]))
            full_batch_list = batch_list[0].detach().cpu().numpy()

            for domain in domains:
                embed[full_batch_list == domain.item()] = (
                    topic_embed[full_batch_list == domain.item()] @ (domain_embed)
                    + self.decoder.betas[domain].detach().cpu().numpy()[0]
                )

            return embed

    def encode(
        self,
        xs,
        batch_index,
        batch_list,
        domains=None,
        graphs=[None],
        graph_contrast=True,
        log=False,
    ):
        """Helper for get_embed()"""

        if domains is None:
            domains = self.domains

        gcontr_losses = [None]
        if graphs[0] is not None and False:
            results = zip(
                *[
                    self.encoders[i](graphs[i], xs[i]) if len(xs[i]) > 0 else ([], [])
                    for i in range(len(xs))
                ]
            )
            mu_mods, log_sigma_mods = map(list, results)
        else:
            results = zip(
                *[
                    self.encoders[i](xs[i]) if len(xs[i]) > 0 else ([], [])
                    for i in range(len(xs))
                ]
            )
            mu_mods, log_sigma_mods = map(list, results)
        if graphs[0] is not None:
            gcontr_losses = []

        for ind, i in enumerate(mu_mods):
            if len(i) == 0:
                continue
            if log == True:
                self.writer.add_scalar(
                    f"norms_{self.mode}/mu_norm_{ind}", torch.norm(i, 2), self.step
                )
                self.writer.add_scalar(
                    f"norms_{self.mode}/log_sigma_norm_{ind}",
                    torch.norm(log_sigma_mods[ind], 2),
                    self.step,
                )

            if "graphbefore" in self.exp:
                if graphs[0] is not None:
                    embed = F.softmax(
                        self.reparameterize(mu_mods[ind], log_sigma_mods[ind]), dim=-1
                    )
                    if graph_contrast:
                        if graphs[ind].number_of_nodes() > 0:
                            gcontr_losses.append(
                                self.contrastive_loss(
                                    graphs[ind],
                                    embed,
                                ).mean()
                            )
                        else:
                            gcontr_losses.append(None)

        # product of gaussian among available modalities for each domain
        all_mu, all_log_sigma, all_batches = [], [], []

        for domain_id in torch.cat([i for i in batch_list]).unique():
            # available samples in modality, index those from domain_id
            mu_mods_include = []
            log_sigma_mods_include = []

            mods_avail = [
                i - 1 for i in range(1, len(batch_list)) if domain_id in batch_list[i]
            ]

            for i in mods_avail:
                domain_indices_mod = (batch_list[i + 1] == domain_id).nonzero().T[0]
                mu_mods_include.append(mu_mods[i][domain_indices_mod].unsqueeze(0))
                log_sigma_mods_include.append(
                    log_sigma_mods[i][domain_indices_mod].unsqueeze(0)
                )
            try:
                mu_mods_include = torch.cat(mu_mods_include, dim=0)
                log_sigma_mods_include = torch.cat(log_sigma_mods_include, dim=0)

                mu_prior, logsigma_prior = self.prior_expert(
                    (1, domain_indices_mod.shape[0], mu_mods[i].shape[1]), use_cuda=True
                )
            except:
                import ipdb

                ipdb.set_trace()

            try:
                Mu = torch.cat(
                    (
                        mu_prior,
                        mu_mods_include,
                    ),
                    dim=0,
                )
            except:
                import ipdb

                ipdb.set_trace()

            Log_sigma = torch.cat(
                (
                    logsigma_prior,
                    log_sigma_mods_include,
                ),
                dim=0,
            )

            mu, log_sigma = self.experts(Mu, Log_sigma)

            assert (
                mu.shape[0] == log_sigma.shape[0]
            ), "Mean/std embedding shapes do not match"

            all_mu.append(mu)
            all_log_sigma.append(log_sigma)
            all_batches.append((batch_list[0] == domain_id).nonzero())

            del Mu, Log_sigma, mu_mods_include, log_sigma_mods_include, mu, log_sigma

        # cell topic embeddings collected for each domain
        try:
            all_mu = torch.cat(all_mu, dim=0)
        except:
            import ipdb

            ipdb.set_trace()
        all_log_sigma = torch.cat(all_log_sigma, dim=0)
        all_batches = torch.cat(all_batches, dim=0).T[0]

        batches_sort = torch.argsort(all_batches)
        all_mu = all_mu[batches_sort]
        all_log_sigma = all_log_sigma[batches_sort]

        if "graphafter" in self.exp:
            if graphs[0] is not None:
                embed = F.softmax(self.reparameterize(all_mu, all_log_sigma), dim=-1)

                for i in range(len(graphs)):
                    if graphs[i].number_of_nodes() == 0:
                        gcontr_losses.append(None)
                        continue
                    if not torch.equal(batch_index[0], batch_index[i + 1]):
                        mask = torch.isin(
                            batch_index[0],
                            self.dataloader.batch_index_mods[i + 1][batch_index[i + 1]],
                        )
                    else:
                        mask = torch.isin(batch_index[0], batch_index[i + 1])
                    gcontr_losses.append(
                        self.contrastive_loss(
                            graphs[i],
                            embed[mask],
                        ).mean()
                    )

        del batches_sort
        # gc.collect()

        return all_mu, all_log_sigma, gcontr_losses

    def load_params(self, outdir, ckpt_key):
        super().load_params(outdir, ckpt_key)
        self.mu_priors.load_state_dict(
            torch.load(f"{outdir}/moetm_mu_priors_ckpt{ckpt_key}.pth")
        )
        self.logsigma_priors.load_state_dict(
            torch.load(f"{outdir}/moetm_logsigma_priors_ckpt{ckpt_key}.pth")
        )

    def save_params(self, outdir, key, epoch, prev_ckpt=0):
        super().save_params(outdir, key, epoch, prev_ckpt)
        if os.path.exists(f"{outdir}/moetm_mu_priors_ckpt{key}_{prev_ckpt}.pth"):
            os.remove(f"{outdir}/moetm_mu_priors_ckpt{key}_{prev_ckpt}.pth")
            os.remove(f"{outdir}/moetm_logsigma_priors_ckpt{key}_{prev_ckpt}.pth")

        torch.save(
            self.mu_priors.state_dict(),
            f"{outdir}/moetm_mu_priors_ckpt{key}_{epoch}.pth",
        )
        torch.save(
            self.logsigma_priors.state_dict(),
            f"{outdir}/moetm_logsigma_priors_ckpt{key}_{epoch}.pth",
        )
