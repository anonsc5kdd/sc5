import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
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
from torch.utils.tensorboard import SummaryWriter
import glob
import logging
from moetm_utils import *
from layers import encoder, decoder, graph_encoder, decoder_patchwork
from sklearn.decomposition import PCA
import colorcet as cc
from matplotlib.patches import Patch


import scanpy as sc
import anndata as ad

import pickle as pkl


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


class Trainer_moETM(object):
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
        self.encoders = torch.nn.ModuleList([encoder for encoder in encoders]).to(
            "cuda"
        )
        self.decoder = decoder.to("cuda")

        for module in self.encoders:
            toogle_grad(module, requires_grad=True)
        toogle_grad(self.decoder, requires_grad=True)
        for module in self.decoder.mod_batch_bias:
            toogle_grad(module, requires_grad=True)

        self.optimizer = optimizer
        self.contrastive_loss = InfoNCE()
        self.exp = exp
        self.lr = lr
        self.run_mode = run_mode

        self.train_domains, self.test_domains = None, None

        self.best_encoder_mod1 = None
        self.best_encoder_mod2 = None
        self.writer = SummaryWriter(f"test_moETM_simple_missing/{exp}")

        self.domains = decoder.domains

        self.logging_interval = logging_interval

        LIST = np.arange(0, num_cells)
        np.random.shuffle(LIST)
        self.TRAIN_LIST = LIST[: int(0.8 * LIST.shape[0])]
        self.VAL_LIST = LIST[int(0.8 * LIST.shape[0]) :]

    def load_ckpt(self, outdir, mode="train"):
        """
        Loads a checkpoint from a file.

        Args:
            outdir (str): The directory where the checkpoint files are stored.
            mode (str, optional): The mode for loading the checkpoint. Defaults to "finetune".

        Returns:
            int: The epoch at which the checkpoint was saved.
        """
        ckpt_epoch = max(
            glob.glob(f"{outdir}/moetm_encoders_ckpt{mode}_*"),
            key=lambda x: int(x.split(f"ckpt{mode}_")[-1].split(".pth")[0]),
            default=None,
        )

        if ckpt_epoch is not None:
            ckpt_epoch = int(ckpt_epoch.split(f"ckpt{mode}_")[-1].split(".pth")[0])

            self.load_params(outdir, f"{mode}_{ckpt_epoch}")

        epoch_start = -1 if ckpt_epoch is None else ckpt_epoch

        return epoch_start

    def set_mode(self, mode):
        if mode == "train":
            self.encoders.train()
            self.decoder.train()

        elif mode == "eval":
            self.encoders.eval()
            self.decoder.eval()

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
        self.decoder.cuda()

        for module in self.decoder.mod_batch_bias[len(self.train_domains) :]:
            toogle_grad(module, requires_grad=True)
            self.optimizer.add_param_group({"params": module.parameters()})

        LIST = np.arange(0, num_ft_cells)
        np.random.shuffle(LIST)
        self.TRAIN_LIST = LIST[: int(0.8 * LIST.shape[0])]
        self.VAL_LIST = LIST[int(0.8 * LIST.shape[0]) :]

    def reconstruction(self, xs, batch_index, batch_list, domains=None, graphs=[None]):
        """Returns reconstructed features from frozen model"""
        domains = self.domains if domains is None else domains

        mu, log_sigma, gcontr_loss = self.get_embed(
            xs, batch_index, batch_list, domains=domains, graphs=graphs, eval=True
        )

        Theta = F.softmax(
            self.reparameterize(mu, log_sigma), dim=-1
        )  # log-normal distribution

        recon_log_mods = self.decoder(
            Theta, batch_list, domains=domains, cross_prediction=True, impute=False
        )
        return recon_log_mods

    def imputation(
        self,
        xs,
        batch_index,
        batch_list_impute,
        batch_list_masked,
        domains,
        graphs=[None],
    ):
        """Returns reconstructed features from frozen model"""

        # get embeddings from available features
        mu, log_sigma, gcontr_loss = self.get_embed(
            xs,
            batch_index,
            batch_list_masked,
            domains=domains,
            graphs=graphs,
            eval=True,
        )
        Theta = F.softmax(
            self.reparameterize(mu, log_sigma), dim=-1
        )  # log-normal distribution

        # get reconstructions for missing features
        recon_log_mods = self.decoder(
            Theta,
            batch_list_impute,
            domains=domains,
            cross_prediction=True,
            impute=True,
        )
        return recon_log_mods

    def train(
        self,
        xs,
        batch_index,
        batch_list,
        hyperparams,
        graphs=[None],
        mods_impute={},
        log=False,
    ):
        KL_weight = hyperparams["kl"]
        gcontr_loss = [None]

        if "impute" in self.exp:
            recon_log_mods, mu, log_sigma = self.imputation(
                mods_impute,
                xs,
                batch_index,
                batch_list,
                mode="train",
                graphs=graphs,
            )
        else:
            mu, log_sigma, _ = self.get_embed(
                xs, batch_index, batch_list, graphs=graphs, log=log
            )
            Theta = F.softmax(
                self.reparameterize(mu, log_sigma), dim=-1
            )  # log-normal distribution

            try:
                recon_log_mods = self.decoder(Theta, batch_list)
            except:
                import ipdb

                ipdb.set_trace()
        # import ipdb ; ipdb.set_trace()

        nll_mods = torch.stack(
            [(-recon_log_mods[i] * xs[i]).sum(-1).mean() for i in range(len(xs))]
        )

        KL = self.get_kl(mu, log_sigma).mean()

        if "impute" in self.exp:
            Loss = (
                nll_mods[np.setdiff1d(np.arange(mu_mods.shape[0]), mods_impute)]
                + KL_weight * KL
            )
        else:
            Loss = nll_mods.sum() + KL_weight * KL

        return (Loss, nll_mods, KL_weight * KL.item(), gcontr_loss, 0)

    def get_embeddings_shared(
        self, X_mods, batch_index_mods, batch_list_mods, graph_batch
    ):
        return self.get_embed(
            [(X_mods[i].cuda()) for i in range(len(X_mods))],
            batch_index_mods,
            batch_list_mods,
            domains=batch_list_mods[0].unique(),
            graphs=graph_batch,
            eval=True,
            embed_type="shared",
        )

    def run_epoch(
        self,
        outdir,
        hyperparams,
        epoch,
        Total_epoch,
        batch_size,
        Eval_kwargs,
        mode="train",
        inference=True,
    ):
        self.set_mode("train")
        self.mode = mode
        self.step = epoch

        train_adata = self.dataloader.adata
        train_dataset_idx = self.dataloader.dataset_idx
        batch_list_datasets = self.dataloader.batch_list_datasets
        mask_mods = self.dataloader.mask_mods
        cell_avail = self.dataloader.cell_avail

        start_time = time.time()
        # assumes gex available across all datasets
        dataset_keys = self.dataloader.adata["GEX"].obs["dataset"].unique()

        Loss_all, NLL_all, KL_all, gcontr_all, beta_loss_all = (
            0,
            0,
            0,
            [0 for i in range(len(train_adata.keys()))],
            0,
        )
        full_NLL_all_mods, full_batch_lists = None, None
        NLL_all_mods = {
            k: [0 for m in range(len(train_adata.keys()))] for k in dataset_keys
        }

        LIST = self.TRAIN_LIST
        np.random.shuffle(LIST)
        LIST = torch.tensor(LIST).cuda()

        if "cca_baseline" in self.exp:
            int_eval_dict = self.collect_results(epoch, Eval_kwargs, mode)
            return int_eval_dict

        val_loss, val_nll_mods = None, None
        epoch_start_time = time.time()

        self.optimizer.zero_grad()

        for iteration in range(max(1, LIST.shape[0] // batch_size)):
            if LIST.shape[0] // batch_size > 1:
                minibatch_idx = torch.tensor(
                    LIST[iteration * batch_size : (iteration + 1) * batch_size]
                ).cuda()
            else:
                minibatch_idx = LIST

            start = time.time()
            (
                batch_index,
                batch_index_mods_minibatch_unmasked,
                batch_index_mods_minibatch,
                x_minibatch_index_mods,
                batch_list_mods_minibatch_unmasked,
                batch_list_mods_minibatch,
                graph_batch,
            ) = self.dataloader[minibatch_idx]
            print("time to load", time.time() - start)
            """
            assert torch.equal((self.dataloader.X_mods[0].cuda())[minibatch_idx], x_minibatch_index_mods[0]), "X_mods should be the same for all modalities"
            assert torch.equal((self.dataloader.batch_list_mods[0].cuda())[minibatch_idx], batch_list_mods_minibatch_unmasked[0]), "X_mods should be the same for all modalities"
            assert torch.equal((self.dataloader.batch_index_mods[0].cuda())[minibatch_idx], batch_index_mods_minibatch_unmasked[0]), "X_mods should be the same for all modalities"
            """

            # forward/collect loss
            beta_loss = None

            loss, nll_mods, kl, gcontr, beta_loss = self.train(
                x_minibatch_index_mods,
                batch_index_mods_minibatch,
                batch_list_mods_minibatch,
                hyperparams,
                graphs=graph_batch,
                log=True,
            )

            loss_no_beta = 0
            for i in range(len(nll_mods)):
                loss_no_beta += nll_mods[i].mean()
            loss_no_beta = kl * hyperparams["kl"] + loss_no_beta
            self.writer.add_scalar("loss without beta", loss_no_beta, epoch)

            Loss_all = loss if iteration == 0 else Loss_all + loss
            KL_all += kl
            NLL_all += sum([i.mean().item() for i in nll_mods])
            
            if gcontr[0] is not None:
                gcontr_all = [
                    gcontr_all[i] + gcontr[i]
                    for i in range(len(gcontr_all))
                    if gcontr[i] is not None
                ]

            if beta_loss is not None:
                beta_loss_all += beta_loss
                
        Loss_all.backward()

        torch.nn.utils.clip_grad_norm_(self.encoders.parameters(), 50)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 50)

        self.optimizer.step()

        ##### EVAL ####
        self.set_mode("eval")
        with torch.no_grad():
            VAL_LIST = self.VAL_LIST
            np.random.shuffle(VAL_LIST)
            VAL_LIST = torch.tensor(VAL_LIST).cuda()

            (
                batch_index,
                batch_index_mods_minibatch_unmasked,
                batch_index_mods_minibatch,
                x_minibatch_index_mods,
                batch_list_mods_minibatch_unmasked,
                batch_list_mods_minibatch,
                graph_batch,
            ) = self.dataloader[VAL_LIST]

            # forward/collect loss
            beta_loss = None

            val_loss, val_nll_mods, val_kl, val_gcontr, val_beta_loss = self.train(
                x_minibatch_index_mods,
                batch_index_mods_minibatch,
                batch_list_mods_minibatch,
                hyperparams,
                graphs=graph_batch,
            )

            val_mask_mods = [
                batch_index_mods_minibatch_unmasked[i + 1][
                    torch.isin(batch_index_mods_minibatch_unmasked[i + 1], mask_mods[i])
                ]
                for i in range(len(mask_mods))
            ]

            if (
                (epoch % self.logging_interval == 0 or epoch == Total_epoch - 1)
            ) and epoch > 0:  # and (epoch > 0 or 'ft' in mode)):# or True:
                val_dataset = {
                    "adata": {
                        k: v[
                            batch_index_mods_minibatch_unmasked[i + 1]
                            .detach()
                            .cpu()
                            .numpy()  # batch_index_mods_val_minibatch[i+1]
                        ].copy()
                        for i, (k, v) in enumerate(train_adata.items())
                    },
                    "mask": val_mask_mods,
                    "missing_settings": cell_avail,
                    "batch_index": batch_index,
                }
                self.inference(
                    batch_index_mods_minibatch_unmasked[0],
                    epoch,
                    f"{mode}_val",
                    Eval_kwargs,
                    plot=False,  # (True if 'plot' in self.exp else False),
                    save=True,
                    dataset=val_dataset,
                )

            epoch_duration = time.time() - epoch_start_time

            val_loss, val_nll_mods = (
                Loss_all.item() if val_loss is None else val_loss,
                nll_mods if val_loss is None else val_nll_mods,
            )

            num_batches = max(1, ((train_adata["GEX"].shape[0] // batch_size) - 1))

            Loss_all = Loss_all / num_batches

            logging.info(
                f"{mode} Epoch {epoch} - Loss {Loss_all.item()} - Val. Loss {val_loss} - Duration: {epoch_duration:.2f} seconds"
            )

            int_eval_dict = {
                "Loss_all": Loss_all.item(),
                "NLL_all": NLL_all / num_batches,
                "NLL_mods": nll_mods,
                "Val_loss": val_loss,
                "Val_NLL_mods": val_nll_mods,
                "Val_NLL_all": sum([i.mean().item() for i in val_nll_mods]),
                "KL_all": KL_all,
                "gcontr_all": [i * hyperparams["ncl"] for i in gcontr_all],
                "beta_loss_all": beta_loss_all,
            }
            log_results(
                self.exp,
                Eval_kwargs["plot_batch_col"],
                self.writer,
                self,
                epoch,
                int_eval_dict,
                mode,
                train_adata,
                modalities=self.modalities,
                logging_mode="train",
            )

            # inference/plotting
            if (
                ((epoch % self.logging_interval == 0 or epoch == Total_epoch - 1))
                and epoch > 0
                and inference
            ):  # and (epoch > 0 or 'ft' in mode)):# or True:
                int_eval_dict.update(self.collect_results(epoch, Eval_kwargs, mode))

        return int_eval_dict

    def collect_results(self, epoch, Eval_kwargs, mode, plot=True):
        int_eval_dict = {}
        if mode == "train":
            logging.info("Inference on train set")
            batch_index_inf = torch.arange(self.dataloader.adata["GEX"].shape[0]).cuda()
            return int_eval_dict
            inf_dict = self.inference(
                batch_index_inf,
                epoch,
                mode,
                Eval_kwargs,
                plot=plot,
                save=True,
            )
            self.dataloader.adata = inf_dict[f"{mode}_save"]
            int_eval_dict.update(inf_dict)

            return int_eval_dict

        else:

            logging.info("Inference on fine-tuning train set")
            batch_index_inf = torch.tensor(
                (self.dataloader.adata["GEX"].obs["mode"] == "0").values.nonzero()[0]
            ).cuda()
            inf_dict = self.inference(
                batch_index_inf,
                epoch,
                f"train_{mode}",
                Eval_kwargs,
                plot=plot,
                save=True,
            )

            int_eval_dict.update(inf_dict)

            logging.info("Inference on fine-tuning test set")

            batch_index_inf = torch.tensor(
                (self.dataloader.adata["GEX"].obs["mode"] == "1").values.nonzero()[0]
            ).cuda()
            inf_dict_test = self.inference(
                batch_index_inf,
                epoch,
                f"test_{mode}",
                Eval_kwargs,
                plot=plot,
                save=True,
            )
            int_eval_dict.update(inf_dict_test)

            return int_eval_dict

    def impute_modalities(
        self,
        domain,
        X_mods_masked_domain,
        batch_index_mods_impute,
        batch_list_mods_domain,
        batch_list_mods_masked_domain,
        cell_avail,
        subgraphs,
        missing_mode,
    ):
        """
        Imputes the missing modalities for a given domain.

        Args:
            domain (int): The domain to impute the missing modalities for.
            X_mods_masked_domain (list): The masked data for each modality.
            batch_list_mods_domain (list): The batch indices for each modality.
            batch_list_mods_masked_domain (list): The masked batch indices for each modality.
            cell_avail (dict): The available cells for each modality.
            subgraphs (list): The subgraphs for each modality.
            missing_mode (str): The mode of missing modalities, e.g., "missing" or "missing_sim".

        Returns:
            list: The imputed modalities for the given domain.
        """
        if not any([domain in i for i in cell_avail[missing_mode]]):
            return [[] for i in range(len(X_mods_masked_domain))]

        batch_list_mods_impute = []
        for i in range(len(cell_avail[missing_mode])):
            if domain in cell_avail[missing_mode][i]:
                if "sim" in missing_mode:
                    batch_list_mods_impute.append(
                        batch_list_mods_domain[i + 1][
                            batch_list_mods_domain[i + 1] == domain
                        ].cuda()
                    )
                else:
                    batch_list_mods_impute.append(
                        batch_list_mods_domain[0][
                            batch_list_mods_domain[0] == domain
                        ].cuda()
                    )
            else:
                batch_list_mods_impute.append(torch.tensor([]).cuda())

        batch_list_mods_impute.insert(
            0, batch_list_mods_domain[0][batch_list_mods_domain[0] == domain].cuda()
        )

        # batch_list_mods_impute = [batch_list_mods_masked_domain[0] for i in range(len(batch_list_mods_masked_domain))]
        domains_impute = [
            [domain] if domain in cell_avail[missing_mode][i] else []
            for i in range(len(cell_avail[missing_mode]))
        ]
        modalities = torch.tensor([len(i) > 0 for i in domains_impute])
        logging.info(
            f"Imputing {missing_mode} modalities {self.modalities[modalities.nonzero().T[0]]} for domain {domain}"
        )

        return self.imputation(
            X_mods_masked_domain,
            batch_index_mods_impute,
            batch_list_mods_impute,
            batch_list_mods_masked_domain,
            graphs=subgraphs,
            domains=domains_impute,
        )

    def generate_feat(
        self,
        domain,
        X_mods_masked_domain,
        batch_index_mods_domain,
        batch_list_mods_domain,
        batch_list_mods_masked_domain,
        cell_avail,
        subgraphs,
    ):
        recon_mods_domain, impute_mods_domain, impute_mods_sim_domain = (
            [[] for i in range(len(X_mods_masked_domain))],
            [[] for i in range(len(X_mods_masked_domain))],
            [[] for i in range(len(X_mods_masked_domain))],
        )

        # RECONSTRUCTION/IMPUTATION results per domain

        ###### RECONSTRUCTION RESULTS ######
        if any(
            [
                domain in batch_list_mods_masked_domain[i]
                for i in range(len(batch_list_mods_masked_domain))
            ]
        ):
            domains_recons = [
                [domain] if domain in batch_list_mods_masked_domain[i + 1] else []
                for i in range(len(cell_avail["missing"]))
            ]

            # assert torch.stack([torch.tensor(len(i)) for i in batch_list_mods_masked_domain if len(i)> 0]).unique().shape[0] == 1, "batch_list_mods_masked_domain should be the same for all modalities"

            recon_mods_domain = self.reconstruction(
                X_mods_masked_domain,
                batch_index_mods_domain,
                batch_list_mods_masked_domain,
                domains=domains_recons,
                graphs=subgraphs,
            )

            modalities = torch.tensor([len(i) > 0 for i in domains_recons])
            try:
                logging.info(
                    f"Reconstructing modalities {self.modalities[modalities.nonzero().T[0].numpy()]} for domain {domain}"
                )
            except:
                import ipdb

                ipdb.set_trace()

        ###### IMPUTATION OF UNAVAILABLE MODALITY ######
        impute_mods_domain = self.impute_modalities(
            domain,
            X_mods_masked_domain,
            batch_index_mods_domain,
            batch_list_mods_domain,
            batch_list_mods_masked_domain,
            cell_avail,
            subgraphs,
            missing_mode="missing",
        )

        ###### IMPUTATION OF MASKED MODALITY #######
        impute_mods_sim_domain = self.impute_modalities(
            domain,
            X_mods_masked_domain,
            batch_index_mods_domain,
            batch_list_mods_domain,
            batch_list_mods_masked_domain,
            cell_avail,
            subgraphs,
            missing_mode="missing_sim",
        )

        ###### FULL RECONSTRUCTED/IMPUTED FEATURES #######

        # collect impute_mods_sim_domain and recon_mods_domain
        full_feat = [
            i.clone() if len(i) > 0 else torch.tensor([]) for i in recon_mods_domain
        ]
        for i in range(len(full_feat)):
            if len(full_feat[i]) == 0:
                if len(impute_mods_domain[i]) > 0:
                    full_feat[i] = impute_mods_domain[i]
                else:
                    full_feat[i] = impute_mods_sim_domain[i]
            if torch.nan in recon_mods_domain[i]:
                if len(impute_mods_domain[i]) > 0:
                    full_feat[
                        (torch.isnan(recon_mods_domain[i].max(1).values)).nonzero().T[0]
                    ] = impute_mods_domain[i]
                else:
                    full_feat[
                        (torch.isnan(recon_mods_domain[i].max(1).values)).nonzero().T[0]
                    ] = impute_mods_sim_domain[i]

        return recon_mods_domain, impute_mods_domain, impute_mods_sim_domain, full_feat

    def update_cell_avail(self, adata, cell_avail):
        return {
            k: [
                v[i][
                    torch.isin(
                        v[i],
                        torch.tensor(adata[omic].obs["batch_indices"].values).cuda(),
                    )
                ]
                for i, omic in enumerate(self.modalities)
            ]
            for k, v in cell_avail.items()
        }

    def assign_results(
        self, results, results_domain, batch_list_mods, domain, ref_used=False
    ):
        """
        Assigns the results from a specific domain to the overall results.

        Args:
            results (list): The overall results list where each element is a tensor.
            results_domain (list): The results for a specific domain where each element is a tensor.
            batch_list_mods (list): List of batch indices for each modality.
            domain (int): The domain to which the results belong.
            ref_used (bool, optional): Indicates whether a reference batch is used. Defaults to False.

        Returns:
            list: The updated overall results list.
        """
        for i in range(len(results)):
            if len(results_domain[i]) == 0:
                continue
            batch_list = batch_list_mods[i + 1] if not ref_used else batch_list_mods[0]
            try:
                results[i][
                    (batch_list == domain)
                    # .nonzero()
                    # .T[0]
                    .detach().cpu()
                ] = (results_domain[i].detach().cpu())
            except:
                import ipdb

                ipdb.set_trace()
        for i in results_domain:
            del i
        gc.collect()
        return results

    def save_and_plot_results(
        self,
        mode,
        plot,
        save,
        adata,
        embed,
        recon_mods,
        impute_mods,
        impute_mods_sim,
        batch_index_mods,
        batch_list_mods,
        cell_avail,
        Eval_kwargs,
        epoch,
        impute_embed=None,
    ):
        if plot or save:
            adata = save_int_results(
                adata,
                embed,
                recon_mods,
                impute_mods,
                impute_mods_sim,
                batch_index_mods,
                batch_list_mods,
                cell_avail,
                adata.keys(),
                Eval_kwargs,
                exp=self.exp,
                uns_tag=mode,
                impute_embed=impute_embed,
            )

            gc.collect()
            torch.cuda.empty_cache()

            if plot:
                int_eval_dict, recons_eval_dict, adata = plot_integration(
                    adata,
                    batch_list_mods,
                    cell_avail,
                    self.exp,
                    self.run_mode,
                    epoch,
                    mode,
                    mode,
                    Eval_kwargs,
                )

            else:
                int_eval_dict, recons_eval_dict = {}, {}

            return {
                f"{mode}_save": adata,
                f"{mode}_int_eval_dict": int_eval_dict,
                f"{mode}_recons_eval_dict": recons_eval_dict,
            }

    def inference(
        self, batch_index, epoch, mode, Eval_kwargs, plot=True, save=False, dataset=None
    ):
        """
        Performs inference on the given dataset and optionally plots and saves the results.

        Args:
            dataset (dict): The dataset to perform inference on.
            epoch (int): The current epoch.
            mode (str): The mode of inference, e.g., "train_ft" or "test_ft".
            eval_kwargs (dict): Additional arguments for the evaluation.
            graph (Graph, optional): The graph structure of the data. Defaults to None.
            plot (bool, optional): Whether to plot the results. Defaults to False.
            save (bool, optional): Whether to save the results. Defaults to False.

        Returns:
            dict: A dictionary containing the results of the inference.
        """

        if dataset == None:
            dataset = self.dataloader
            adata = dataset.adata
            mask_mods = dataset.mask_mods
            cell_avail = dataset.cell_avail
        else:
            adata = dataset["adata"]
            mask_mods = dataset["mask"]
            cell_avail = dataset["missing_settings"]

        if "val" not in mode and "mode" in adata["GEX"].obs.keys():
            if "train" in mode:
                adata = {k: v[v.obs["mode"] == "0"].copy() for k, v in adata.items()}
            elif "test" in mode:
                adata = {k: v[v.obs["mode"] == "1"].copy() for k, v in adata.items()}

        (
            batch_index,
            batch_index_mods_unmasked,
            batch_index_mods,
            X_mods,
            batch_list_mods_unmasked,
            batch_list_mods,
            graph,
        ) = self.dataloader[batch_index]

        if "cca_baseline" in self.exp:
            embed, recon_mods = self.cca_baseline(
                X_mods, batch_index_mods, batch_index_mods_unmasked, adata
            )

            impute_mods, impute_mods_sim = [[] for i in range(len(X_mods))], [
                [] for i in range(len(X_mods))
            ]

            X_mods_unmasked = []
            for ind, i in enumerate(self.dataloader.X_mods):
                X_mods_unmasked.append((i.cuda())[batch_index_mods_unmasked[ind + 1]])

            int_eval_dict = self.save_and_plot_results(
                mode,
                True,
                save,
                adata,
                embed,
                recon_mods,
                impute_mods,
                impute_mods_sim,
                batch_index_mods_unmasked,
                batch_list_mods_unmasked,
                cell_avail,
                Eval_kwargs,
                epoch,
            )
            log_results(
                self.exp,
                Eval_kwargs["plot_batch_col"],
                self.writer,
                self,
                0,
                int_eval_dict[f"{mode}_int_eval_dict"],
                mode,
                train_adata=adata,
                X_mods=X_mods_unmasked,
                modalities=list(adata.keys()),
                logging_mode="inference",
            )
            return int_eval_dict

        """
        try:
            assert torch.equal(batch_index.detach().cpu(),batch_index_mods_unmasked[0].detach().cpu())
            assert batch_index_mods[0].shape[0] == batch_index_mods_unmasked[0].shape[0]
            assert batch_list_mods[0].shape[0]  == batch_list_mods_unmasked[0].shape[0], "batch_list_mods should be the same for all modalities"
        except:
            import ipdb ; ipdb.set_trace()
        """
        # get reconstructions per domain (including training/testing) for efficiency
        domains = torch.unique(batch_list_mods[0])

        # initialize as nan arrays
        recon_mods = [torch.full(i.shape, torch.nan) for i in adata.values()]

        impute_mods, impute_mods_sim, full_feat_mods = (
            [
                torch.full((adata["GEX"].shape[0], i.shape[1]), torch.nan)
                for i in X_mods
            ],
            [torch.full(i.shape, torch.nan) for i in adata.values()],
            [
                torch.full((adata["GEX"].shape[0], i.shape[1]), torch.nan)
                for i in X_mods
            ],
        )
        X_mods_unmasked = []
        graph_unmasked = []
        for ind, i in enumerate(self.dataloader.X_mods):
            X_mods_unmasked.append((i.cuda())[batch_index_mods_unmasked[ind + 1]])
        if self.dataloader.graph[0] is not None:
            for ind, i in enumerate(self.dataloader.graph):
                graph_unmasked.append(
                    dgl.node_subgraph(i.to("cuda"), batch_index_mods_unmasked[ind + 1])
                )
        else:
            graph_unmasked = self.dataloader.graph

        # collect domain-specific info for reconstruction

        #  gets the MASKED and UNMASKED data for each domain
        for domain in domains:
            (
                X_mods_masked_domain,
                batch_index_mods_domain,
                batch_list_mods_domain,
                batch_list_mods_masked_domain,
                cell_avail,
                graph_domain,
            ) = index_domain(
                domain, X_mods, batch_index_mods, batch_list_mods, cell_avail, graph
            )
            (
                X_mods_unmasked_domain,
                batch_index_mods_domain_unmasked,
                batch_list_mods_domain_unmasked,
                _,
                _,
                _,
            ) = index_domain(
                domain,
                X_mods_unmasked,
                batch_index_mods_unmasked,
                batch_list_mods_unmasked,
                cell_avail,
                graph_unmasked,
            )

            print(
                "metrics for",
                adata["GEX"][
                    (batch_list_mods_unmasked[0] == domain).detach().cpu().numpy()
                ]
                .obs["Site"]
                .unique(),
            )

            assert (
                batch_list_mods_domain[0].shape[0]
                == np.array(
                    adata["GEX"][
                        (batch_list_mods_unmasked[0] == domain).detach().cpu().numpy()
                    ]
                    .obs["Site"]
                    .values
                )
                .nonzero()[0]
                .shape[0]
            ), "batch_list_mods_domain should be the same for all modalities"

            (
                recon_mods_domain,
                impute_mods_domain,
                impute_mods_sim_domain,
                full_feat_domain,
            ) = self.generate_feat(
                domain,
                X_mods_masked_domain,
                batch_index_mods_domain,
                batch_list_mods_domain_unmasked,
                batch_list_mods_masked_domain,
                cell_avail,
                graph_domain,
            )

            if domain in torch.cat(cell_avail["missing"]):
                impute_mods = self.assign_results(
                    impute_mods,
                    impute_mods_domain,
                    batch_list_mods,
                    domain,
                    ref_used=True,
                )
                assert (
                    torch.unique(
                        torch.tensor(
                            [i.shape[0] for i in impute_mods_domain if len(i) > 0]
                        )
                    ).shape[0]
                    == 1
                ), "impute_mods_domain should be the same for all modalities"
            if domain in torch.cat(cell_avail["missing_sim"]):
                impute_mods_sim = self.assign_results(
                    impute_mods_sim,
                    impute_mods_sim_domain,
                    batch_list_mods_unmasked,
                    domain,
                    ref_used=False,
                )
                assert (
                    torch.unique(
                        torch.tensor(
                            [i.shape[0] for i in impute_mods_sim_domain if len(i) > 0]
                        )
                    ).shape[0]
                    == 1
                ), "impute_mods_sim_domain should be the same for all modalities"
            full_feat_mods = self.assign_results(
                full_feat_mods,
                full_feat_domain,
                [batch_list_mods[0] for i in range(len(batch_list_mods))],
                domain,
                ref_used=True,
            )

            # recon_mods_domain = self.generate_feat(domain, X_mods_masked_domain, batch_index_mods_domain, batch_list_mods_domain, batch_list_mods_masked_domain, cell_avail, graph)

            modalities = torch.tensor([len(i) > 0 for i in recon_mods_domain])
            """
            assert torch.unique(torch.tensor([i.shape[0] for i in recon_mods_domain if len(i) > 0])).shape[0] == 1, "recon_mods_domain should be the same for all modalities"
            for i in range(len(recon_mods)):
                try:
                    assert recon_mods[i].shape[0]==batch_list_mods_unmasked[i+1].shape[0], "recon_mods should have the same batch_list_mods as batch_list_mods"
                except:
                    import ipdb ; ipdb.set_trace()
            """

            recon_mods = self.assign_results(
                recon_mods,
                recon_mods_domain,
                batch_list_mods_unmasked,
                domain,
                ref_used=False,
            )

            gc.collect()

        """
        for i in range(len(batch_list_mods)-1):
            assert torch.equal(batch_list_mods_unmasked[i+1][(~torch.isnan(recon_mods[i].max(1).values)).nonzero().T[0]].unique(return_counts=True)[-1], batch_list_mods[i+1].unique(return_counts=True)[-1]), "recon_mods should have the same batch_list_mods as batch_list_mods"
            #assert torch.equal((~torch.isnan(recon_mods[i].max(1).values)).nonzero().T[0], (batch_index_mods[i+1]-batch_index_mods_unmasked[i+1].min()).detach().cpu())
            assert torch.equal(batch_list_mods_unmasked[i+1][(~torch.isnan(impute_mods_sim[i].max(1).values)).nonzero().T[0]].unique(return_counts=True)[0], cell_avail['missing_sim'][i][torch.isin(cell_avail['missing_sim'][i],domains)]), "impute_mods_sim should have the same batch_list_mods as batch_list_mods"
            assert torch.equal(batch_list_mods_unmasked[0][(~torch.isnan(impute_mods[i].max(1).values)).nonzero().T[0]].unique(return_counts=True)[0], cell_avail['missing'][i][torch.isin(cell_avail['missing'][i],domains)]), "impute_mods should have the same batch_list_mods as batch_list_mods"

        """

        batch_index_impute = [
            batch_index_mods_unmasked[0] for i in range(len(batch_index_mods_unmasked))
        ]
        batch_list_mods_impute = [
            batch_list_mods_unmasked[0] for i in range(len(batch_list_mods_unmasked))
        ]
        graphs_impute = [graph_unmasked[0] for i in range(len(graph_unmasked))]


        full_feat_mods = [i[~torch.isnan(i).max(1).values] for i in full_feat_mods]

        embed_impute = None

        ###### COLLECT EMBEDDINGS #######
        embed = self.get_embeddings_shared(
            X_mods, batch_index_mods, batch_list_mods, graph
        )

        ###### PLOT TOPIC X FEATURE MATRICES #####

        if plot:
            if "patchwork" in self.exp:
                (mu, log_sigma, _) = self.encode(
                    X_mods, batch_index_mods, batch_list_mods, graphs=graph
                )
            else:
                (mu, log_sigma) = self.encode(
                    full_feat_mods,
                    batch_index_impute,
                    batch_list_mods_impute,
                    graphs=graphs_impute,
                )
            # top topics across domains
            Theta = (
                self.reparameterize(mu, log_sigma).detach().cpu().numpy()
            )  # log-normal distribution

            cell_types = np.array(adata["GEX"].obs["cell_type"].values)
            top_topics = np.where(Theta.sum(0) > np.quantile(Theta.sum(0), 0.8))[0]

            #  heatmap of donor x cell type x topic (group cells by donor and cell type)
            donors = np.array(adata["GEX"].obs["Site"].values)
            embed_avgs = np.zeros((len(np.unique(donors)), top_topics.shape[0]))

            for ind, donor_unique in enumerate(np.unique(donors)):
                embed_avgs[ind] = Theta[donors == donor_unique][:, top_topics].mean(0)

            plot_heatmaps_donor(donors, embed_avgs, top_topics, exp)
            plot_heatmaps_cell_types(donors, cell_types, Theta, top_topics)
            plot_heatmaps_features(adata, top_topics)
            

        ###### RESULT SAVING & PLOTTING #######
        int_eval_dict = self.save_and_plot_results(
            mode,
            plot,
            save,
            adata,
            embed,
            recon_mods,
            impute_mods,
            impute_mods_sim,
            batch_index_mods_unmasked,
            batch_list_mods_unmasked,
            cell_avail,
            Eval_kwargs,
            epoch,
            impute_embed=embed_impute,
        )

        pearson = log_results(
            self.exp,
            Eval_kwargs["plot_batch_col"],
            self.writer,
            self,
            epoch,
            int_eval_dict[f"{mode}_int_eval_dict"],
            mode,
            train_adata=adata,
            X_mods=X_mods_unmasked,
            modalities=list(adata.keys()),
            logging_mode="inference",
        )
        int_eval_dict.update({f"{mode}_pearson": pearson})
        return int_eval_dict

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_kl(self, mu, logsigma, batch_list=None):
        """Calculate KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        Args:
            mu: the mean of the q distribution.
            logsigma: the log of the standard deviation of the q distribution.
        Returns:
            KL(q||p) where q = Normal(mu, sigma and p = Normal(0, I).
        """

        logsigma = 2 * logsigma
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(-1)

    # Create a function for differentiable padding
    def differentiable_pad(self, larger, smaller):
        # Calculate the number of elements to pad
        pad_size = larger.size(0) - smaller.size(0)

        # Create a zero tensor of the same size as larger
        padding = torch.zeros(pad_size, dtype=larger.dtype, device=larger.device)

        # Create a mask to identify the non-overlapping elements
        mask = torch.zeros(larger.size(0), dtype=torch.bool, device=larger.device)
        mask[torch.where(larger == smaller.unsqueeze(1))[-1]] = True

        # Apply the mask to the padding tensor to get the padded non-overlapping elements
        result = torch.cat((smaller, padding[mask]))

        return result

    def cca_baseline(self, xs_all, batch_index, batch_index_unmasked, adata):
        # pca =PCA(n_components=400)

        # Center the data in each xs
        xs_all = [i.detach().cpu() for i in xs_all]
        xs_centered = [xs - xs.mean(0) for xs in xs_all]

        # Calculate the covariance matrices for each xs
        C_xx_list = [torch.matmul(xs.t(), xs) for xs in xs_centered]

        # Regularize the covariance matrices (optional)
        alpha = 1e-5
        C_xx_list = [
            C_xx + alpha * torch.eye(xs.shape[1])
            for C_xx, xs in zip(C_xx_list, xs_centered)
        ]

        # Compute the Cholesky decomposition for each covariance matrix
        L_xx_list = [torch.linalg.cholesky(C_xx, upper=False) for C_xx in C_xx_list]

        # Solve the generalized eigenvalue problem for each covariance matrix
        eigenpairs = [
            torch.linalg.eig(L_xx.inverse() @ C_xx @ L_xx.inverse())
            for L_xx, C_xx in zip(L_xx_list, C_xx_list)
        ]

        # Sort eigenvalues and eigenvectors for each covariance matrix
        sorted_eigenpairs = [eigenpairs[i].eigenvectors for i in range(len(eigenpairs))]

        # Canonical variables for each xs
        embed_dim = min([i.shape[1] for i in xs_centered])

        X_canonical_list = [
            xs_centered[i] @ ((sorted_eigenpairs[i].real)[:, :embed_dim])
            for i in range(len(xs_centered))
        ]

        X_recons = [torch.full(i.shape, torch.nan) for i in adata.values()]

        X_canon = torch.zeros(
            (batch_index[0].shape[0], embed_dim * 2), dtype=torch.float32
        )
        xs_pad = [X_canon for i in range(len(batch_index) - 1)]
        for i in range(1, len(batch_index)):
            mask = torch.isin(
                batch_index[0], self.dataloader.batch_index_mods[i][batch_index[i]]
            )

            if len(batch_index[i]) == 0:
                xs_pad.append(
                    torch.zeros(
                        (batch_index[0].shape[0], xs[i - 1].shape[-1]),
                        dtype=torch.float32,
                    )
                )
                continue

            mask = (
                torch.isin(
                    batch_index[0], self.dataloader.batch_index_mods[i][batch_index[i]]
                )
                .detach()
                .cpu()
            )

            if len(batch_index[i]) == 0:
                continue

            if i == 1:
                X_canon[mask, :embed_dim] = X_canonical_list[i - 1]
            else:
                X_canon[mask, embed_dim:] = X_canonical_list[i - 1]

            X_recons[i - 1][
                torch.isin(batch_index_unmasked[i], batch_index[i]).detach().cpu()
            ] = F.softmax(
                X_canonical_list[i - 1]
                @ (sorted_eigenpairs[i - 1][:, :embed_dim].real).T
            )

        return X_canon, X_recons

    def encode(
        self, xs, batch_index, batch_list, cell_avail=None, graphs=[None], log=False
    ):
        if graphs[0] is not None:
            results = zip(
                *[
                    self.encoders[i](graphs[i], xs[i]) if len(xs[i]) > 0 else ([], [])
                    for i in range(len(xs))
                ]
            )
        else:
            try:
                results = zip(
                    *[
                        self.encoders[i](xs[i]) if len(xs[i]) > 0 else ([], [])
                        for i in range(len(xs))
                    ]
                )
            except:
                import ipdb

                ipdb.set_trace()

        mu_mods, log_sigma_mods = map(list, results)

        if log == True:
            for ind, i in enumerate(mu_mods):
                if len(i) == 0:
                    continue
                self.writer.add_scalar(
                    f"norms_{self.mode}/mu_norm_{ind}", torch.norm(i, 2), self.step
                )
                self.writer.add_scalar(
                    f"norms_{self.mode}/log_sigma_norm_{ind}",
                    torch.norm(log_sigma_mods[ind], 2),
                    self.step,
                )

        # product of gaussian among available modalities for each domain
        all_mu, all_log_sigma, all_batches = [], [], []

        # get all possible batch indices (idx 0)
        full_batch_index = batch_index[0].tile(len(batch_list) - 1, 1)

        mu_mods_pad, log_sigmas_pad = [], []
        for i in range(1, len(batch_list)):
            # check if all shapes in batch_index are the same
            if not torch.equal(batch_index[0], batch_index[i]):
                mask = torch.isin(
                    batch_index[0], self.dataloader.batch_index_mods[i][batch_index[i]]
                )
            else:
                mask = torch.isin(batch_index[0], batch_index[i])

            if len(batch_index[i]) == 0:
                mu_mods_pad.append(
                    torch.zeros(
                        batch_index[0].shape[0],
                        100,
                        dtype=self.decoder.alpha_mods[0].dtype,
                        device="cuda",
                    )
                )
                log_sigmas_pad.append(
                    torch.zeros(
                        batch_index[0].shape[0],
                        100,
                        dtype=self.decoder.alpha_mods[0].dtype,
                        device="cuda",
                    )
                )
                continue

            padded_matrix = torch.zeros(
                batch_index[0].shape[0],
                100,
                dtype=self.decoder.alpha_mods[0].dtype,
                device="cuda",
            )
            try:
                padded_matrix[mask.nonzero().T[0]] = mu_mods[i - 1]
            except:
                import ipdb

                ipdb.set_trace()
            mu_mods_pad.append(padded_matrix)

            padded_matrix = torch.zeros(
                batch_index[0].shape[0],
                100,
                dtype=self.decoder.alpha_mods[0].dtype,
                device="cuda",
            )
            padded_matrix[mask.nonzero().T[0]] = log_sigma_mods[i - 1]
            log_sigmas_pad.append(padded_matrix)

        del padded_matrix

        mu_prior, logsigma_prior = self.prior_expert(
            (1, batch_index[0].shape[0], 100), use_cuda=True
        )

        Mu = torch.cat(
            (
                mu_prior,
                torch.stack(mu_mods_pad),
            ),
            dim=0,
        )

        Log_sigma = torch.cat(
            (
                logsigma_prior,
                torch.stack(log_sigmas_pad),
            ),
            dim=0,
        )

        mu, log_sigma = self.experts(Mu, Log_sigma)
        del Mu, Log_sigma
        for i in mu_mods_pad:
            del i
        for i in log_sigmas_pad:
            del i
        return mu, log_sigma

        # cell topic embeddings collected for each domain
        try:
            all_mu = torch.cat(all_mu, dim=0)
        except:
            import ipdb

            ipdb.set_trace()

        all_log_sigma = torch.cat(all_log_sigma, dim=0)
        all_batches = torch.cat(all_batches, dim=0)  # .T[0]

        assert all_batches.shape[0] == batch_list[0].shape[0]

        batches_sort = torch.argsort(all_batches)
        try:
            all_mu = all_mu[batches_sort]
        except:
            import ipdb

            ipdb.set_trace()
        all_log_sigma = all_log_sigma[batches_sort]

        del batches_sort

        return all_mu, all_log_sigma

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
        (
            mu,
            log_sigma,
        ) = self.encode(xs, batch_index, batch_list, graphs=graphs, log=log)
        if embed_type == "topic":
            return mu, log_sigma, gcontr_loss
        elif embed_type == "shared":
            Theta = F.softmax(
                self.reparameterize(mu, log_sigma), dim=-1
            )  # log-normal distribution
            domain_embed = (self.decoder.shared_embedding).detach().cpu().numpy()
            domains = batch_list[0].unique()
            topic_embed = Theta.detach().cpu().numpy()
            embed = np.zeros((topic_embed.shape[0], domain_embed.shape[-1]))
            full_batch_list = batch_list[0].detach().cpu().numpy()
            embed = topic_embed @ domain_embed
            return embed

    def get_NLL_recons(self, xs, preds, batch_list):
        nll_mods = []
        try:
            for i, pred in enumerate(preds):
                nll_mods.append(
                    (-preds[i] * xs[i]).sum(-1).mean()
                    if xs[i].shape[0] != 0
                    else torch.tensor(0.0).cuda()
                )
        except:
            import ipdb

            ipdb.set_trace()

        return nll_mods

    def prior_expert(self, size, use_cuda=False, requires_grad=False):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        @param size: integer
                    dimensionality of Gaussian
        @param use_cuda: boolean [default: False]
                        cast CUDA on variables
        """
        mu = torch.zeros(size, requires_grad=requires_grad)
        logvar = torch.zeros(size, requires_grad=requires_grad)
        if use_cuda:
            mu, logvar = mu.cuda(), logvar.cuda()
        return mu, logvar

    def experts(self, mu, logsigma, eps=1e-8):
        var = torch.exp(2 * logsigma) + eps
        # precision of i-th Gaussian expert at point x
        T = 1.0 / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1.0 / torch.sum(T, dim=0)
        pd_logsigma = 0.5 * torch.log(pd_var + eps)
        return pd_mu, pd_logsigma

    def reset_optimizer(self):
        PARA = [
            {"params": self.encoders.parameters()},
            {"params": self.decoder.parameters()},
        ]

        self.optimizer = optim.Adam(PARA, lr=self.lr)

    def load_params(self, outdir, ckpt_key):
        self.encoders.load_state_dict(
            torch.load(f"{outdir}/moetm_encoders_ckpt{ckpt_key}.pth")
        )
        self.decoder.load_state_dict(
            torch.load(f"{outdir}/moetm_decoder_ckpt{ckpt_key}.pth")
        )

        self.optimizer.load_state_dict(
            torch.load(f"{outdir}/moetm_optimizer_ckpt{ckpt_key}.pth")
        )

    def save_params(self, outdir, key, epoch, prev_ckpt=0):
        if os.path.exists(f"{outdir}/moetm_encoders_ckpt{key}_{prev_ckpt}.pth"):
            os.remove(f"{outdir}/moetm_encoders_ckpt{key}_{prev_ckpt}.pth")
            os.remove(f"{outdir}/moetm_decoder_ckpt{key}_{prev_ckpt}.pth")
            os.remove(f"{outdir}/moetm_optimizer_ckpt{key}_{prev_ckpt}.pth")
        torch.save(
            self.encoders.state_dict(), f"{outdir}/moetm_encoders_ckpt{key}_{epoch}.pth"
        )
        torch.save(
            self.decoder.state_dict(), f"{outdir}/moetm_decoder_ckpt{key}_{epoch}.pth"
        )
        torch.save(
            self.optimizer.state_dict(),
            f"{outdir}/moetm_optimizer_ckpt{key}_{epoch}.pth",
        )
