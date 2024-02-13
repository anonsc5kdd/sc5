import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
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
import copy
import random
import re
from scipy import sparse
import csv
import seaborn as sns
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad

def plot_heatmaps_features(adata, top_topics):
    top_topics = np.array([7, 18, 21, 46, 61])
    # heatmap of feature x topic
    for i in range(len(self.decoder.rhos)):
        mod_embed = (
            (self.decoder.alpha @ self.decoder.rhos[i].t())
            .T.detach()
            .cpu()
            .numpy()
        )

        top_topic_feats = []
        for topic in top_topics:
            top_topic_feats.append(np.argsort(-mod_embed[:, topic])[:5])
        top_feat_idx = np.concatenate(top_topic_feats)

        feat_names = adata[list(adata.keys())[i]].var_names[top_feat_idx]

        # top 5 features for each topic: feat by topic
        plt.figure(figsize=(12, 12))
        sns.heatmap(
            mod_embed[top_feat_idx][:, top_topics]
            / np.abs(mod_embed[top_feat_idx][:, top_topics].max(0)),
            cmap="coolwarm",
            yticklabels=feat_names,
            cbar_kws={"shrink": 0.5},
            xticklabels=top_topics  # ,
            # vmin=-80,
            # vmax=80
        )
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"topic_x_feature_{i}.png")

def plot_heatmaps_cell_types(donors, cell_types, Theta, top_topics):
    full_legend = {}
    for batch in np.unique(donors):
        np.random.seed(3)
        random.seed(4)
        donor_idx = np.where(donors == batch)[0]
        cell_types_sorted_idx = np.argsort(cell_types[donor_idx])

        rand_idx = np.random.choice(
            np.arange(Theta[donor_idx].shape[0]), 250, replace=False
        )
        rand_idx.sort()
        row_labels = cell_types[donor_idx][cell_types_sorted_idx][rand_idx]

        palette = cc.glasbey_light
        df = pd.DataFrame(row_labels)
        df.index = row_labels

        label_to_color = {
            label: color for label, color in zip(set(row_labels), palette)
        }

        # map each row to its corresponding color
        row_colors = df.index.map(label_to_color)

        fig, ax = plt.subplots(figsize=(25, 20), dpi=310)

        cg = sns.clustermap(
            Theta[donor_idx][cell_types_sorted_idx][rand_idx][:, top_topics],
            cmap="coolwarm",
            row_colors=row_colors,
            xticklabels=top_topics,
            yticklabels=False,
            row_cluster=False,
            col_cluster=False,
            vmax=2,
            vmin=-0.8,
            # figsize=(10,19),
            tree_kws={"linewidths": 0.0},
        )

        ax.set_xlabel("Topic")
        legend_patches = [
            Patch(color=color, label=label)
            for label, color in label_to_color.items()
        ]
        # Plot legend
        plt.legend(
            handles=legend_patches,
            frameon=False,
            fontsize="x-small",
            loc="upper right",
            bbox_to_anchor=(0.2, 0.7),
            title="Cell type",
            bbox_transform=plt.gcf().transFigure,
        )

        plt.savefig(f"embed_heatmap_cell_types_{batch}.png", dpi=310)

def plot_heatmaps_donor(donors, embed_avgs, top_topics, exp):
    renamed_batches = []
    for i in np.unique(donors):
        # split then join
        new_str = i.split("test_")[-1]
        new_str = new_str.replace("_", " ")
        new_str = " ".join([word.capitalize() for word in new_str.split()])

        if "Site" in new_str:
            new_str = new_str.replace("Site", "Site ")
        if "Cite" in new_str:
            new_str = new_str.replace("Cite", "CITE-seq")
        renamed_batches.append(new_str)
    renamed_batches = np.array(renamed_batches)

    # sort renamed batches by the last string
    renamed_sort_idx = np.argsort([s.split()[-1] for s in renamed_batches])

    # fig, ax = plt.subplots(figsize=(12,6))

    sns.heatmap(
        embed_avgs[renamed_sort_idx],
        cmap="coolwarm",
        xticklabels=top_topics,
        yticklabels=np.unique(renamed_batches)[renamed_sort_idx],
    )
    plt.xlabel("Topic")
    plt.ylabel("Domain")
    plt.yticks(rotation=0)
    # ax.figure.tight_layout()
    plt.savefig(f"embed_heatmap_donors_{self.exp}.png", dpi=310)


def sample_fraction_of_subranges(arr, num_points):
    nonzero_indices = np.nonzero(arr)[0]
    subranges = np.array_split(nonzero_indices, 5)

    sampled_indices = []
    for i in subranges:
        sampled_indices.append(np.random.choice(i, num_points))

    return np.concatenate(sampled_indices)


def split_data(train_adata, X_mods, ft_mode):
    ft_mode = "0" if ft_mode == "train" else "1"
    train_idx = {
        k: (v.obs["mode"] == ft_mode).values.nonzero()[0]
        for k, v in train_adata.items()
    }
    train_adata = {k: train_adata[k][train_idx[k]] for k, v in train_adata.items()}
    X_mods = [X_mods[i][train_idx[k]] for i, (k, v) in enumerate(train_adata.items())]
    return train_adata, X_mods


def index_domain(domain, X_mods, batch_index_mods, batch_list_mods, cell_avail, graph):
    try:
        X_mods_masked_domain = [
            (X_mods[i].cuda())[batch_list_mods[i + 1] == domain]
            if domain in batch_list_mods[i + 1]
            else []
            for i in range(len(X_mods))
        ]
    except:
        import ipdb

        ipdb.set_trace()

    if graph[0] is not None:
        graph_domain = [
            dgl.node_subgraph(
                graph[i], (batch_list_mods[i + 1] == domain).nonzero().T[0]
            )
            for i in range(len(graph))
        ]
    else:
        graph_domain = graph

    batch_list_mods_masked_domain = [
        batch_list_mods[i + 1][batch_list_mods[i + 1] == domain]
        for i in range(len(X_mods))
    ]
    batch_list_mods_masked_domain.insert(
        0, batch_list_mods[0][batch_list_mods[0] == domain]
    )
    batch_index_mods_domain = [
        batch_index_mods[i][batch_list_mods[i] == domain]
        for i in range(len(batch_index_mods))
    ]
    batch_list_mods_domain = [
        batch_list_mods[i][batch_list_mods[i] == domain]
        for i in range(len(batch_list_mods))
    ]
    if graph[0] is not None:
        subgraphs = [
            dgl.node_subgraph(
                i.to("cuda"), (batch_list_mods[ind + 1] == domain).nonzero().T[0].cuda()
            )
            for ind, i in enumerate(graph)
        ]
    else:
        subgraphs = [None]

    for i in range(len(X_mods)):
        if i not in batch_list_mods_masked_domain[i]:
            continue
        try:
            assert (
                batch_list_mods_masked_domain[i].unique()[0] == domain
                and batch_list_mods_masked_domain[i].unique().shape[0] == 1
            )
            assert (
                batch_list_mods_domain[i].unique()[0] == domain
                and batch_list_mods_domain[i].unique().shape[0] == 1
            )
            assert (
                torch.tensor([i.shape[0] for i in batch_list_mods_domain])
                .unique()[
                    torch.tensor([i.shape[0] for i in batch_list_mods_domain]).unique()
                    != 0
                ]
                .shape[0]
                == 1
            )
        except:
            import ipdb

            ipdb.set_trace()

    return (
        X_mods_masked_domain,
        batch_index_mods_domain,
        batch_list_mods_domain,
        batch_list_mods_masked_domain,
        cell_avail,
        graph_domain,
    )


def seed_everything(seed=1234):
    """Set random seeds for run"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_int_results(
    adata,
    embed,
    recon_mods,
    impute_mods,
    impute_mods_sim,
    batch_index,
    batch_list,
    cell_avail,
    adata_keys,
    Eval_kwargs,
    split_idx=None,
    exp="",
    uns_tag="train",
    impute_embed=None,
):
    # save integration results to GEX
    all_missing = torch.cat(cell_avail["missing_sim"])
    all_missing = all_missing[
        torch.unique(all_missing, return_counts=True)[-1] == len(recon_mods)
    ]

    if impute_embed is not None:
        adata["GEX"].obsm[f"X_int_imp_{exp}"] = np.full(
            [adata["GEX"].shape[0], impute_embed.shape[-1]], np.nan
        )
        try:
            adata["GEX"].obsm[f"X_int_imp_{exp}"][
                ~torch.isin(
                    torch.tensor(adata["GEX"].obs["batch_indices"].values),
                    all_missing.detach().cpu(),
                )
            ] = impute_embed
        except:
            import ipdb

            ipdb.set_trace()
        adata["GEX"].obsm[f"X_int_imp_{exp}"] = sparse.csr_matrix(
            adata["GEX"].obsm[f"X_int_imp_{exp}"]
        )

    adata["GEX"].obsm[f"X_int_{exp}"] = np.full(
        [adata["GEX"].shape[0], embed.shape[-1]], np.nan
    )
    try:
        adata["GEX"].obsm[f"X_int_{exp}"][
            ~torch.isin(
                torch.tensor(adata["GEX"].obs["batch_indices"].values),
                all_missing.detach().cpu(),
            )
        ] = embed
    except:
        import ipdb

        ipdb.set_trace()
    adata["GEX"].obsm[f"X_int_{exp}"] = sparse.csr_matrix(
        adata["GEX"].obsm[f"X_int_{exp}"]
    )

    # adata["GEX"].obsm[f"X_int_{exp}"] = sparse.csr_matrix(embed)

    # save reconstruction results to obsm
    for ind, k in enumerate(adata_keys):
        adata[k].obsm[f"X_recons_{k}_{exp}"] = (
            sparse.csr_matrix(recon_mods[ind].numpy())
            if len(recon_mods[ind]) > 0
            else sparse.csr_matrix(np.full(adata[k].shape, np.nan))
        )
        if len(recon_mods[ind]) > 0:
            print(
                uns_tag,
                k,
                "reconstruction logged for batches",
                adata[k]
                .obs[Eval_kwargs["batch_col"]][
                    ~np.isnan(recon_mods[ind].numpy()).max(1)
                ]
                .unique(),
            )

        # save simulated imputation results to obsm
        adata[k].obsm[f"X_impute_sim_{k}_{exp}"] = (
            sparse.csr_matrix(impute_mods_sim[ind].numpy())
            if len(impute_mods_sim[ind]) > 0
            else sparse.csr_matrix(np.full(adata[k].shape, np.nan))
        )

        if len(impute_mods[ind]) == 0:
            continue

        # imputation results: not simulated (save to uns)
        impute_idx = (
            (1 - np.isnan(impute_mods[ind]))
            .max(1)
            .values.to(bool)
            .nonzero()
            .T[0]
            .numpy()
        )
        if len(impute_mods[ind]) > 0:
            print(
                uns_tag,
                k,
                "sim. imputation logged for batches",
                adata[k]
                .obs[Eval_kwargs["batch_col"]][
                    ~np.isnan(impute_mods_sim[ind].numpy()).max(1)
                ]
                .unique(),
            )

        if impute_idx.shape[0] > 0:
            print(
                uns_tag,
                k,
                "imputation logged for batches",
                adata["GEX"].obs[Eval_kwargs["batch_col"]][impute_idx].unique(),
            )

            adata[k].uns[f"X_impute_{uns_tag}_{k}_{exp}"] = sparse.csr_matrix(
                impute_mods[ind][impute_idx].numpy()
            )

            adata[k].uns[f"X_impute_obs_{uns_tag}_{k}_{exp}"] = adata["GEX"].obs_names[
                impute_idx
            ]

            adata[k].uns[f"X_impute_batch_{uns_tag}_{k}_{exp}"] = (
                batch_list[0][impute_idx].detach().cpu().numpy()
            )

            adata[k].uns[f"X_impute_cell_type_{uns_tag}_{k}_{exp}"] = (
                adata["GEX"].obs["cell_type"][impute_idx].values
            )
            batch_effect = Eval_kwargs["batch_col"]
            adata[k].uns[f"X_impute_batchval_{uns_tag}_{k}_{exp}"] = (
                adata["GEX"].obs[batch_effect][impute_idx].values
            )

    for i in recon_mods:
        del i
    for i in impute_mods_sim:
        del i
    for i in impute_mods:
        del i

    gc.collect()
    torch.cuda.empty_cache()

    return adata


def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def plot_integration(
    adata_omics,
    batch_list,
    cell_avail,
    exp,
    run_mode,
    epoch,
    mode,
    uns_key,
    Eval_kwargs,
):
    seed = int(run_mode.split("seed_")[-1])
    seed_everything(seed)
    int_eval_dict, int_recons_dict = {}, {}
    print("plotting..")

    # domains = batch_list[0].unique().detach().cpu().numpy()

    batch_effect = Eval_kwargs["batch_col"]
    plot_domains = np.array(
        adata_omics["GEX"].obs[Eval_kwargs["plot_batch_col"]].unique()
    )
    domains = np.array(adata_omics["GEX"].obs[Eval_kwargs["batch_col"]].unique())

    # import ipdb ; ipdb.set_trace()
    if (
        "impute" not in run_mode
        and "recons" not in run_mode
        and "raw" not in run_mode
        and "noplot" not in run_mode
    ):
        ###### PLOT EMBEDDINGS ######
        adata_omics["GEX"].obsm[f"X_int_{exp}"] = np.array(
            adata_omics["GEX"].obsm[f"X_int_{exp}"].todense()
        )
        valid_int_idx = ~np.isnan(adata_omics["GEX"].obsm[f"X_int_{exp}"]).max(1)
        adata_copy = adata_omics["GEX"][valid_int_idx].copy()

        all_nmi, all_ari = [], []
        all_nmi_imp, all_ari_imp = [], []
        """
        int_eval_dict_ = evaluate(
                adata=adata_copy,
                n_epoch=epoch,
                return_fig=True,
                plot_fname=f"{mode}_",
                embedding_key=f"X_int_{exp}",
                **Eval_kwargs,
            )
        """

        for batch in adata_omics["GEX"].obs[Eval_kwargs["batch_col"]].unique():
            adata_batch = adata_copy[
                (adata_copy.obs[Eval_kwargs["batch_col"]] == batch)
            ].copy()
            int_eval_dict_ = evaluate(
                adata=adata_batch,
                n_epoch=epoch,
                return_fig=True,
                plot_fname=f"{mode}_{batch}_",
                embedding_key=f"X_int_{exp}",
                **Eval_kwargs,
            )
            all_nmi.append(int_eval_dict_[f"{mode}_{batch}_nmi"])
            all_ari.append(int_eval_dict_[f"{mode}_{batch}_ari"])

            int_eval_dict.update(int_eval_dict_)
            print(int_eval_dict_)
            del adata_batch
            gc.collect()

        adata_omics["GEX"].obsm[f"X_int_{exp}"] = scipy.sparse.csr_matrix(
            adata_omics["GEX"].obsm[f"X_int_{exp}"]
        )
        print(int_eval_dict)
        all_nmi = np.array(all_nmi)
        all_ari = np.array(all_ari)

        all_nmi = np.array(all_nmi_imp)
        all_ari = np.array(all_ari_imp)
        del adata_copy
        gc.collect()

    ###### PLOT RECONSTRUCTED FEATURES ######
    for modality, adata in adata_omics.items():
        if "plotrecons" in run_mode:
            print("plotting reconstruction for modality", modality)
            adata.obsm[f"X_recons_{modality}_{exp}"] = np.array(
                adata.obsm[f"X_recons_{modality}_{exp}"].todense()
            )
            valid_recons_idx = ~np.isnan(adata.obsm[f"X_recons_{modality}_{exp}"]).max(
                1
            )

            adata_copy = adata.copy()[valid_recons_idx]

            assert (
                np.nan not in adata_copy.obsm[f"X_recons_{modality}_{exp}"]
            ), "Nan found in plotted reconstruction"

            for batch in adata_copy.obs[batch_effect].unique():
                adata_batch = adata_copy[
                    (adata_copy.obs[Eval_kwargs["batch_col"]] == batch)
                ].copy()
                int_recons_dict_ = evaluate(
                    adata=adata_batch,
                    n_epoch=epoch,
                    return_fig=True,
                    plot_fname=f"{mode}_recons_{batch}_{modality}_epoch",
                    embedding_key=f"X_recons_{modality}_{exp}",
                    **Eval_kwargs,
                )
                int_recons_dict.update(int_recons_dict_)
                del adata_batch
                adata.obsm[f"X_recons_{modality}_{exp}"] = sparse.csr_matrix(
                    adata.obsm[f"X_recons_{modality}_{exp}"]
                )

            if "plotraw" in exp and (epoch == 0 or epoch == 1):
                adata_copy = adata[:, adata.var.feature_types == modality]
                evaluate(
                    adata=adata_copy,
                    n_epoch=epoch,
                    return_fig=True,
                    plot_fname=f"{mode}_reconsRAW_{modality}_epoch",
                    embedding_key=f"raw_features",
                    **Eval_kwargs,
                )

            print(int_recons_dict)

        if "plotimpute" in run_mode:
            # plotting imputed features for domain

            if f"X_impute_{uns_key}_{modality}_{exp}" in adata.uns.keys() and False:
                impute_adata = ad.AnnData(
                    np.array(
                        adata.uns[f"X_impute_{uns_key}_{modality}_{exp}"].todense()
                    )
                )
                if impute_adata.shape[0] != 0:
                    impute_adata.obs_names = adata.uns[
                        f"X_impute_obs_{uns_key}_{modality}_{exp}"
                    ]
                    impute_adata.obs["batch_indices"] = adata.uns[
                        f"X_impute_batch_{uns_key}_{modality}_{exp}"
                    ]
                    impute_adata.obs["cell_type"] = adata.uns[
                        f"X_impute_cell_type_{uns_key}_{modality}_{exp}"
                    ]
                    impute_adata.obsm["impute"] = np.log(1 + impute_adata.X)

                    batch_effect = Eval_kwargs["batch_col"]
                    impute_adata.obs[batch_effect] = adata.uns[
                        f"X_impute_batchval_{uns_key}_{modality}_{exp}"
                    ]
                    for batch in impute_adata.obs[batch_effect].unique():
                        print("plotting imputed", modality, "for batch", batch)

                        impute_adata_batch = impute_adata[
                            impute_adata.obs[batch_effect] == batch
                        ].copy()
                        int_eval_dict_imp = evaluate(
                            adata=impute_adata_batch,
                            n_epoch=epoch,
                            return_fig=True,
                            plot_fname=f"{mode}_{uns_key}_impute_{batch}_{modality}_epoch",
                            embedding_key="impute",
                            **Eval_kwargs,
                        )
                        del impute_adata_batch
                        gc.collect()
                del impute_adata
                gc.collect()

            # plot simulated imputed features
            if f"X_impute_sim_{modality}_{exp}" in adata.obsm.keys():
                # plot per dataset

                datasets = adata.obs["dataset"].unique()

                for dataset in datasets:
                    adata_dataset = adata[adata.obs["dataset"] == dataset].copy()
                    print("plotting imputed", modality, "features for dataset", dataset)
                    adata_dataset.obsm[f"X_impute_sim_{modality}_{exp}"] = np.array(
                        adata_dataset.obsm[f"X_impute_sim_{modality}_{exp}"].todense()
                    )
                    valid_impute_idx = ~np.isnan(
                        adata_dataset.obsm[f"X_impute_sim_{modality}_{exp}"]
                    ).max(1)
                    if valid_impute_idx.max():
                        adata_copy = adata_dataset.copy()[valid_impute_idx]
                        adata_copy.obsm[f"X_impute_sim_{modality}_{exp}"] = np.log(
                            1 + adata_copy.obsm[f"X_impute_sim_{modality}_{exp}"]
                        )
                        int_imputesim_dict_ = evaluate(
                            adata=adata_copy,
                            n_epoch=epoch,
                            return_fig=False,
                            plot_fname=f"{mode}_impute_sim_{dataset}_{modality}_epoch",
                            embedding_key=f"X_impute_sim_{modality}_{exp}",
                            **Eval_kwargs,
                        )
                        print(int_imputesim_dict_)

                        if "plotraw" in exp and (epoch == 0 or epoch == 1):
                            adata_copy = adata_copy[
                                :, adata_copy.var.feature_types == modality
                            ]
                            evaluate(
                                adata=adata_copy,
                                n_epoch=epoch,
                                return_fig=True,
                                plot_fname=f"{mode}_imputeRAW_sim_{modality}_epoch",
                                embedding_key=f"raw_features",
                                **Eval_kwargs,
                            )
                    del adata_dataset
                    gc.collect()

    # plot original features for each modality
    if "plotraw" in run_mode:
        for ind, (modality, adata) in enumerate(adata_omics.items()):
            for batch in adata_omics["GEX"].obs[Eval_kwargs["batch_col"]].unique():
                print("plotting raw", modality, "features for batch", batch)

                adata_batch = adata[
                    (adata.obs[Eval_kwargs["batch_col"]] == batch)
                ].copy()

                adata_batch.obsm["raw_features"] = np.array(adata_batch.X.todense())

                seq_types = (
                    adata_batch.obs[Eval_kwargs["batch_col"]].str.split("_").str[-1]
                )
                for seq_type in seq_types.unique():
                    adata_copy_seq = adata_batch[
                        np.isin(
                            adata_batch.obs[Eval_kwargs["batch_col"]]
                            .str.split("_")
                            .str[-1],
                            seq_type,
                        )
                    ].copy()

                    int_eval_dict_raw = evaluate(
                        adata=adata_copy_seq,
                        n_epoch=epoch,
                        # color_dict=color_dict,
                        return_fig=True,
                        plot_fname=f"{mode}_raw_{seq_type}_{batch}_{modality}",
                        embedding_key="raw_features",
                        **Eval_kwargs,
                    )
                    del adata_copy_seq
                    print("raw", int_eval_dict_raw)

                del adata_batch

    gc.collect()
    return int_eval_dict, int_recons_dict, adata_omics


def log_results(
    exp,
    batch_effect,
    writer,
    trainer,
    epoch,
    int_eval_dict,
    mode,
    train_adata=None,
    X_mods=None,
    modalities=None,
    logging_mode="train",
):
    if logging_mode == "train":
        (
            Loss_all,
            NLL_all,
            nll_mods,
            Val_loss,
            val_nll_mods,
            Val_NLL_all,
            KL_all,
            gcontr_all,
            beta_loss_all,
        ) = int_eval_dict.values()
        # record training
        for ind, encoder in enumerate(trainer.encoders):
            for name, param in encoder.named_parameters():
                writer.add_histogram(
                    f"Encoder{modalities[ind]}_params_{mode}/{name}", param, epoch
                )
        if "patchwork" in exp:
            for key, mu_prior in zip(trainer.domains, trainer.mu_priors):
                writer.add_histogram(f"Priors_{mode}/{key}_mu", mu_prior, epoch)
            for key, logsigma_prior in zip(trainer.domains, trainer.logsigma_priors):
                writer.add_histogram(
                    f"Priors_{mode}/{key}_logsigma", logsigma_prior, epoch
                )
        for name, param in trainer.decoder.named_parameters():
            writer.add_histogram(f"Decoder_params_{mode}/{name}", param, epoch)

        writer.add_scalar(f"Loss_{mode}/total_train_loss", Loss_all, epoch)
        writer.add_scalar(f"Loss_{mode}/total_val_loss", Val_loss, epoch)
        writer.add_scalar(
            f"Loss_{mode}/train_vae_loss",
            NLL_all,
            epoch,
        )
        writer.add_scalar(
            f"Loss_{mode}/val_vae_loss",
            Val_NLL_all,
            epoch,
        )

        for ind, i in enumerate(nll_mods):
            writer.add_scalar(
                f"Loss_{mode}/NLL_mod{modalities[ind]}",
                i.mean(),
                epoch,
            )
            
        for ind, i in enumerate(nll_mods):
            writer.add_scalar(
                f"Loss_{mode}/NLL2_mod{modalities[ind]}",
                i,
                epoch,
            )

        writer.add_scalar(f"Loss_{mode}/KL_loss", KL_all, epoch)
        if gcontr_all[0] > 0:
            writer.add_scalar(f"Loss_{mode}/GCL", torch.stack(gcontr_all).sum(), epoch)
            for ind, gcontr_loss in enumerate(gcontr_all):
                writer.add_scalar(
                    f"Loss_{mode}/GCL_{modalities[ind]}", gcontr_loss, epoch
                )

        if beta_loss_all is not None:
            writer.add_scalar(f"Loss_{mode}/beta_loss", beta_loss_all, epoch)
    else:
        if "mode" in train_adata["GEX"].obs.keys() and "val" not in mode:
            ft_mode = mode.split("_ft")[0]
            train_adata, X_mods = split_data(train_adata, X_mods, ft_mode)

        domains = np.array(train_adata["GEX"].obs[batch_effect].unique())

        if len(int_eval_dict) > 0:
            for batch in train_adata["GEX"].obs[batch_effect].unique():
                writer.add_scalar(
                    f"Clustering_{mode}/ARI_{batch}",
                    int_eval_dict[f"{mode}_{batch}_ari"],
                    epoch,
                )
                writer.add_scalar(
                    f"Clustering_{mode}/NMI_{batch}",
                    int_eval_dict[f"{mode}_{batch}_nmi"],
                    epoch,
                )
                if f"{mode}_{batch}_imp_ari" in int_eval_dict.keys():
                    writer.add_scalar(
                        f"Clustering_{mode}_imp/ARI_{batch}",
                        int_eval_dict[f"{mode}_{batch}_imp_ari"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"Clustering_{mode}_imp/NMI_{batch}",
                        int_eval_dict[f"{mode}_{batch}_imp_nmi"],
                        epoch,
                    )

        # pearson/spearman/nmse (RECONSTRUCTION) per domain
        pearson_recons_all = []
        for i in range(len(X_mods)):
            omic = list(train_adata.keys())[i]
            for batch in train_adata[omic].obs[batch_effect].unique():
                all_pearsons = []
                all_spearmans = []
                recons_flatten, impute_flatten = None, None

                if f"X_recons_{omic}_{exp}" in train_adata[omic].obsm.keys():
                    feats = np.array(
                        train_adata[omic][train_adata[omic].obs[batch_effect] == batch]
                        .obsm[f"X_recons_{omic}_{exp}"]
                        .todense()
                    )

                    print("metrics for modality", modalities[i])
                    valid_recons_idx = ~np.isnan(feats).max(1)
                    feats_recons = X_mods[i][
                        train_adata[omic].obs[batch_effect] == batch
                    ][valid_recons_idx]

                    predict_recons = feats[valid_recons_idx]

                    feats_flatten = (
                        torch.squeeze(feats_recons.reshape([1, -1]))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    recons_flatten = np.squeeze(predict_recons.reshape([1, -1]))
                    del feats_recons, predict_recons, feats
                    gc.collect()

                    if recons_flatten is not None and recons_flatten.shape[0] > 0:
                        pearson_recons = scipy.stats.pearsonr(
                            np.log(1 + feats_flatten),
                            np.log(1 + recons_flatten)
                        )

                        spearman_recons = scipy.stats.spearmanr(
                            np.log(1 + feats_flatten),
                            np.log(1 + recons_flatten)
                        )
                        pearson_recons_all.append(pearson_recons.statistic)


                        writer.add_scalar(
                            f"Pearson_recons_{mode}/pearson_recons_{modalities[i]}_{batch}",
                            pearson_recons.statistic,
                            epoch,
                        )
                        writer.add_scalar(
                            f"Spearman_recons_{mode}/spearman_recons_{modalities[i]}_{batch}",
                            spearman_recons.statistic,
                            epoch,
                        )
                        all_pearsons.append(pearson_recons.statistic)
                        all_spearmans.append(spearman_recons.statistic)

                        del feats_flatten, recons_flatten
                        gc.collect()

            all_pearsons = np.array(all_pearsons)
            all_spearmans = np.array(all_spearmans)
            print(
                "OMIC:",
                omic,
                "PEARSON:",
                all_pearsons.mean(),
                "+/-",
                all_pearsons.std(),
            )
            print(
                "OMIC:",
                omic,
                "SPEARMAN:",
                all_spearmans.mean(),
                "+/-",
                all_spearmans.std(),
            )

            if f"X_impute_sim_{omic}_{exp}" in train_adata[omic].obsm.keys():
                feats = np.array(
                    train_adata[omic].obsm[f"X_impute_sim_{omic}_{exp}"].todense()
                )
                scale_factor = train_adata[omic].obs["feat_sum"].values

                valid_recons_idx = ~np.isnan(feats).max(1)
                batches_missing = (
                    train_adata[omic].obs[batch_effect][valid_recons_idx].unique()
                )
                for batch_missing in batches_missing:
                    adata_batch = train_adata[omic][valid_recons_idx].copy()
                    # missing_idx = adata_batch.obs[batch_effect] == batch_missing
                    if X_mods[i].shape[0] == feats.shape[0]:
                        feats_impute = (
                            X_mods[i].detach().cpu().numpy()
                        )[valid_recons_idx][
                            adata_batch.obs[batch_effect] == batch_missing
                        ]
                    else:
                        feats_impute = np.log(
                            1
                            + X_mods[i][adata_batch.obs[batch_effect] == batch_missing]
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    predict_impute = feats[
                        valid_recons_idx
                    ][adata_batch.obs[batch_effect] == batch_missing]

                    ###### save per-cell correlation ######

                    samp_idx = np.random.choice(np.arange(feats_impute.shape[0]), 1000)

                    # Calculate the correlation matrix
                    feats_impute_ = feats_impute
                    adata_batch_missing = adata_batch[
                        adata_batch.obs[batch_effect] == batch_missing
                    ]
                    batch_cell_types = adata_batch_missing.obs["cell_type"].unique()


                    if 1 in valid_recons_idx:
                        feats_flatten_impute = np.squeeze(feats_impute.reshape([1, -1]))
                        impute_flatten = np.squeeze(predict_impute.reshape([1, -1]))
                    del feats_impute, predict_impute
                    gc.collect()

                    if impute_flatten is not None and impute_flatten.shape[0] > 0:
                        pearson_impute = scipy.stats.pearsonr(
                            feats_flatten_impute,
                            impute_flatten,
                        )

                        spearman_impute = scipy.stats.spearmanr(
                            feats_flatten_impute,
                            impute_flatten,
                        )

                        writer.add_scalar(
                            f"Pearson_impute_{mode}/pearson_impute_{modalities[i]}_{batch}",
                            pearson_impute.statistic,
                            epoch,
                        )
                        writer.add_scalar(
                            f"Spearman_impute_{mode}/spearman_impute_{modalities[i]}_{batch}",
                            spearman_impute.statistic,
                            epoch,
                        )
                        all_pearsons.append(pearson_impute.statistic)
                        all_spearmans.append(spearman_impute.statistic)

                        del feats_flatten_impute, impute_flatten
                        gc.collect()

                        print(
                            "OMIC:",
                            omic,
                            "BATCH:",
                            batch_missing,
                            "PEARSON IMPUTE:",
                            pearson_impute.statistic,
                            "+/-",
                            pearson_impute.pvalue,
                        )
                        print(
                            "OMIC:",
                            omic,
                            "BATCH:",
                            batch_missing,
                            "SPEARMAN IMPUTE:",
                            spearman_impute.statistic,
                            "+/-",
                            spearman_impute.pvalue,
                        )
