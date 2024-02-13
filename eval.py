import anndata as ad
import numpy as np
import pandas as pd
#from scglue import metrics
import matplotlib.pyplot as plt
import scipy
import os
from utils import *
from utils import _get_knn_indices
from dgl.nn import GraphConv
import torch
import torch.nn.functional as F
from eval_classes.clf_runner import *
from functools import partial
import copy
import time

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
)

from math import inf
import seaborn as sns


import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.sparse.csr import spmatrix
from scipy.stats import chi2
from typing import Mapping, Sequence, Tuple, Iterable, Union
from scipy.sparse import issparse
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_samples,
)
import random
from sklearn.neighbors import NearestNeighbors

import psutil

import matplotlib.colors as mcolors



def clustering(
    adata: ad.AnnData,
    resolutions: Sequence[float],
    clustering_method: str = "leiden",
    cell_type_col: str = "cell_types",
    batch_col: str = "batch_indices",
) -> Tuple[str, float, float]:
    """Clusters the data and calculate agreement with cell type and batch
    variable.

    This method cluster the neighborhood graph (requires having run sc.pp.
    neighbors first) with "clustering_method" algorithm multiple times with the
    given resolutions, and return the best result in terms of ARI with cell
    type.
    Other metrics such as NMI with cell type, ARi with batch are logged but not
    returned. (TODO: also return these metrics)

    Args:
        adata: the dataset to be clustered. adata.obsp shouhld contain the keys
            'connectivities' and 'distances'.
        resolutions: a list of leiden/louvain resolution parameters. Will
            cluster with each resolution in the list and return the best result
            (in terms of ARI with cell type).
        clustering_method: Either "leiden" or "louvain".
        cell_type_col: a key in adata.obs to the cell type column.
        batch_col: a key in adata.obs to the batch column.

    Returns:
        best_cluster_key: a key in adata.obs to the best (in terms of ARI with
            cell type) cluster assignment column.
        best_ari: the best ARI with cell type.
        best_nmi: the best NMI with cell type.
    """

    assert len(resolutions) > 0, f"Must specify at least one resolution."

    if clustering_method == "leiden":
        clustering_func = sc.tl.leiden
    elif clustering_method == "louvain":
        clustering_func = sc.tl.louvain
    else:
        raise ValueError(
            "Please specify louvain or leiden for the clustering method argument."
        )
    # _logger.info(f'Performing {clustering_method} clustering')
    assert cell_type_col in adata.obs, f"{cell_type_col} not in adata.obs"
    best_res, best_ari, best_nmi = None, -inf, -inf
    for res in resolutions:
        col = f"{clustering_method}_{res}"
        clustering_func(adata, resolution=res, key_added=col)
        ari = adjusted_rand_score(adata.obs[cell_type_col], adata.obs[col])
        nmi = normalized_mutual_info_score(adata.obs[cell_type_col], adata.obs[col])
        n_unique = adata.obs[col].nunique()
        if ari > best_ari:
            best_res = res
            best_ari = ari
        if nmi > best_nmi:
            best_nmi = nmi
        if batch_col in adata.obs and adata.obs[batch_col].nunique() > 1:
            ari_batch = adjusted_rand_score(adata.obs[batch_col], adata.obs[col])
            # print(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\tbARI: {ari_batch:7.4f}\t# labels: {n_unique}')
        else:
            # print(f'Resolution: {res:5.3g}\tARI: {ari:7.4f}\tNMI: {nmi:7.4f}\t# labels: {n_unique}')
            a = None

    return f"{clustering_method}_{best_res}", best_ari, best_nmi


def _get_knn_indices(
    adata: ad.AnnData,
    use_rep: str = "delta",
    n_neighbors: int = 25,
    random_state: int = 0,
    calc_knn: bool = True,
) -> np.ndarray:
    if calc_knn:
        assert (
            use_rep == "X" or use_rep in adata.obsm
        ), f'{use_rep} not in adata.obsm and is not "X"'
        neighbors = sc.Neighbors(adata)
        neighbors.compute_neighbors(
            n_neighbors=n_neighbors,
            knn=True,
            use_rep=use_rep,
            random_state=random_state,
            write_knn_indices=True,
        )
        adata.obsp["distances"] = neighbors.distances
        adata.obsp["connectivities"] = neighbors.connectivities
        adata.obsm["knn_indices"] = neighbors.knn_indices
        adata.uns["neighbors"] = {
            "connectivities_key": "connectivities",
            "distances_key": "distances",
            "knn_indices_key": "knn_indices",
            "params": {
                "n_neighbors": n_neighbors,
                "use_rep": use_rep,
                "metric": "euclidean",
                "method": "umap",
            },
        }
    else:
        assert "neighbors" in adata.uns, "No precomputed knn exists."
        assert (
            adata.uns["neighbors"]["params"]["n_neighbors"] >= n_neighbors
        ), f"pre-computed n_neighbors is {adata.uns['neighbors']['params']['n_neighbors']}, which is smaller than {n_neighbors}"

    return adata.obsm["knn_indices"]

class Evaluation:
    def __init__(self, settings):
        self.exp = settings["EXP"]
        self.dataset = settings["DATASET"]["NAME"]
        self.omic_types = settings["DATASET"]["OMICS"]
        self.seq_type = settings["DATASET"]["SEQ_TYPES"]
        self.seq_type_map = {'cite': 'GEX + ADT', 'multiome': 'GEX + ATAC'}

        self.batch_effect = settings["DATASET"]["BATCH_EFFECT"]
        self.protein_coding_idx = settings["DATASET"]["PROTEIN_CODING_ONLY"]
        self.model_name = settings["MODEL"]["NAME"]
        self.split_key = (
            settings["DATASET"]["SPLIT"] if self.model_name != "vis_only" else None
        )

        self.eval_baselines = settings["EVAL"]["BASELINES"]

        self.outdir = settings["MODEL"]["OUTDIR"]
        self.clf_key = settings["EVAL"]["CLF_KEY"]
        self.missing_omic_batch_train = settings["PATCHWORK"]["MISSING_TRAIN"]
        self.missing_omic_batch_test = settings["PATCHWORK"]["MISSING_TEST"]

        self.hid_dim = 400
        self.all_reps = ["X_int"]
        self.ref_keys = ["X_pca"]
        self.clf_runner = ClfRunner(self.hid_dim, self.clf_key)
        self.run_mode = settings["RUN_MODE"]
        self.seed = int(self.run_mode.split("seed_")[-1]) if 'seed' in self.run_mode else 1234


    def setup(self, mode, omics, adata_omics, eval_reps, outdir, rename_keys):
        """Configure evaluation suite"""
        self.clf_runner.setup(mode, adata_omics['full'])
        self.mode = mode
        self.mode_key = mode.split('_')[0]
        self.omics = omics
        self.all_reps = eval_reps
        self.rename_keys = rename_keys
        self.outdir = outdir
        
        # rename experiment names for visualization
        new_keys  = []
        for omic in ['full']:
            obsm_keys = list(adata_omics[omic].obsm.keys())
            for ind, rep in enumerate(self.all_reps):
                rep_obsm_keys =  [i for i in obsm_keys if rep in i]
                assert len(rep_obsm_keys)>0
                for k in rep_obsm_keys:

                    if rep in k:
                        new_key = k.replace(rep,self.rename_keys[ind])
                    new_keys.append(new_key)
                    adata_omics[omic].obsm[new_key] = adata_omics[omic].obsm[k]
                    if k != new_key:
                        del adata_omics[omic].obsm[k]


        for new_key in new_keys:
            if scipy.sparse.issparse(adata_omics['full'].obsm[new_key]):
                adata_omics['full'].obsm[new_key] = adata_omics['full'].obsm[new_key].todense()

        sites = adata_omics['full'].obs['Site'].unique()
        self.all_reps = self.rename_keys
        # evaluate baseline features
        if 'raw' in self.eval_baselines:
            for k in adata_omics.keys():
                adata_omics[k].obsm["X_int_raw"] = np.array(adata_omics[k].X.todense())
            self.all_reps.append('raw')

        # evaluate pca
        for omic in (self.omics + ['full']):
            sc.pp.pca(adata_omics[omic], n_comps=min(adata_omics[omic].shape[-1] - 1, self.hid_dim))
            adata_omics[omic].obsm['X_int_pca'] = adata_omics[omic].obsm['X_pca']
            
        if 'pca' in self.eval_baselines:
            self.all_reps.append('pca')
        
    
        self.results_dict = {
            'full': pd.DataFrame(index=np.hstack((self.all_reps, "Metric Type")))
        }

        self.results_dict = {'full': pd.DataFrame(index=np.hstack(([[f'{self.rename_keys[0]}_{site}' for site in sites], "Metric Type"])))}
        return adata_omics

    
    def benchmark(self, adata_omics):
        """Execute benchmarking; send output to csv file"""
        seed_everything(self.seed)

        adata_omics["full"].obs[self.batch_effect] = adata_omics[
            self.omic_types[0]
        ].obs[self.batch_effect]
    
        read_path = f"{self.outdir}/full_{self.mode}_eval.csv"
        print(f"writing eval to {self.outdir}")

        # cell type clustering
        #self.clust_eval(adata_omics['full'], emb_key='int')

        for eval_rep in self.all_reps:
            if 'raw' == eval_rep: continue
            print('clf for',eval_rep)
            #self.recons_eval(adata_omics, eval_rep)
            self.clf_eval(adata_omics['full'], eval_rep, emb_key='imp')

        # record results
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        self.record_results(self.results_dict, "integration", self.rename_keys)

        return self.results_dict


    def plot_bar_chart(self, results_key, omic_, title, ylabel, fig_tag):
        if results_key not in self.results_dict.keys():
            return
        fig, ax = plt.subplots(figsize=(10, 8))
        std_mask = self.results_dict.index.str.contains('std')
        ydata = np.array(
            self.results_dict[results_key][~std_mask].tolist()
        ).astype(np.float32)
        ystds = np.array(
            self.results_dict[results_key][std_mask].tolist()
        ).astype(np.float32)

        try:
            max_error_height = (ydata + ystds).max()
        except:
            import ipdb ; ipdb.set_trace()
        ax.set_title(title, pad=20, fontsize=20)
        width = 0.35

        # Define custom colors for bars
        colors = ['skyblue', 'lightcoral']
        num_colors = len(colors)

        # Group bars into double/triple integration
        double_ints = [ydata[i] for i in range(len(ydata)) if 'Double' in self.results_dict.index[~std_mask][i]]
        double_ints_stds = [ystds[i] for i in range(len(ydata)) if 'Double' in self.results_dict.index[std_mask][i]]
        triple_ints = [ydata[i] for i in range(len(ydata)) if 'Triple' in self.results_dict.index[~std_mask][i]]
        triple_ints_stds = [ystds[i] for i in range(len(ydata)) if 'Triple' in self.results_dict.index[std_mask][i]]
        
        if len(double_ints) > 0:
            xdata = np.arange(len(double_ints))
        else:
            xdata = np.arange(len(triple_ints))

        err_kwargs = {'linewidth': 2, 'ecolor': 'black'}
        rects_double = ax.bar(xdata, double_ints, width, yerr=double_ints_stds, label='Double', color=colors[0 % num_colors], capsize=5, error_kw=err_kwargs)
        rects_triple = ax.bar(xdata + width, triple_ints, width, yerr=triple_ints_stds, label='Triple', color=colors[1 % num_colors], capsize=5, error_kw=err_kwargs)

        # Remove 'double/triple' from index names
        index_names = [i.replace(' (Double)', '').replace(' (Triple)', '') for i in self.results_dict.index[~std_mask].tolist()]
        index_names = list(set(index_names))

        #plt.xticks(xdata, index_names)
        try:
            middle_points = xdata + width / 2
            ax.set_xticks(middle_points)
            ax.set_xticklabels(index_names)
        except:
            import ipdb ; ipdb.set_trace()

        ax.set_xlabel('Model')

        def labelvalue(rects):
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 0.02,
                        '%.2f' % height, ha='center', va='bottom')

        # Create legend using plt.Line2D
        legend_elements = [
            plt.Line2D([0], [0], color=colors[0 % num_colors], lw=10, label='Double integration'),
            plt.Line2D([0], [0], color=colors[1 % num_colors], lw=10, label='Triple integration')
        ]
        if len(rects_double) > 0:
            labelvalue(rects_double)
        if len(rects_triple) > 0:
            labelvalue(rects_triple)
            
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.1), loc='upper center', frameon=True, framealpha=0)

        ax.set_ylabel(ylabel)
        
        plt.subplots_adjust(top=1+(ystds.max()+.2))

        max_height = 0
        if len(rects_double) > 0:
            max_height = max([rect.get_height() + rect.get_y() for rect in rects_double])
        if len(rects_triple) > 0:
            max_height = max(max_height, max([rect.get_height() + rect.get_y() for rect in rects_triple]))

        ax.set_ylim([0, 1.1*max_height])
        plt.savefig(
            f"{self.outdir}/{self.mode}_{fig_tag}_{omic_}.png",
            dpi=300,
            bbox_inches="tight",
            format='png'
        )
        plt.show()

    def make_result_barplots(self):
        mode_title = self.mode.split('_ft')[0].capitalize() + ' set'
        seq_types = {'cite':  'GEX + ADT', 'multiome': 'GEX + ADT'}

        # should be done per seq type
        seq_type = seq_types[self.seq_type]
        
        for omic in self.omics:
            self.plot_bar_chart(f"Reconstruction Pearson corr. {omic}",
                                omic,
                                f"Pearson correlation of true/reconstructed {omic} features ({self.mode})",
                                "Pearson coefficient",
                                "pearson")

            self.plot_bar_chart(f"Reconstruction Spearman corr. {omic}",
                                omic,
                                f"Spearman correlation of true/reconstructed {omic} features ({self.mode})",
                                "Spearman coefficient",
                                "spearman")
            
            self.plot_bar_chart(f"Imputation Pearson corr. {omic}",
                                omic,
                                f"Pearson correlation of true/imputed {omic} features ({self.mode})",
                                "Pearson coefficient",
                                "pearson")

            self.plot_bar_chart(f"Imputation Spearman corr. {omic}",
                                omic,
                                f"Spearman correlation of true/imputed {omic} features ({self.mode})",
                                "Spearman coefficient",
                                "spearman")

        
        for clf_mode in ['Train', 'Test']:
            self.plot_bar_chart(f"{clf_mode} cross entropy",
                                    seq_type,
                                    f"{seq_type} classification cross entropy ({mode_title})",
                                    "Cross entropy",
                                    "clf_ce")
            '''
            self.plot_bar_chart(f"{clf_mode} classification ROC AUC",
                                    seq_type,
                                    f"{seq_type} classification ROC AUC ({mode_title})",
                                    "ROC AUC",
                                    "clf_acc")
            '''
            self.plot_bar_chart(f"{clf_mode} classification precision",
                                    seq_type,
                                    f"{seq_type} classification precision ({mode_title})",
                                    "Precision",
                                    "clf_prec")

            self.plot_bar_chart(f"{clf_mode} classification F1 score",
                                    seq_type,
                                    f"{seq_type} classification F1 score ({mode_title})",
                                    "F1 score",
                                    "clf_f1")
            
        self.plot_bar_chart(f"KMeans NMI",
                                seq_type,
                                f"{seq_type} k-means NMI clustering using available features ({mode_title})",
                                "NMI",
                                "nmi")


        self.plot_bar_chart(f"KMeans ARI",
                                seq_type,
                                f"{seq_type} k-means ARI clustering using available features ({mode_title})",
                                "ARI",
                                "ari")


    def get_ll(self, x_mod1, pred1):
        """Log likelihood calculation"""
        mod1_ll, mod2_ll = None, None
        if x_mod1 is not None:
            mod1_ll = (
                -(F.log_softmax(torch.tensor(pred1), dim=-1) * torch.tensor(x_mod1))
                .sum(-1)
                .mean()
                .item()
            )

        return mod1_ll

    def get_recons_metrics(self, gt, prediction, omic, site, eval_rep, key):
        if gt.shape[0] == 0:
            return

        ll = self.get_ll(gt, prediction)
        self.update_results_df(
            {f"{key} Log-likelihood {omic}": ll}, 'full', f'{eval_rep}_{self.mode_key}_{site}', f'{omic} {key}'
        )

        # correlation
        pearson = scipy.stats.pearsonr(
            np.log(1+np.squeeze(gt.reshape([1, -1]))), np.log(1+np.squeeze(prediction.reshape([1, -1])))
        )

        spearman = scipy.stats.spearmanr(
            np.log(1+np.squeeze(gt.reshape([1, -1]))), np.log(1+np.squeeze(prediction.reshape([1, -1])))
        )
        self.update_results_df({f"{key} Pearson corr. {omic}": pearson[0]}, 'full', f'{eval_rep}_{self.mode_key}_{site}', f'{omic} {key}')

        self.update_results_df({f"{key} Spearman corr. {omic}": spearman[0]}, 'full', f'{eval_rep}_{self.mode_key}_{site}', f'{omic} {key}')

        

    def recons_eval(self, adata, eval_rep):
        """Evaluate reconstruction / imputation if available"""
        # TODO: get reconstruction metrics per cell type
        for omic in self.omic_types:
            sites = adata[omic].obs['Site'].unique()
            for site in sites:
                adata_site = adata[omic][adata[omic].obs['Site']==site].copy()
                if 'train' in self.mode:
                    missing_omic_batch = self.missing_omic_batch_train[omic]
                elif 'test' in self.mode:
                    missing_omic_batch = self.missing_omic_batch_test[omic]

                # reconstruction eval
                adata_recons = adata_site[
                    ~np.isin(
                        adata_site.obs[self.batch_effect].values, missing_omic_batch
                    )
                ].copy()

                adata_recons_full = adata['full'][adata['full'].obs['Site']==site][
                        ~np.isin(
                            adata_site.obs[self.batch_effect].values, missing_omic_batch
                        )
                    ].copy()



                if f"X_recons_{omic}_{eval_rep}" in adata_recons_full.obsm.keys(): 
                    print(f'computing reconstruction metrics for omic {omic} in {site}')
                    gt = adata_site[
                        ~np.isin(
                            adata_site.obs[self.batch_effect].values, missing_omic_batch
                        )
                    ].copy()
                    prediction = adata_recons_full.obsm[f"X_recons_{omic}_{eval_rep}"]
                    
                    self.get_recons_metrics(
                        np.array(gt.X.todense()),
                        prediction,
                        omic, site, eval_rep, "Reconstruction"
                    )
                    

                # imputation eval
                if f"X_impute_sim_{omic}_{eval_rep}" in adata['full'].obsm.keys(): 
                    print(f'computing simulated imputation metrics for omic {omic} in {site}')
                    adata_recons = adata['full'][
                        np.isin(
                            adata['full'].obs[self.batch_effect].values, missing_omic_batch
                        )
                    ].copy()
                    adata_recons_full = adata['full'][adata['full'].obs['Site']==site][
                        np.isin(
                            adata_site.obs[self.batch_effect].values, missing_omic_batch
                        )
                    ].copy()

                    gt = adata_site[
                            np.isin(
                                adata_site.obs[self.batch_effect].values, missing_omic_batch
                            )
                        ].copy()
                    prediction = adata_recons_full.obsm[f"X_impute_sim_{omic}_{eval_rep}"]
                    if (~np.isnan(prediction)).max(1).nonzero()[0].shape[0] == 0: continue

                    # there may be nan values in prediction (patchwork: simulated missingness)
                    self.get_recons_metrics(
                        np.array(gt.X.todense()),
                        prediction,
                        omic, site, eval_rep, "Imputation"
                    )
                    del adata_recons, gt, prediction

    def record_results(self, dict, eval_type, rename_keys):
        """Update CSV file & update plots recording integration results"""
        # save benchmarking results per omic
        df = dict['full']
        read_path = f"{self.outdir}/{self.mode}_eval.csv"
        if os.path.exists(read_path):
            df_plot = pd.read_csv(
                f"{self.outdir}/{self.mode}_eval.csv", index_col="Unnamed: 0"
            ).dropna(axis=0)
            df_plot = df_plot.apply(pd.to_numeric, errors="ignore")
            df_plot = pd.concat([df, df_plot], ignore_index=False, axis=1)
            df_plot = df_plot[~df_plot.index.duplicated(keep="last")]

        else:
            df_plot = df
        df_plot = df_plot.dropna(how="all").dropna(how="all", axis=1)

        df_plot = df_plot.fillna(0)

        df_plot.to_csv(f"{self.outdir}/{self.mode}_eval.csv")

        prefixes =  np.unique([i.split('_')[0] for i in df_plot.index[:-1]])

        if not df_plot.empty:
            prefix_dict = {prefix: [] for prefix in prefixes}

            # Iterate through the DataFrame and group rows by prefix
            for idx, row in df_plot.iterrows():
                for prefix in prefixes:
                    if idx.startswith(prefix):
                        prefix_dict[prefix].append(row)

            # Calculate the average for each group of rows with the same prefix
            averaged_data = {}

            for prefix, rows in prefix_dict.items():
                if rows:  # Check if there are rows with the prefix
                    try:
                        all_rows = np.stack(rows).astype(np.float64)
                    except:
                        continue
                    
                    avg_row = all_rows.mean(0)
                    std_row = all_rows.std(0)
                    averaged_data[prefix] = avg_row
                    averaged_data[f'{prefix}_std'] = std_row
            
            
            # Create a new DataFrame with the averaged values
            averaged_df = pd.DataFrame.from_dict(averaged_data, orient='index', columns=df_plot.columns)

            fig, ax = plt.subplots(figsize=(25, 15),dpi=500)
            ax.axis('tight')
            ax.axis('off')
            table_data = averaged_df.reset_index().values.tolist()  # Include row names (index)
            table_data.insert(0, averaged_df.columns.tolist())      # Add column names as the first row
            table_data[0].insert(0,'Metric Type')
            ax.table(cellText=table_data, loc='center', cellLoc='center')
            plt.savefig(f'{self.outdir}/{self.mode}_results.png')

        # sort alphabetically
        self.results_dict = averaged_df.sort_index(axis=0)
        return
        self.make_result_barplots()
        #self.make_impute_matrix(f"Imputation Pearson corr. {omic}")
        

    def align_eval(self, adata_omics):
        """GLUE multi-omic alignment quality metrics"""

        for eval_rep in self.all_reps:
            if eval_rep in self.ref_keys:
                continue

            results_dict = {}
            adata = adata_omics["full"]
            adata.obsm["combined"] = adata.obsm[f"X_int_{eval_rep}"]
            multiomic_batches = (
                1
                - (adata.var.feature_types == self.omic_types[0]).astype(np.int8).values
            )

            batches = [np.zeros(i.shape[0]) for i in adata_omics.values()]

            def cell2id(cell_types):
                unique_strings, inverse_indices = np.unique(
                    cell_types, return_inverse=True
                )
                encoded_array = inverse_indices.reshape(cell_types.shape)

                return encoded_array

            cell_types = [
                cell2id(np.array(i.obs["cell_type"].values.tolist()))
                for i in adata_omics.values()
            ]
            maps = [
                metrics.mean_average_precision(
                    i.obsm[f"X_int_{eval_rep}"], cell_types[ind]
                )
                for ind, i in enumerate(adata_omics.values())
            ]

            sas = metrics.seurat_alignment_score(
                adata.obsm["combined"], multiomic_batches
            )

            neighbor_cosistencies = [
                metrics.neighbor_conservation(
                    i.obsm[f"X_int_{eval_rep}"], i.obsm[self.ref_keys[0]], batches[ind]
                )
                for ind, i in enumerate(adata_omics.values())
            ]

            for ind, omic in adata_omics.keys():
                results_dict.update(
                    {
                        f"{omic} neighbor consistency": neighbor_cosistencies[ind],
                        f"{omic} MAP": maps[ind],
                        "Seurat alignment": sas,
                    }
                )
            self.update_results_df(results_dict, "full", eval_rep, "Bio conservation")
            results_dict = {
                "Seurat alignment": sas,
            }
            self.update_results_df(results_dict, "full", eval_rep, "Aggregate scores")


    def clust_eval(self, adata_og, eval_reps=[], emb_key='int'):
        #import faiss
        """Assess clustering of integration"""
        if len(eval_reps) == 0: eval_reps = self.all_reps
        sites = adata_og.obs['Site'].unique()

        for eval_rep in eval_reps:
            for site in sites:
                print('clust eval for site',site)
                adata = adata_og[adata_og.obs['Site']==site].copy()

                if isinstance(adata.obsm[f"X_{emb_key}_{eval_rep}"], scipy.sparse.csr_matrix):
                    adata.obsm[f"X_{emb_key}_{eval_rep}"] = np.asarray(
                        adata.obsm[f"X_{emb_key}_{eval_rep}"].todense()
                    )

                adata_copy = adata[~np.isnan(adata.obsm[f"X_{emb_key}_{eval_rep}"]).max(1)].copy()

                _get_knn_indices(
                    adata_copy,
                    use_rep=f'X_{emb_key}_{eval_rep}',
                    n_neighbors=15,
                    random_state=0,
                    calc_knn=True,
                )
                resolutions = np.arange(0.75, 2, 0.1)
                # calculate clustering metrics
                cluster_key, best_ari, best_nmi = clustering(
                    adata_copy,
                    resolutions=resolutions,
                    cell_type_col='cell_type',
                    batch_col=self.batch_effect,
                    clustering_method="louvain",
                )

                if 'imp' in emb_key:
                    clust_dict = {
                        'KMeans ARI (+ imputed)': best_ari,
                        'KMeans NMI (+ imputed)': best_nmi,
                    }
                    print(f'for site:{site}, best ari is {best_ari}, best nmi is {best_nmi}')
                    self.update_results_df(
                        clust_dict, 'full', f'{eval_rep}_{self.mode_key}_{site}', "Cell type clustering"
                    )
                else:
                    clust_dict = {
                        'KMeans ARI': best_ari,
                        'KMeans NMI': best_nmi,
                    }
                    print(f'for site:{site}, best ari is {best_ari}, best nmi is {best_nmi}')
                    self.update_results_df(
                        clust_dict, 'full', f'{eval_rep}_{self.mode_key}_{site}', "Cell type clustering"
                    )
                    

    def update_results_df(self, dict, dict_key, rep, group):
        #rep = f'{rep}_{self.seq_type}'
        try:
            if rep not in self.results_dict[dict_key].index:
                self.results_dict[dict_key][rep] = 0.
        except:
            import ipdb ; ipdb.set_trace()

        for col_name, values in dict.items():
            self.results_dict[dict_key].at[rep, col_name] = values
            self.results_dict[dict_key].at["Metric Type", col_name] = group

    def clf_eval(self, adata, eval_reps=[], emb_key='int'):
        """Given cell type annotations, perform cell type classification and report loss"""
        if len(eval_reps) == 0: eval_reps = self.all_reps
        str_labels = np.unique(np.array(list(adata.obs[self.clf_key])))
        
        sites = adata.obs['Site'].unique()
        for eval_key in np.unique(np.array(eval_reps)):
            for site in sites:
                print('site',site)
                adata_copy = adata[adata.obs['Site']==site].copy()
                self.clf_runner.setup(self.mode, adata_copy)

                confusion_mats = self.clf_runner.run_clf(adata_copy, eval_key, emb_key='int')
                self.update_results_df(
                    self.clf_runner.results_dict, 'full', f'{eval_key}_{self.mode_key}_{site}', "Cell type classification"
                )
            