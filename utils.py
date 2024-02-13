import anndata as ad
import networkx as nx
import numpy as np
import scanpy as sc
import matplotlib
from matplotlib import rcParams
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import os
import sys
import sklearn
import scipy
import numpy.linalg as npla
import torch
import glob
#import scglue
import psutil
from matplotlib import colors
import matplotlib.colors as mcolors
import random
import dgl
import gc
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import random
import scipy.sparse as sparse


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


def get_logger(name, *, level="INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s")
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


def create_graph(omic_types, adata_omics, mode, adata_omics_val, neighbor_path):
    adatas_train = list(adata_omics.values())

    if os.path.exists(neighbor_path):
        for i, omic in enumerate(omic_types):
            with open(f"{neighbor_path}/{omic}_train.pkl", "rb") as fin:
                adatas_train[i].obsp["connectivities"] = pkl.load(fin)

        if adata_omics_val is not None:
            adatas_val = list(adata_omics_val.values())
            for i, omic in enumerate(omic_types):
                with open(f"{neighbor_path}/{omic}_val.pkl", "rb") as fin:
                    adatas_val[i].obsp["connectivities"] = pkl.load(fin)
    else:
        os.mkdir(neighbor_path)

        for adata in adatas_train:
            sc.pp.pca(adata, n_comps=30)
            sc.pp.neighbors(adata, n_neighbors=15)

        for i, omic in enumerate(omic_types):
            with open(f"{neighbor_path}/{omic}_train.pkl", "wb") as fout:
                pkl.dump(adatas_train[i].obsp["connectivities"], fout)

        if adata_omics_val is not None:
            adatas_val = list(adata_omics_val.values())
            for adata in adatas_val:
                sc.pp.pca(adata, n_comps=30)
                sc.pp.neighbors(adata, n_neighbors=15)

            for i, omic in enumerate(omic_types):
                with open(f"{neighbor_path}/{omic}_val.pkl", "wb") as fout:
                    pkl.dump(adatas_val[i].obsp["connectivities"], fout)

    adata_omics = {omic: adatas_train[i] for i, omic in enumerate(omic_types)}
    if adata_omics_val is not None:
        adata_omics_val = {omic: adatas_val[i] for i, omic in enumerate(omic_types)}
    return adata_omics, adata_omics_val


def _find_dominate_set(W, K=20):
    """
    Retains `K` strongest edges for each sample in `W`

    Parameters
    ----------
    W : (N, N) array_like
        Input data
    K : (0, N) int, optional
        Number of neighbors to retain. Default: 20

    Returns
    -------
    Wk : (N, N) np.ndarray
        Thresholded version of `W`
    """
    # let's not modify W in place
    Wk = W.copy().todense()

    # determine percentile cutoff that will keep only `K` edges for each sample
    # remove everything below this cutoff
    cutoff = 100 - (100 * (K / W.shape[0]))

    topk_indices = np.argpartition(Wk, -K, axis=0)[-K:]
    mask = np.zeros_like(Wk, dtype=bool)
    Wk = np.where(mask, Wk, 0)

    # Wk[Wk < np.percentile(Wk, np.tile(cutoff,Wk.shape[0]), axis=1, keepdims=True)] = 0

    # normalize by strength of remaining edges
    Wk = Wk / Wk.sum(1)
    return sparse.coo_matrix(Wk)


def _B0_normalized(W, alpha=1.0):
    """
    Adds `alpha` to the diagonal of `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF
    alpha : (0, 1) float, optional
        Factor to add to diagonal of `W` to increase subject self-affinity.
        Default: 1.0

    Returns
    -------
    W : (N, N) np.ndarray
        Normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    W = W + (alpha * scipy.sparse.eye(W.shape[0]))
    # W = check_symmetric(W, raise_warning=False)
    assert (W != W.T).nnz == 0
    return W


def compute_label_shift(adata, group_key, label_key, site_key, save_path="./"):
    agg_key = site_key
    agg_labels = {
        group: list(adata[adata.obs[agg_key] == group].obs[group_key].cat.categories)
        for group in adata.obs[agg_key].cat.categories
    }

    agg_groups = np.array(
        list([val for sublist in agg_labels.values() for val in sublist])
    )
    groups = agg_groups[np.sort(np.unique(agg_groups, return_index=True)[-1])]

    group_labels = {
        group: list(adata[adata.obs[group_key] == group].obs[label_key].cat.categories)
        for group in groups
    }

    group_labels = {
        group: list(adata[adata.obs[group_key] == group].var.feature_types)
        for group in groups
    }

    all_labels = list(
        set([val for sublist in group_labels.values() for val in sublist])
    )
    all_labels = [i for i in all_labels if i != ""]
    binary_matrix = np.zeros((len(group_labels), len(all_labels)), dtype=int)

    for i, row_key in enumerate(group_labels.keys()):
        for j, col_value in enumerate(all_labels):
            if col_value in group_labels[row_key] and col_value != "":
                binary_matrix[i, j] = 1

    for agg_ind, (agg_group_key, items) in enumerate(agg_labels.items()):
        rows = np.intersect1d(np.array(groups), np.array(items), return_indices=True)[1]
        # Create a heatmap
        # fig, ax = plt.subplots(1,1, figsize=(12,5))
        fig, ax = plt.subplots(1, 1, figsize=(20, 15))

        plt.title(f"Cell types across donors in {agg_group_key} for COVID dataset")
        plt.xticks(np.arange(len(all_labels)))

        plt.yticks(np.arange(len(rows)), items)
        # ax.yaxis.set_major_locator(ticker.FixedLocator([tick + 0.5 for tick in np.arange(len(rows))]))
        ax.set_yticklabels(items)
        # import ipdb ; ipdb.set_trace()
        ax.set_xticks(np.arange(0, len(all_labels), 1), minor=True)
        ax.set_yticks(np.arange(0, len(rows), 1), minor=True)

        maj_locator = ticker.IndexLocator(2, 0)
        min_locator = ticker.IndexLocator(1, 0)
        # ax.xaxis.set_major_locator(maj_locator)
        ax.xaxis.set_minor_locator(min_locator)

        # maj_locator = ticker.IndexLocator(2, 0)
        min_locator = ticker.IndexLocator(1, 0)
        # ax.yaxis.set_major_locator(maj_locator)
        ax.yaxis.set_minor_locator(min_locator)

        # Gridlines based on minor ticks
        ax.grid(which="minor", color="w", linestyle="-", linewidth=2)

        # Remove minor ticks
        ax.tick_params(which="minor", bottom=False, left=False)

        plt.xlabel(label_key)
        plt.ylabel(group_key)

        if 0 in binary_matrix[rows]:
            plt.imshow(
                binary_matrix[rows], cmap="binary", aspect="auto", origin="lower"
            )
        else:
            cmap = plt.cm.colors.ListedColormap(["black"])
            plt.imshow(binary_matrix[rows], cmap=cmap, aspect="auto", origin="lower")

        legend_labels = all_labels  # These are your true x-labels
        legend_values = np.arange(1, len(all_labels) + 1)  # Create numbers from 1 to n

        labels = [f"{i}: {label}" for i, label in zip(legend_values, legend_labels)]
        ax.text(
            1.1,
            0.98,
            "\n".join(labels),
            fontsize=10,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"facecolor": "white", "alpha": 0.7},
        )
        plt.tight_layout()
        legend_labels = ["Absent", "Present"]
        legend_patches = [
            mpatches.Patch(color="white", label=legend_labels[0]),
            mpatches.Patch(color="black", label=legend_labels[1]),
        ]
        plt.legend(
            handles=legend_patches,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=True,
            ncols=2,
            mode="expand",
        )

        plt.savefig(f"{save_path}_{agg_group_key}.png", bbox_inches="tight")


# https://github.com/yoseflab/scib-metrics/blob/0.4.1/src/scib_metrics/benchmark/_core.py#L276-L364
def plot_results_table(
    omic, save_dir, results_df, compare_reps, mode, seq_type, eval_type, rename_keys
):
    EVAL_EXP = "FULL"
    SCORE_COL = "Total"

    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.plots import bar
    from plottable.formatters import decimal_to_percent

    """Plot results dictionary from eval.py"""
    num_embeds = len(compare_reps)

    import ipdb ; ipdb.set_trace()

    groups = results_df.loc["Metric Type"]

    numeric_df = results_df.drop('Metric Type', axis=0)
    per_class_score = numeric_df.transpose().groupby("Metric Type").mean()
    per_class_score = per_class_score.rename(index={})
    
    #per_class_score["Total"] = per_class_score #results_df.transpose().mean()
    

    # Calculate the mean score for each row
    results_df[SCORE_COL] = numeric_df.mean(axis=1)
    results_df.loc['Metric Type', SCORE_COL] = SCORE_COL

    df = results_df.drop("Metric Type")

    df = df.apply(pd.to_numeric)

    # Filter df to keep only indices that contain at least one element from combined_elements
    cmap_fn = lambda col, col_data: normed_cmap(col_data, cmap=plt.cm.Greens, num_stds=0.5) if 'loss' not in col and 'entropy' not in col else normed_cmap(col_data, cmap=plt.cm.Greens_r, num_stds=0.5)

    import ipdb ; ipdb.set_trace()

    # Split columns by Metric Type, using df as it doesn't have the new method col
    score_cols = results_df.columns[results_df.loc["Metric Type"] == SCORE_COL]
    other_cols = results_df.columns[results_df.loc["Metric Type"] != SCORE_COL]
    column_definitions = [
        ColumnDefinition(
            "Method", width=1.5, textprops={"ha": "left", "weight": "bold"}
        ),
    ]
    norms = [
        mcolors.Normalize(
            vmin=float(df[col].min()), vmax=float(df[col].max())
        )
        for col in other_cols
    ]

    # Circles for the metric values
    for i, col in enumerate(other_cols):
        if df.shape[0] > 1:
            # norm = plt.Normalize(vmin=df[col].min(), vmax=df[col].max())
            column_definitions += [
                ColumnDefinition(
                    col,
                    title=col.replace(" ", "\n", 1),
                    width=5,
                    textprops={
                        "ha": "center",
                        "bbox": {"boxstyle": "circle", "pad": 0.5},
                    },
                    cmap=cmap_fn(col, df[col]),
                    group=groups.iloc[i],
                    # formatter=decimal_to_percent
                    formatter="{:.2f}",
                )
            ]
        else:
            column_definitions += [
                ColumnDefinition(
                    col,
                    title=col.replace(" ", "\n", 1),
                    width=5,
                    textprops={
                        "ha": "center",
                        "bbox": {"boxstyle": "circle", "pad": 0.5},
                    },
                    # cmap=plt.cm.Greens(total),
                    group=groups.iloc[i],
                    # formatter=decimal_to_percent
                    formatter="{:.2f}",
                )
            ]

    # Bars for the aggregate scores
    column_definitions += [
        ColumnDefinition(
            col,
            width=3,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": matplotlib.cm.YlGnBu,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=results_df.loc["Metric Type", col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols)
    ]

    # Allow to manipulate text post-hoc (in illustrator)
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.8, 3 + 3.0 * num_embeds))
        tab = Table(
            df,
            column_definitions=column_definitions,
            row_dividers=True,
            footer_divider=True,
            ax=ax,
            textprops={"fontsize": 8},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
        ).autoset_fontcolors(colnames=df.columns)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if save_dir is not None:
        fig.savefig(
            os.path.join(save_dir, f"results_{mode}_{seq_type}_{omic}_{EVAL_EXP}.png"),
            facecolor=ax.get_facecolor(),
            dpi=300,
        )

def save_int(adata_omic, set_key, save_path, settings, partial=False):
    """Save integrated features into AnnData object"""
    # TODO: validate saving procedure
    omic_types = settings["DATASET"]["OMICS"]
    model_keys = np.array(adata_omic[omic_types[0]].obsm_keys())
    try:
        model_key = model_keys[np.char.startswith(model_keys, "X_")][0]
    except:
        import ipdb ; ipdb.set_trace()

    if os.path.exists(save_path):
        adata = ad.read_h5ad(save_path)

        if partial == True:
            split_key = settings["DATASET"]["SPLIT_KEY"]
            test_idx = np.intersect1d(
                adata_omic['full'].obs[split_key][adata_omic['full'].obs['dataset']==set_key],
                np.array(adata.obs[split_key][adata_omic['full'].obs['dataset']==set_key].cat.categories),
                return_indices=True,
            )[1]
            adata.obsm[model_key] = adata_omic['full'].obsm[model_key][adata_omic['full'].obs['dataset']==set_key][test_idx]
        else:
            for i in adata_omic['full'][adata_omic['full'].obs['dataset']==set_key].obs.keys():
                try:
                    adata.obs[i] = adata_omic['full'][adata_omic['full'].obs['dataset']==set_key].obs[i]
                except Exception as e:
                    print(e)
                    pass
            for i in adata_omic['full'].obsm.keys():
                try:
                    adata.obsm[i] = adata_omic['full'][adata_omic['full'].obs['dataset']==set_key].obsm[i]
                except Exception as e:
                    print(e)
                    import ipdb ; ipdb.set_trace()
                    pass
            for i in adata_omic['full'][adata_omic['full'].obs['dataset']==set_key].uns.keys():
                try:
                    adata.uns[i] = adata_omic['full'][adata_omic['full'].obs['dataset']==set_key].uns[i]
                except Exception as e:
                    print(e)
                    pass

    else:
        adata = adata_omic["GEX"]
    if len(adata_omic["GEX"].uns.keys()) > 0:
        adata.uns[list(adata_omic["GEX"].uns.keys())[0]] = list(
            adata_omic["GEX"].uns.values()
        )[0]
    if not os.path.exists(save_path[: save_path.rfind("/")]):
        os.makedirs(save_path[: save_path.rfind("/")])
    print(
        "writing", model_key, "to anndata object of size", adata.shape, "at", save_path
    )

    try:
        adata.write_h5ad(save_path)
    except Exception as e:
        print(e)
        import ipdb

        ipdb.set_trace()
        print("rewrite")


def seed_everything(seed=1234):
    """Set random seeds for run"""
    random.seed(seed)
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_gpu_usage(tag):
    """Logging GPU usage for debugging"""
    allocated_bytes = torch.cuda.memory_allocated(torch.device("cuda"))
    cached_bytes = torch.cuda.memory_cached(torch.device("cuda"))

    allocated_gb = allocated_bytes / 1e9
    cached_gb = cached_bytes / 1e9
    print(
        f"{tag} -> GPU Memory - Allocated: {allocated_gb:.2f} GB, Cached: {cached_gb:.2f} GB"
    )


def check_cpu_usage(tag):
    return
    # Get CPU times
    cpu_usage = psutil.cpu_percent(interval=1)
    print(f"{tag} -> CPU usage: {cpu_usage}")


# TODO: how to select best distance metric? chebyshev/correlation also good options
def get_knn_graph(adata, rep=None, n_neighbors=15, n_comps=50):
    """Compute KNN graph describing omic-specific interactions and corresponding statistics"""
    check_cpu_usage(
        f"computing knn graph for rep {rep} for {n_neighbors} neighbors and {n_comps} comps"
    )
    if n_comps is None and rep is not None:
        n_comps = min(n_comps, adata.obsm[rep].shape[1])
    elif n_comps is not None:
        n_comps = min(n_comps, adata.shape[1])

    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors, n_pcs=n_comps)
    adj = adata.obsp["connectivities"]
    return adj


# TODO: there is a numpy dependency conflict with faiss nearest neighbors with scanpy (numba),
# this implementation might be useful in the future for computing nn on large graphs
'''
import faiss
from scib_metrics.nearest_neighbors import NeighborsOutput
def faiss_hnsw_nn(X: np.ndarray, k: int):
    """Gpu HNSW nearest neighbor search using faiss.

    See https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md
    for index param details.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    res = faiss.StandardGpuResources()
    M = 32
    index = faiss.IndexHNSWFlat(X.shape[1], M, faiss.METRIC_L2)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(X)
    distances, indices = gpu_index.search(X, k)
    del index
    del gpu_index
    # distances are squared
    return NeighborsOutput(indices=indices, distances=np.sqrt(distances))


def faiss_brute_force_nn(X: np.ndarray, k: int):
    """Gpu brute force nearest neighbor search using faiss."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(X.shape[1])
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(X)
    distances, indices = gpu_index.search(X, k)
    del index
    del gpu_index
    # distances are squared
    return NeighborsOutput(indices=indices, distances=np.sqrt(distances))
'''


def get_geneAct(adata):
    """..."""
    return adata.copy().transpose()
    adata_geneAct = ad.AnnData(X=adata.obsm["ATAC_gene_activity"].T)
    adata_geneAct.obs_names = adata.uns["ATAC_gene_activity_var_names"]
    # obs_keys = ['nCount_peaks','atac_fragments','reads_in_peaks_frac','blacklist_fraction','nucleosome_signal','Site','D']
    obsm_keys = ["umap"]
    for key in adata.obs.keys():
        adata_geneAct.obs[key] = adata.obs[key].copy()
    return adata_geneAct


def configure_visualizer(out_dir):
    # scglue.plot.set_publication_params()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sc.settings.figdir = out_dir
    sc.settings.verbosity = 0
    # rcParams["figure.figsize"] = (4, 4)

    colors1 = plt.cm.Blues_r(np.linspace(0, 0.1, 1))
    colors2 = plt.cm.Reds(np.linspace(0, 1, 98))
    colors3 = plt.cm.Purples_r(np.linspace(0, 0.1, 1))
    colorsComb = np.vstack([colors1, colors2, colors3])
    mymap = colors.LinearSegmentedColormap.from_list("my_colormap", colorsComb)
    return mymap


def compare_shared_topology(rna, atac, nx_rna, nx_atac, rna_batches):
    """For *paired* KNN graphs describing omic-specific interactions, compare topologies"""
    # ged = nx.graph_edit_distance(nx_rna,nx_atac,node_match=(lambda x, y : True))
    # NOTE: expensive
    # _,cost = nx.optimal_edit_paths(nx_rna,nx_atac,node_match =(lambda x, y: True))
    raw_neighbor_consistency = metrics.neighbor_conservation(rna.X, atac.X, rna_batches)
    pca_neighbor_consistency = metrics.neighbor_conservation(
        rna.obsm["X_pca"], atac.obsm["X_pca"], rna_batches
    )
    glue_neighbor_consistency = metrics.neighbor_conservation(
        rna.obsm["X_glue"], atac.obsm["X_glue"], rna_batches
    )
    return (
        None,
        raw_neighbor_consistency,
        pca_neighbor_consistency,
        glue_neighbor_consistency,
    )
