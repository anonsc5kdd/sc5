import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import torch
import scipy
import gc
import glob
import os
from utils import *
import warnings

warnings.filterwarnings("ignore")


class DataLoader:
    """Prepare/normalize counts from selected omics data"""

    def __init__(self, settings):
        self.exp = settings["EXP"]
        self.dataset = settings["DATASET"]["DIR"]
        self.omic_types = settings["DATASET"]["OMICS"]
        self.seq_types = settings["DATASET"]["SEQ_TYPES"]
        self.denoise = settings["DENOISE"]["PRE"]
        self.batch_effect = settings["DATASET"]["BATCH_EFFECT"]
        self.plot_batch_effect = settings["DATASET"]["PLOT_BATCH_EFFECT"]
        self.site = settings["DATASET"]["SITE"]
        self.status = settings["DATASET"]["STATUS"]
        self.donor = settings["DATASET"]["DONOR"]
        self.model_name = settings["MODEL"]["NAME"]
        self.split_key = (
            settings["DATASET"]["SPLIT"] if self.model_name != "vis_only" else None
        )
        self.valid_split_frac = settings["DATASET"]["VAL_SPLIT_FRAC"]
        self.test_split_frac = settings["DATASET"]["TEST_SPLIT_FRAC"]
        self.counts_key = settings["DATASET"]["RAW_COUNTS_KEY"]

        self.donor_key = settings["DATASET"]["DONOR_KEY"]
        self.status_key = settings["DATASET"]["STATUS_KEY"]
        self.site_key = settings["DATASET"]["SITE_KEY"]
        self.validation = settings["MODEL"]["VALIDATION"]
        self.outdir = settings["MODEL"]["OUTDIR"]

        self.outdir = self.outdir[: self.outdir.rfind("/")]
        self.ppdir = settings["DATASET"]["PPDIR"]
        self.split_idx = None
        self.train_keys, self.test_keys = (
            settings["DATASET"]["TRAIN_KEYS"],
            settings["DATASET"]["TEST_KEYS"],
        )
        self.run_mode = settings["RUN_MODE"]
        self.seed = int(self.run_mode.split("seed_")[-1]) if "seed_" in self.run_mode else 1234

        seed_everything(self.seed)

    def slice_adata(self, adata, omic, aux_adata=None):
        """Slices AnnData object by omic"""
        if omic == "full":
            feat_idx = np.arange(adata.shape[-1])
        elif aux_adata is not None:
            feat_idx, aux_feat_idx = np.intersect1d(
                adata.var_names, aux_adata.var_names, return_indices=True
            )[1:]
            feat_idx = feat_idx[np.isin(feat_idx,np.array(adata.var.feature_types.values == omic).nonzero())]
            aux_feat_idx = aux_feat_idx[np.isin(aux_feat_idx,np.array(aux_adata.var.feature_types.values == omic).nonzero())]

            aux_adata = aux_adata[:, aux_feat_idx]
        else:
            feat_idx = np.array(adata.var.feature_types == omic)
        adata = adata[:, feat_idx]
        check_cpu_usage(f"indexing: {adata.shape[-1]} -> {feat_idx.shape[-1]}")

        if np.unique(adata.var.feature_types.values).shape[
            0
        ] != 1 and omic not in np.unique(adata.var.feature_types.values):
            import ipdb ; ipdb.set_trace()
            raise ValueError("AnnData object not correctly indexed!")
        check_cpu_usage("indexed")
        if aux_adata is None:
            return adata
        else:
            return adata, aux_adata
        

    def preprocess_counts(
        self, adata_og, omic, aux_adata=None
    ):
        """Log-normalize counts and index by highly variable genes"""
        if aux_adata:
            adata, aux_adata = self.slice_adata(adata_og, omic, aux_adata=aux_adata)
            adata = ad.concat([adata, aux_adata], merge='same', axis=0, label='merge_keys').copy()
            del aux_adata
        else:
            adata = self.slice_adata(adata_og, omic).copy()

        if self.counts_key != "None":
            adata.X = adata.layers[self.counts_key]#.copy()

        # selects highly-variable genes within each batch, and merge -> lightweight batch correction
        batch = self.batch_effect if self.denoise == True else None

        if "neurips" in self.dataset:
            adata.X = scipy.sparse.csr_matrix(adata.X)

        if omic != "ADT":

            adata_og = ad.AnnData.copy(adata)

            sc.pp.normalize_total(adata, target_sum=1e4)
            
            sc.pp.log1p(adata)
            if omic == 'ATAC':
                sc.pp.highly_variable_genes(adata,n_top_genes=4000)
            else:
                sc.pp.highly_variable_genes(adata)
            index = adata.var["highly_variable"].values
            adata = adata_og[:, index].copy()

        adata.X = adata.X.todense()

        adata.obs['feat_sum'] = adata.X.sum(1)
        adata.X = (adata.X / adata.X.sum(1))

        adata.X = scipy.sparse.csr_matrix(adata.X)

        # save to backed mode
        adata.write_h5ad(f"{self.outdir}/tmp_{self.exp}_{omic}.h5ad")
        del adata
        gc.collect()

    def prep_batches(self, adata):  # , mode_keys):
        """Inject batch indices, and index subjects for integration by desired batch effect"""
        batch_effect_idx = None
        # train/test across a specific group
        if self.site != "None":
            batch_effect_idx = (adata.obs[self.site_key] == self.site).values
            # adata = adata[site_idx]
        elif self.donor != "None":
            batch_effect_idx = (adata.obs[self.donor_key] == self.donor).values
            # adata = adata[subject_idx]
        elif self.status != "None":
            batch_effect_idx = (adata.obs[self.status_key] == self.status).values
            # adata = adata[status_idx]
        batch_effect_idx = (
            batch_effect_idx.nonzero()[0]
            if batch_effect_idx is not None
            else np.arange(adata.shape[0])
        )
        return batch_effect_idx

    def train_test_split(self, adata, mode, batch_effect_idx):
        """Split AnnData into train and test splits, according to configuration"""
        if (
            self.split_key is None or self.test_split_frac == 1
        ) and self.validation == False:
            return {
                "main": adata[batch_effect_idx].obs[self.split_key].cat.categories,
                "validation": None,
            }

        adata_split_keys = adata[batch_effect_idx].obs[self.split_key].cat.categories

        mode_keys = {"main": None, "validation": None}

        if mode == "train":
            if 'None' in self.train_keys:
                adata_split_keys = np.random.permutation(adata_split_keys)
                split_idx = (
                    int(len(adata_split_keys) * self.test_split_frac)
                    if len(adata_split_keys) > 2
                    else 1
                )

                self.train_keys = adata_split_keys[:split_idx]
                self.test_keys = (
                    np.setdiff1d(adata_split_keys, adata_split_keys[:split_idx])
                )

            else:
                split_idx = len(self.train_keys)

            train_split = int(np.ceil(split_idx * self.valid_split_frac))
            mode_keys["main"] = (
                self.train_keys[:split_idx]
                if split_idx > 1
                else self.train_keys[: (split_idx + 1)]
            )
            if self.validation == True:
                mode_keys["main"], mode_keys["validation"] = (
                    adata_split_keys[:train_split],
                    adata_split_keys[train_split:split_idx],
                )

        elif mode == "test":
            mode_keys["main"] = self.test_keys
        print('mode keys',mode_keys)

        return mode_keys

    def split_adata_omics(
        self, adata, omic_types, mode_keys, batch_effect_idx, aux_adata=None, aux_batch_effect_idx=None
    ):
        """Train/test split of AnnData object into omic-specific objects. Combines datasets if requested"""
        adata_omics = {}

        adata = adata[batch_effect_idx].to_memory().copy()
        if aux_adata is not None:
            aux_adata = aux_adata[aux_batch_effect_idx].to_memory().copy()

        for omic in omic_types:
            # whether to combine anndata objects or not
            if aux_adata and omic in aux_adata.var.feature_types.values and omic in adata.var.feature_types.values:
                self.preprocess_counts(
                    adata,
                    omic,
                    aux_adata=aux_adata,
                )
            elif aux_adata and omic in aux_adata.var.feature_types.values:
                self.preprocess_counts(
                    aux_adata,
                    omic
                )
            else:
                self.preprocess_counts(adata,omic)
            adata_load = ad.read_h5ad(
                f"{self.outdir}/tmp_{self.exp}_{omic}.h5ad"
            )
            adata_omics[omic] = adata_load#[batch_effect_idx]

        for i in glob.glob(f"{self.outdir}/tmp_{self.exp}*"):
            os.remove(i)

        return adata_omics

    def prep_adata(self, mode):
        """Retreive omic-specific AnnData objects and index by desired batches"""
        self.mode = mode
        seed_everything(self.seed)

        # assumes that GEX is always available
        base_key = list(self.seq_types.keys())[0]
        self.fdir = glob.glob(f"{self.dataset}/*{base_key}*.h5ad")[0]
        adata = ad.read_h5ad(self.fdir, backed="r")


        check_cpu_usage("prepping batches")

        batch_effect_idx = self.prep_batches(adata)

        mode_keys = self.train_test_split(adata, mode, batch_effect_idx)
            
        if len(glob.glob(f"{self.ppdir}/pp*.h5ad")) > 0:
            adata_omics_all = {}
            for k in self.omic_types:
                adata_omics_all[k] = ad.read_h5ad(f"{self.ppdir}/pp_{k}_rename.h5ad")
                
        else:
            if len(self.seq_types.keys()) > 1:
                aux_key = list(self.seq_types.keys())[1]
                aux_fdir = glob.glob(f"{self.dataset}/*{aux_key}*.h5ad")[0]
                adata_aux = ad.read_h5ad(aux_fdir, backed="r")

                aux_batch_effect_idx = self.prep_batches(adata_aux)

                if 'dataset' in self.batch_effect:
                    adata.obs[self.batch_effect] = pd.Categorical(np.full(adata.shape[0], base_key), categories=[base_key])
                    adata_aux.obs[self.batch_effect] = pd.Categorical(np.full(adata_aux.shape[0], aux_key), categories=[aux_key])
                    
                    # rename batches used for plotting
                    renamed_batches = adata.obs[self.plot_batch_effect].astype(str) + f"_{base_key}"
                    adata.obs[self.plot_batch_effect] = pd.Categorical(renamed_batches, categories=np.unique(renamed_batches))
                    renamed_batches = adata_aux.obs[self.plot_batch_effect].astype(str) + f"_{aux_key}"
                    adata_aux.obs[self.plot_batch_effect] = adata_aux.obs[self.plot_batch_effect].astype(str) + f"_{aux_key}"

                else:
                    # adds dataset labels
                    renamed_batches = adata.obs[self.batch_effect].astype(str) + f"_{base_key}"
                    adata.obs[self.batch_effect] = pd.Categorical(renamed_batches, categories=np.unique(renamed_batches))
                    renamed_batches = adata_aux.obs[self.batch_effect].astype(str) + f"_{aux_key}"
                    adata_aux.obs[self.batch_effect] = adata_aux.obs[self.batch_effect].astype(str) + f"_{aux_key}"

                # set as separate batches for now
                adata_omics_all = self.split_adata_omics(
                    adata,
                    self.omic_types,
                    mode_keys,
                    batch_effect_idx,
                    aux_adata=adata_aux,
                    aux_batch_effect_idx=aux_batch_effect_idx,
                )

                del adata_aux
                gc.collect()
            else:
                if 'dataset' not in self.batch_effect:
                    renamed_batches = adata.obs[self.batch_effect].astype(str) + f"_{base_key}"
                    adata.obs[self.batch_effect] = pd.Categorical(renamed_batches, categories=np.unique(renamed_batches))
                adata_omics_all = self.split_adata_omics(
                    adata, self.omic_types, mode_keys, batch_effect_idx
                )

                gc.collect()

            check_cpu_usage("merged")

            # remove samples with empty features across modalities (after preprocessing)
            obs_names_del = None
            for omic in adata_omics_all.keys():
                if obs_names_del is None:
                    obs_names_del = np.array(
                        adata_omics_all[omic][
                            (adata_omics_all[omic].X.sum(1) == 0) | np.isnan(adata_omics_all[omic].X.sum(1))
                        ].obs_names
                    )
                else:
                    obs_names_del = np.concatenate(
                        (
                            obs_names_del,
                            np.array(
                                adata_omics_all[omic][
                                    (adata_omics_all[omic].X.sum(1) == 0) | np.isnan(adata_omics_all[omic].X.sum(1))
                                ].obs_names
                            ),
                        )
                    )
                check_cpu_usage(f"preprocessed for omic {omic}")


            if obs_names_del is not None:
                for omic in adata_omics_all.keys():
                    adata_omics_all[omic] = adata_omics_all[omic][
                        ~np.isin(adata_omics_all[omic].obs_names, obs_names_del)
                    ]    

            for k in adata_omics_all.keys():
                adata_omics_all[k].write_h5ad(f"{self.ppdir}/pp_{k}_rename.h5ad")
        del adata 

        # split combined dataset into train & validation/test sets
        adata_omics_ret = []
        for mode_ind, (mode_group, idx_groups) in enumerate(mode_keys.items()):
            if idx_groups is None:
                adata_omics_ret.append(None)
                continue

            adata_omics_mode = {}
            unique_batches = np.array([])

            if 'dataset' not in self.batch_effect:

                for k, a in adata_omics_all.items():
                    if k not in self.omic_types: continue
                    # temporarily strips dataset key for identifying batches for training/testing
                    batch_vals_common = np.array([i.rsplit('_', 1)[0] if '_' in i else i for i in adata_omics_all[k].obs[self.split_key]]).astype(object)
                    idx_all = np.isin(batch_vals_common, idx_groups).nonzero()[0]

                    adata_omics_mode[k] = adata_omics_all[k][idx_all.astype(int)]
                    dataset_keys = np.array([i.rsplit('_', 1)[-1] for i in adata_omics_mode[k].obs[self.batch_effect]]).astype(object)
                    
                    adata_omics_mode[k] = adata_omics_mode[k][np.isin(dataset_keys, list(self.seq_types.keys()))]
                   
                    dataset_keys = np.array([i.rsplit('_', 1)[-1] for i in adata_omics_mode[k].obs[self.batch_effect]]).astype(object)
                    
                    adata_omics_mode[k].obs['dataset'] = pd.Categorical(dataset_keys, categories=list(self.seq_types.keys()))
                    unique_batches = np.concatenate((unique_batches, np.unique(np.array(adata_omics_mode[k].obs[self.split_key].values))))
            else:
                for k, a in adata_omics_all.items():
                    batch_vals_common = np.array([i.rsplit('_', 1)[0] if '_' in i else i for i in adata_omics_all[k].obs[self.split_key]]).astype(object)
                    idx_all = np.isin(batch_vals_common, idx_groups).nonzero()[0]
                    adata_omics_mode[k] = adata_omics_all[k][idx_all.astype(int)]
                    unique_batches = list(self.seq_types.keys())
                    adata_omics_mode[k] = adata_omics_mode[k][np.isin(adata_omics_mode[k].obs['dataset'], unique_batches)]
            # inject batch indices
            unique_batches = adata_omics_mode['GEX'].obs[self.batch_effect].unique()
            index_to_string = {string: idx for idx, string in enumerate(unique_batches)}

            for k, a in adata_omics_mode.items():
                try:
                    batch_idx = np.vectorize(index_to_string.get)(np.array(adata_omics_mode[k].obs[self.batch_effect].values))
                except:
                    import ipdb ; ipdb.set_trace()
                adata_omics_mode[k].obs['batch_indices'] = pd.Categorical(batch_idx, categories=np.unique(batch_idx))

            adata_omics_ret.append(adata_omics_mode)

            gc.collect()

        return adata_omics_ret

    def postprocess(self, adata_omics_int):
        for omic, adata in adata_omics_int.items():
            sc.pp.highly_variable_genes(adata, batch_key=self.batch_effect)
            adata_omics_int[omic] = adata[:, adata.var["highly_variable"].values]
        return adata_omics_int