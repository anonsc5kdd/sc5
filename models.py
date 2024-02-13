import numpy as np
from sklearn import preprocessing
import itertools
import os
import scanpy as sc
import scipy.spatial
import sklearn.decomposition
import sys
from tqdm import tqdm
import anndata as ad
from utils import *
from utils import get_knn_graph
import time
import torch
import torch.nn.functional as F
import pickle as pkl
from matplotlib import cm
import colorcet as cc


sys.path.insert(1, "moETM/")

from train_moetm import Train_moETM
from trainer_moetm_patchwork import Trainer_moETM_patchwork
from trainer_moetm import Trainer_moETM
from build_model import build_moETM
import mo_utils
from mo_utils import calc_weight
import eval_utils
from utils import create_graph


def collect_int(adata, omic_types, exp, model_name, agg="concat", int_key="int"):
    exp = exp[exp.rfind(model_name) :]

    omic_types_combine = [omic_type for omic_type in omic_types if omic_type != 'GEX']

    if "full" not in adata.keys():
        adata['full'] = adata['GEX'].copy()

        for omic_combine in omic_types_combine:
            adata['full'] = ad.concat([adata['full'], adata[omic_combine]], axis=1, merge='first')
            for k in adata[omic_combine].obsm.keys():
                if k not in adata[omic_combine].obsm.keys():
                    adata["full"].obsm[k] = adata[omic_combine].obsm[k] 
            for k in adata[omic_combine].uns.keys():
                if k not in adata[omic_combine].uns.keys():
                    adata["full"].uns[k] = adata[omic_combine].uns[k] 
            adata["full"].obs_names_make_unique()


    return adata


class AlignmentModel:
    def __init__(self, settings):
        self.settings = settings
        self.exp = settings["EXP"]
        self.epochs = settings["MODEL"]["EPOCHS"]
        self.ft_epochs = settings["MODEL"]["FT_EPOCHS"]
        self.model_name = settings["MODEL"]["NAME"]
        self.dataset = settings["DATASET"]["NAME"]
        self.omic_types = settings["DATASET"]["OMICS"]
        self.seq_type = settings["DATASET"]["SEQ_TYPES"]
        self.hidden_dim = settings["MODEL"]["HIDDEN_DIM"]
        self.lr = float(settings["MODEL"]["LR"])
        self.outdir = settings["MODEL"]["OUTDIR"]
        self.clf_key = settings["EVAL"]["CLF_KEY"]
        self.batch_size = int(settings["DATASET"]["BATCH_SIZE"])
        self.target_inference = settings["MODEL"]["TARGET_INFERENCE"]
        self.counts_key = settings['DATASET']['RAW_COUNTS_KEY']
        self.batch_effect = settings["DATASET"]["BATCH_EFFECT"]
        self.plot_batch_effect = settings["DATASET"]["PLOT_BATCH_EFFECT"]
        self.setting_exp = settings["SETTING_EXP"]
        self.run_mode = settings["RUN_MODE"]
        
        self.missing_mod_train = settings["PATCHWORK"]["MISSING_TRAIN"]
        self.missing_mod_test = settings["PATCHWORK"]["MISSING_TEST"]

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        sc.settings.figdir = self.outdir
        self.rep_vis = "X"
        self.model_path = None
        self.guidance_hvf = None

        
    def get_ll(self, x_mod1, x_mod2, pred1, pred2):
        """Log likelihood calculation"""
        mod1_ll, mod2_ll = None, None
        if x_mod1 is not None:
            mod1_ll = (
                (F.log_softmax(torch.tensor(pred1), dim=-1) * torch.tensor(x_mod1))
                .sum(-1)
                .mean()
                .item()
            )
        if x_mod2 is not None:
            mod2_ll = (
                (F.log_softmax(torch.tensor(pred2), dim=-1) * torch.tensor(x_mod2))
                .sum(-1)
                .mean()
                .item()
            )
        return mod1_ll, mod2_ll

    def configure(self, adata_omics, mode, adata_omics_val=None):
        self.mode = mode
        return adata_omics, adata_omics_val

    def fit(self, adata_omics, adata_omics_val=None):
        return adata_omics

    def impute(self, adata_omics, source_key, target_key):
        return adata_omics

def append_substring(text, substring_to_append):
    return substring_to_append + "_" + text

class moETM(AlignmentModel):
    def __init__(self, settings):
        super().__init__(settings)
        self.Eval_kwargs = {}
        self.Eval_kwargs["batch_col"] = self.batch_effect
        self.Eval_kwargs["plot_batch_col"] = self.plot_batch_effect
        self.Eval_kwargs["cell_type_col"] = self.clf_key
        self.Eval_kwargs["clustering_method"] = "louvain"
        #self.Eval_kwargs["resolutions"] = [0.75, 1, 1.25, 1.5, 1.75]
        self.Eval_kwargs["resolutions"] = np.arange(0.75, 2, 0.1)
        if not os.path.exists(f"{self.outdir}/result_fig"):
            os.makedirs(f"{self.outdir}/result_fig")
        self.Eval_kwargs["plot_dir"] = f"{self.outdir}/result_fig"
        self.logging_interval = settings['MODEL']['LOGGING_INTERVAL']

    def configure(self, adata_omics, mode, adata_omics_val=None):
        super().configure(adata_omics, mode, adata_omics_val)

        shared_outdir = self.outdir.rsplit('/',1)[0]
        # predefine a color-map for cell types & batches
        if not os.path.exists(f'{shared_outdir}/color_map2.pkl'):
            colormap = {}
            distinct_colors = cc.glasbey_light
            for k in adata_omics_val:
                adata_omics_val[k].obs[self.batch_effect] = (
                    adata_omics_val[k]
                    .obs[self.batch_effect]
                    .apply(append_substring, substring_to_append="test")
                )
            adata_full = ad.concat([adata_omics['GEX'], adata_omics_val['GEX']],merge='same')
            unique_categories = np.unique(np.array(adata_full.obs['cell_type']))
            num_categories = len(unique_categories)
            celltype_to_color = dict(zip(unique_categories, distinct_colors[:num_categories]))
            colormap.update(celltype_to_color)

            distinct_colors = cc.glasbey_category10
            unique_categories = np.unique(np.array(adata_full.obs[self.batch_effect]))
            num_categories = len(unique_categories)
            batch_to_color = dict(zip(unique_categories, distinct_colors[:num_categories]))
            colormap.update(batch_to_color)

            distinct_colors = cc.glasbey_category10
            unique_categories = np.unique(np.array(adata_full.obs[self.plot_batch_effect]))
            num_categories = len(unique_categories)
            batch_to_color = dict(zip(unique_categories, distinct_colors[:num_categories]))
            colormap.update(batch_to_color)

            with open(f'{shared_outdir}/color_map2.pkl', 'wb') as fout:
                pkl.dump(colormap,fout)
            del adata_full
        else:
            with open(f'{shared_outdir}/color_map2.pkl', 'rb') as fin:
                colormap = pkl.load(fin)


        self.Eval_kwargs['cmap'] = colormap
        seed = int(self.run_mode.split("seed_")[-1]) if 'seed_' in self.run_mode else 1234


        # graph construction/loading
        neighbor_path = self.outdir + f"/{mode}_neighbors_{seed}"
        adatas_train = list(adata_omics.values())

        if "graph" in self.exp:
            adata_omics, adata_omics_val = create_graph(
                self.omic_types, adata_omics, mode, adata_omics_val, neighbor_path
            )

        input_dims = [adata.shape[-1] for adata in adata_omics.values()]

        self.hidden_dim = 400
        num_topic = 100

        batch_index = adata_omics[self.omic_types[0]].obs["batch_indices"]
        num_batch = len(batch_index.unique())

        if "patchwork" in self.exp:
            flavor = "patchwork"
        else:
            flavor = "original"

        if "graph" in self.exp:
            flavor += "_graph"

        all_domains = np.array([])
        for k in adata_omics.keys():
            all_domains = np.concatenate((all_domains,np.unique(adata_omics[k].obs['batch_indices'])))

        encoders, decoder, optimizer = build_moETM(
            input_dims,
            num_batch,
            num_topic=num_topic,
            emd_dim=self.hidden_dim,
            lr=self.lr,
            flavor=flavor,
            domains=torch.tensor(np.unique(all_domains).astype(int)).cuda(),
        )

        if "patchwork" in self.exp:
            self.backbone = Trainer_moETM_patchwork(
                adata_omics['GEX'].shape[0], encoders, decoder, optimizer, lr=self.lr, exp=self.exp, run_mode=self.run_mode, logging_interval=self.logging_interval
            )
        else:
            self.backbone = Trainer_moETM(adata_omics['GEX'].shape[0], encoders, decoder, optimizer, lr=self.lr, exp=self.exp, run_mode=self.run_mode, logging_interval=self.logging_interval)
        return adata_omics, adata_omics_val

    def fit(self, adata_omics, adata_omics_val=None):
        adatas = list(adata_omics.values())
        adatas_val = list(adata_omics_val.values())
        
        if 'full' in adata_omics: del adata_omics['full']
        if 'full' in adata_omics_val: del adata_omics_val['full']

        max_cells = 0
        for i in adatas:
            max_cells_i = i.shape[0]
            if max_cells_i > max_cells:
                max_cells = max_cells_i
                max_cell_names = np.array(i.obs_names)

        max_cells_val = 0
        for i in adatas_val:
            max_cells_i = i.shape[0]
            if max_cells_i > max_cells_val:
                max_cells_val = max_cells_i
                max_cell_names_val = np.array(i.obs_names)

        num_batches = self.batch_size
        self.LIST = list(np.arange(0, max_cells))
        start_time = time.time()

        batch_index = torch.tensor(
            adata_omics[self.omic_types[0]].obs["batch_indices"]
        ).cuda()
        batch_index_val = torch.tensor(
            adata_omics_val[self.omic_types[0]].obs["batch_indices"]
        ).cuda()

        train_set = [adata_omics, batch_index]
        test_set = [adata_omics_val, batch_index_val]


        graphs_train, graphs_val = [None], [None]
        if "graph" in self.exp:
            graphs_train = []
            for omic, adata in adata_omics.items():
                train_adj = adata.obsp["connectivities"]
                graph_train = dgl.graph(train_adj.nonzero())#.to("cuda")
                graph_train.edata["weight"] = torch.ones(
                    [graph_train.number_of_edges(), 1], dtype=torch.float32
                )#.to("cuda")
                graphs_train.append(graph_train)
            graphs_val = []
            for omic, adata in adata_omics_val.items():
                val_adj = adata.obsp["connectivities"]
                graph_val = dgl.graph(val_adj.nonzero())#.to("cuda")
                graph_val.edata["weight"] = torch.ones(
                    [graph_val.number_of_edges(), 1], dtype=torch.float32
                )#.to("cuda")
                graphs_val.append(graph_val)

        Train_moETM(
            self.settings,
            self.exp,
            self.outdir,
            f"output/{self.dataset}/{self.setting_exp}",
            self.backbone,
            self.epochs,
            self.ft_epochs,
            self.batch_effect,
            self.batch_size,
            self.omic_types,
            train_set,
            test_set,
            self.Eval_kwargs,
            missing_mod_train=self.missing_mod_train,
            missing_mod_test=self.missing_mod_test,
            graph_train=graphs_train,
            graph_test=graphs_val,
        )

        adata_ret, adata_ret_ft, adata_ret_ft_val = {}, {}, {}
        for seq_type in self.seq_type.keys():
            adata_ret[seq_type] = collect_int(
                {k: v[v.obs['dataset']==seq_type] for k, v in train_adata.items()},
                self.seq_type[seq_type],
                self.exp,
                self.model_name,
                agg="concat",
                int_key="moetm",
            )

            adata_ret_ft[seq_type] = collect_int(
                {k: v[v.obs['dataset']==seq_type] for k, v in train_adata.items()},
                self.seq_type[seq_type],
                self.exp,
                self.model_name,
                agg="concat",
                int_key="moetm",
            )

            adata_ret_ft_val[seq_type] = collect_int(
                {k: v[v.obs['dataset']==seq_type] for k, v in test_adata.items()},
                self.seq_type[seq_type],
                self.exp,
                self.model_name,
                agg="concat",
                int_key="moetm",
            )
    
        return adata_ret, adata_ret_ft, adata_ret_ft_val