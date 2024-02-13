import argparse
import yaml
from dataloader import DataLoader
from eval_classes.visualizer import Visualizer
from eval import Evaluation
from utils import *
import gc
import glob
import os
import warnings

warnings.filterwarnings("ignore")


def run_exps(args, settings):
    # load arguments
    dataset, exp, split, eval_flag = (
        settings["DATASET"]["NAME"],
        settings["EXP"],
        settings["DATASET"]["SPLIT"],
        settings["MODEL"]["EVAL"],
    )

    print("running exp", exp, "on dataset", dataset, "under mode", settings["RUN_MODE"])
    setting_exp = settings["SETTING_EXP"]
    datadir = settings["DATASET"]["DIR"]
    seq_types = settings["DATASET"]["SEQ_TYPES"]
    seq_type_names = list(seq_types.keys())
    model_name = settings["MODEL"]["NAME"]
    rename = settings["MODEL"]["RENAME"]
    omic_types = settings["DATASET"]["OMICS"]
    exp_dir = exp
    config = args.config
    
    outdir = f"output/{dataset}/{setting_exp}"
    ppdir = f"output/{dataset}/{setting_exp}"

    settings["MODEL"]["OUTDIR"] = f"{outdir}/{model_name}/{exp}"
    settings["DATASET"]["PPDIR"] = ppdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(ppdir):
        os.makedirs(ppdir)

    # load data
    dl = DataLoader(settings)
    vis = Visualizer(settings)
    eval = Evaluation(settings)

    # only evaluate saved runs
    if eval_flag == True:
        def run_eval(mode, seq_type, omics):
            model_name_ = exp.split("_")[0]
            if 'train' in mode:
                adata_omics_eval, _ = dl.prep_adata('train')
            else:
                _, adata_omics_eval = dl.prep_adata('train')
            save_path = f"{outdir}/{mode}_{seq_type}_{model_name_}_combine.h5ad"
            adata = ad.read_h5ad(save_path)
            
            adata = adata[adata.obs['dataset']==seq_type]

            adata_omics_eval = {k: v[v.obs['dataset']==seq_type] for k, v in adata_omics_eval.items()}
            adata_omics_eval['full'] = adata_omics_eval['GEX']
            

            for i in adata.obsm.keys():
                adata_omics_eval['full'].obsm[i] = adata.obsm[i].astype('float32')
            #adata.obs['cell_type'] = adata_other.obs['cell_type'][np.where(np.isin(adata.obs_names, obs_names))[0]]

            adata_omics_eval['full'] = adata
            
    
            exp_keys = [model_name]
            rename_keys = [rename]

            print(f"{mode} EVAL ON {dataset} for dataset {seq_type}, ON KEYS: {exp_keys}")

            eval.omic_types = omics
            eval.seq_type = seq_type

            adata_omics_eval = eval.setup(
                mode, omics, adata_omics_eval, exp_keys, outdir, rename_keys
            )
            
            eval.benchmark(adata_omics_eval)

            del adata, adata_omics_eval ; gc.collect()

        #for seq_type in seq_type_names:
        for seq_type in ['multiome']:
            omics = seq_types[seq_type]
            #run_eval("train_ft", seq_type, omics)
            run_eval("test_ft", seq_type, omics)
        return

    # model setup
    model_name = settings["MODEL"]["NAME"]
    model = moETM(settings)
    
    # preprocessing
    if model_name not in ["pamona", "scot"]:
        adata_omics_train, adata_omics_val = dl.prep_adata("train")    
        train_obs_names = adata_omics_train[omic_types[0]].obs_names
        if model_name == "vis_only":
            vis.run_vis(adata_omics_train)
            return

        # perform integration
        adata_omics_train, adata_omics_val = model.configure(
            adata_omics_train, "train", adata_omics_val=adata_omics_val
        )

        adata_omics_train_int, adata_omics_train_ft_int, adata_omics_val_ft_int = model.fit(
            adata_omics_train, adata_omics_val=adata_omics_val
        )

        if model_name in ["moetm"]:
            return
        else:
            save_int(adata_omics_train_int, set_key, f"{outdir}/train.h5ad", settings)
            if adata_omics_val_int is not None:
                save_int(adata_omics_val_int, set_key, f"{outdir}/valid.h5ad", settings)

        del adata_omics_train, adata_omics_train_int
        gc.collect()

    # perform testing
    adata_omics_test, _ = dl.prep_adata("test")
    adata_omics_test, _ = model.configure(adata_omics_test, "test")
    test_obs_names = adata_omics_test[omic_types[0]].obs_names  # debug
    if model_name not in ["pamona", "scot", "proscrutes"]:
        assert np.intersect1d(train_obs_names, test_obs_names).shape[0] == 0

    adata_omics_test_int, _ = model.fit(adata_omics_train, adata_omics_test)

    save_int(adata_omics_test_int, f"{outdir}/test.h5ad", settings)


def load_yaml(config):
    with open(f"configs/{config}.yaml") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config name", default="neurips2021")
    parser.add_argument("--epochs", help="epochs", default=None)
    parser.add_argument("--run_mode", help="run mode", default=None)
    parser.add_argument("--n_comps", help="n_comps", default=50)
    parser.add_argument("--site", help="site", default=None)
    parser.add_argument("--model", help="model", default=None)
    parser.add_argument("--donor", help="donor", default=None)
    parser.add_argument("--status", help="status", default=None)
    parser.add_argument("--split_key", help="split_key", default=None)
    parser.add_argument("--split_frac", help="split_frac", default=None)
    parser.add_argument("--batch_key", help="batch effect", default=None)
    parser.add_argument("--plot_batch_key", help="batch effect for plotting", default=None)
    parser.add_argument("--batch_size", help="batch size", default=None)
    parser.add_argument("--hidden_dim", help="hidden_dim", default=200, type=int)
    parser.add_argument("--ot_weight", help="hidden_dim", default=1.0, type=float)
    parser.add_argument("--int_type", help="hidden_dim", default="", type=str)
    parser.add_argument("--task", help="task", default="joint", type=str)
    parser.add_argument("--graph_type", help="graph type", default="none", type=str)
    parser.add_argument("--agg_type", help="agg type", default="none", type=str)
    parser.add_argument(
        "--contrastive_weight", help="contrastive weight", default=0.0, type=float
    )
    parser.add_argument("--alpha", help="alpha", default=1.0, type=float)
    parser.add_argument("--beta", help="beta", default=0.0, type=float)
    parser.add_argument("--load", help="load", default="False", type=str)

    parser.add_argument(
        "--eval",
        help="no training, evaluation only using saved reps",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--eval_split",
        help="no training, evaluation only using saved reps",
        default=None,
    )
    parser.add_argument(
        "--exp", help="experiment name", default=None
    )
    parser.add_argument(
        "--setting_exp", help="batch effect, train/test split experiment tag", default=None
    )
    parser.add_argument(
        "--denoise", help="denoise before/after integration", default=None
    )
    parser.add_argument(
        "--exp_rename", help="denoise before/after integration", default=None
    )

    args = parser.parse_args()
    settings = load_yaml(args.config)
    batch_key = (
        args.batch_key
        if args.batch_key is not None
        else settings["DATASET"]["BATCH_EFFECT"]
    )
    plot_batch_key = (
        args.plot_batch_key
        if args.plot_batch_key is not None
        else settings["DATASET"]["PLOT_BATCH_EFFECT"]
    )

    # model config
    settings["DATASET"]["BATCH_EFFECT"] = batch_key
    settings["DATASET"]["PLOT_BATCH_EFFECT"] = plot_batch_key
    if args.model != None:
        settings["MODEL"]["NAME"] = args.model

    if args.exp_rename != None:
        settings["MODEL"]["RENAME"] = args.exp_rename
    else:
        settings["MODEL"]["RENAME"] = ""

    if args.setting_exp is not None:
        # settings['EXP'] = args.exp
        settings["SETTING_EXP"] = args.setting_exp

    settings["DATASET"]["N_COMPS"] = int(args.n_comps)

    if args.graph_type != "none":
        settings["SC_MODEL"]["GRAPH_TYPE"] = args.graph_type

    if args.agg_type != "none":
        settings["SC_MODEL"]["AGG_TYPE"] = args.agg_type

    if args.contrastive_weight != 0.0:
        settings["SC_MODEL"]["CONTRASTIVE_WEIGHT"] = args.contrastive_weight

    if args.alpha != 1.0:
        settings["SC_MODEL"]["ALPHA"] = args.alpha

    if args.beta != 0.0:
        settings["SC_MODEL"]["BETA"] = args.beta

    if args.task != "joint":
        settings["SC_MODEL"]["TASK"] = args.task

    # data config
    if args.site != None:
        settings["DATASET"]["SITE"] = args.site
    if args.status != None:
        settings["DATASET"]["STATUS"] = args.status
    if args.donor != None:
        settings["DATASET"]["DONOR"] = args.donor
    if args.split_key != None:
        settings["DATASET"]["SPLIT"] = args.split_key
    if args.split_frac != None:
        settings["DATASET"]["SPLIT_FRAC"] = args.split_frac
    if args.batch_size != None:
        settings["DATASET"]["BATCH_SIZE"] = args.batch_size
    if args.hidden_dim != 50:
        settings["MODEL"]["HIDDEN_DIM"] = args.hidden_dim
    if args.ot_weight != 1:
        settings["MODEL"]["OT_WEIGHT"] = args.ot_weight
    settings["MODEL"]["INT_TYPE"] = "v"
    if args.epochs is not None:
        settings["MODEL"]["EPOCHS"] = args.epochs
    if args.int_type != "":
        settings["MODEL"]["INT_TYPE"] = args.int_type
    if args.eval != None:
        settings["MODEL"]["EVAL"] = args.eval
    if args.denoise != None:
        if args.denoise == "pre" or args.denoise == "both":
            settings["DENOISE"]["PRE"] = True
        if args.denoise == "post" or args.denoise == "both":
            settings["DENOISE"]["POST"] = True

    if args.exp is not None:
        settings["EXP"] = args.exp
    
    if args.run_mode is not None:
        settings["RUN_MODE"] = args.run_mode

    run_exps(args, settings)
