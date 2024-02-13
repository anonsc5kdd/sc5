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
from trainer_moetm_patchwork import Trainer_moETM_patchwork
import glob
import copy
import random
import re
import logging
import time

import scanpy as sc
import anndata as ad

from moetm_utils import *
from moetm_dataloader import *
from layers import encoder, decoder, graph_encoder, decoder_patchwork


def save_int(adata_omic, set_key, save_path, settings, partial=False):
    """Save integrated features into AnnData object"""

    omic_types = settings["DATASET"]["OMICS"]
    model_keys = np.array(adata_omic[omic_types[0]].obsm_keys())
    model_key = model_keys[np.char.startswith(model_keys, "X_")][0]


    if os.path.exists(save_path):
        adata = ad.read_h5ad(save_path)

        if partial == True:
            split_key = settings["DATASET"]["SPLIT_KEY"]
            test_idx = np.intersect1d(
                adata_omic["full"].obs[split_key][
                    adata_omic["full"].obs["dataset"] == set_key
                ],
                np.array(
                    adata.obs[split_key][
                        adata_omic["full"].obs["dataset"] == set_key
                    ].cat.categories
                ),
                return_indices=True,
            )[1]
            adata.obsm[model_key] = adata_omic["full"].obsm[model_key][
                adata_omic["full"].obs["dataset"] == set_key
            ][test_idx]
        else:
            for i in adata_omic["full"][
                adata_omic["full"].obs["dataset"] == set_key
            ].obs.keys():
                try:
                    adata.obs[i] = adata_omic["full"][
                        adata_omic["full"].obs["dataset"] == set_key
                    ].obs[i]
                except Exception as e:
                    print(e)
                    pass
            for i in adata_omic["full"].obsm.keys():
                if "recons" in i or "impute" in i:
                    continue
                try:
                    adata.obsm[i] = adata_omic["full"][
                        adata_omic["full"].obs["dataset"] == set_key
                    ].obsm[i]
                except Exception as e:
                    print(e)
                    import ipdb

                    ipdb.set_trace()
                    pass
            for i in adata_omic["full"][
                adata_omic["full"].obs["dataset"] == set_key
            ].uns.keys():
                try:
                    adata.uns[i] = adata_omic["full"][
                        adata_omic["full"].obs["dataset"] == set_key
                    ].uns[i]
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


def collect_int(adata, omic_types, exp, model_name, agg="concat", int_key="int"):
    exp = exp[exp.rfind(model_name) :]

    # NOTE: assumes that all batches contain GEX
    omic_types_combine = [omic_type for omic_type in omic_types if omic_type != "GEX"]

    if "full" not in adata.keys():
        adata["full"] = adata["GEX"].copy()

        for omic_combine in omic_types_combine:
            adata["full"] = ad.concat(
                [adata["full"], adata[omic_combine]], axis=1, merge="first"
            )
            for k in adata[omic_combine].obsm.keys():
                if k not in adata[omic_combine].obsm.keys():
                    adata["full"].obsm[k] = adata[omic_combine].obsm[k]
            for k in adata[omic_combine].uns.keys():
                if k not in adata[omic_combine].uns.keys():
                    adata["full"].uns[k] = adata[omic_combine].uns[k]
            adata["full"].obs_names_make_unique()
    return adata


def Train_moETM(
    settings,
    exp,
    outdir,
    savedir,
    trainer,
    Total_epoch,
    Total_ft_epoch,
    batch_effect,
    batch_size,
    modalities,
    train_dataset,
    test_dataset,
    Eval_kwargs,
    missing_mod_train={},
    missing_mod_test={},
    graph_train=[None],
    graph_test=[None],
):
    if settings["MODEL"]["TRAINDIR"] == "":
        train_outdir = outdir
    else:
        path = settings["MODEL"]["TRAINDIR"]
        train_outdir = f"{savedir}/moetm/{path}"

    if settings["MODEL"]["FTDIR"] == "":
        ft_outdir = outdir
    else:
        path = settings["MODEL"]["FTDIR"]
        ft_outdir = f"{savedir}/moetm/{path}"

    seed = (
        int(settings["RUN_MODE"].split("seed_")[-1])
        if "seed_" in settings["RUN_MODE"]
        else 1234
    )

    trainer.kl_setting = settings["MODEL"]["KL_SETTING"]

    logging.basicConfig(
        level=logging.INFO,
        format=f"RUN {exp} - %(asctime)s - %(levelname)s - %(message)s",
    )

    method = exp.split("_")[0]

    # data intialization
    seed_everything(seed)
    train_adata, batch_index = train_dataset
    test_adata, batch_index_test = test_dataset
    X_mods = [np.array(adata.X.todense()) for adata in train_adata.values()]
    X_mods = [torch.from_numpy(i).float() for i in X_mods]

    domains = torch.unique(batch_index)

    if "patchwork" in exp:
        trainer.init_priors(domains)

    if "full" in test_adata.keys():
        del test_adata["full"]

    trainer.reset_optimizer()

    epoch_start = trainer.load_ckpt(train_outdir, mode="train") + 1
    # epoch_start=0
    print("loaded train model", train_outdir, epoch_start)

    trainer.modalities = np.array(list(train_adata.keys()))

    dataset_keys = list(train_adata["GEX"].obs["dataset"].unique())

    training_set = moETM_dataloader(
        batch_effect,
        train_adata=train_adata,
        X_mods=X_mods,
        missing_mod_train=missing_mod_train,
        mode="train",
        graph_train=graph_train,
    )

    trainer.dataloader = training_set

    trainer.set_mode("train")
    best_val_loss, best_pearson, counter, patience = (
        torch.inf,
        0,
        0,
        int(settings["MODEL"]["PATIENCE"]),
    )
    best_epoch = -1

    print(
        "HYPERPARAMS: KL -> ",
        settings["MODEL"]["KL"],
        "BETA -> ",
        settings["MODEL"]["BETA"],
        "\nPATIENCE -> ",
        patience,
        "KL_SETTING -> ",
        settings["MODEL"]["KL_SETTING"],
    )
    trainer.logging_interval = 3000
    for epoch in range(epoch_start, Total_epoch):
        if (
            "inference_only" in settings["RUN_MODE"]
            or "finetune_only" in settings["RUN_MODE"]
        ) and epoch > 0:
            break
        hyperparams = {
            "kl": calc_weight(
                epoch, Total_epoch, 0, 1 / 3, 0, float(settings["MODEL"]["KL"])
            ),
            "beta": (
                calc_weight(
                    epoch, Total_epoch, 0, 1 / 3, 0, float(settings["MODEL"]["KL"])
                )
                if float(settings["MODEL"]["BETA"]) == -1
                else float(settings["MODEL"]["BETA"])
            ),
            "ncl": 1e-3,
        }

        train_loss_dict = trainer.run_epoch(
            train_outdir,
            hyperparams,
            epoch,
            Total_epoch,
            batch_size,
            Eval_kwargs,
            mode="train",
            inference=True,
        )

        if "baseline" in exp or settings["MODEL"]["FTDIR"] != "":
            break
        # val_loss = train_loss_dict["Val_loss"]
        val_loss = torch.stack(
            [i.mean() for i in train_loss_dict["Val_NLL_mods"]]
        ).sum()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            trainer.save_params(
                train_outdir, key=f"train", epoch=epoch, prev_ckpt=best_epoch
            )
            best_epoch = epoch
        else:
            if counter == patience:
                logging.info(f"Train early stopping at epoch {epoch}")
                int_eval_dict = trainer.collect_results(
                    epoch, Eval_kwargs, "train", plot=False
                )
                break
            else:
                counter += 1
                logging.info(f"Counter raised to {counter}")

        if "baseline" in exp:
            break

    trainer.set_mode("eval")

    loaded_best_epoch = trainer.load_ckpt(train_outdir, mode="train")
    print("loaded best epoch", loaded_best_epoch)

    if epoch_start == 0 and "baseline" not in exp:
        with torch.no_grad():
            int_eval_dict_train = trainer.inference(
                torch.arange(len(train_adata["GEX"])).cuda(),
                loaded_best_epoch,
                "train",
                Eval_kwargs,
                plot=False,  # (True if 'plot' in exp else False),
                save=True,
            )

            train_adata = int_eval_dict_train["train_save"]

            adata_ret_train = {}

            modalities = {"cite": ["GEX", "ADT"], "multiome": ["GEX", "ATAC"]}

            for seq_type in train_adata["GEX"].obs["dataset"].unique():
                adata_ret_train[seq_type] = collect_int(
                    {
                        k: v[v.obs["dataset"] == seq_type]
                        for k, v in train_adata.items()
                    },
                    modalities[seq_type],
                    exp,
                    "moetm",
                    agg="concat",
                    int_key="moetm",
                )

                try:
                    save_int(
                        adata_ret_train[seq_type],
                        seq_type,
                        f"{savedir}/train_{seq_type}_{method}_combine.h5ad",
                        settings,
                    )
                except Exception as e:
                    print(e)

                for i in adata_ret_train[seq_type]:
                    del i
                    gc.collect()

            del batch_index
            gc.collect()
            torch.cuda.empty_cache()

    seed_everything(seed)
    # FINE-TUNING
    test_X_mods = [
        np.array(test_dataset[0][k].X.todense())
        for k in list(test_dataset[0].keys())
        if k != "full"
    ]
    test_X_mods = [torch.from_numpy(i).float() for i in test_X_mods]

    ft_set = moETM_dataloader(
        batch_effect,
        train_adata=train_adata,
        X_mods=X_mods,
        missing_mod_train=missing_mod_train,
        test_adata=test_adata,
        test_X_mods=test_X_mods,
        missing_mod_test=missing_mod_test,
        mode="fine-tune",
        graph_train=graph_train,
        graph_test=graph_test,
    )

    for i in test_X_mods:
        del i
    for i in X_mods:
        del i

    # updates architecture for fine-tuning with added test set
    domains_add = torch.masked_select(
        ft_set.batch_list_mods[0],
        ~torch.isin(
            ft_set.batch_list_mods[0],
            torch.tensor(train_adata["GEX"].obs["batch_indices"].values).cuda(),
        ),
    ).unique()

    trainer.dataloader = ft_set
    trainer.modalities = np.array(ft_set.modalities)

    # clean mem
    for k, v in list(train_adata.items()):
        del train_adata[k]
    for k, v in list(test_adata.items()):
        del test_adata[k]
    gc.collect()

    trainer.add_ft_params(domains_add, ft_set.adata["GEX"].shape[0])
    trainer.kl_setting = settings["MODEL"]["KL_SETTING_FT"]

    # re-init model
    epoch_start = trainer.load_ckpt(ft_outdir, mode="ft") + 1

    print("loaded fine-tune model", epoch_start)

    trainer.set_mode("train")åå

    best_val_loss, best_pearson, counter, patience = (
        torch.inf,
        0,
        0,
        int(settings["MODEL"]["PATIENCE"]),
    )
    best_epoch = -1
    for epoch in range(epoch_start, Total_ft_epoch):
        if ("inference_only" in settings["RUN_MODE"]) and epoch > 0:
            break

        hyperparams = {
            "kl": calc_weight(
                epoch, Total_ft_epoch, 0, 1 / 3, 0, float(settings["MODEL"]["KL"])
            ),
            "beta": (
                calc_weight(
                    epoch, Total_ft_epoch, 0, 1 / 3, 0, float(settings["MODEL"]["KL"])
                )
                if float(settings["MODEL"]["BETA_FT"]) == -1
                else float(settings["MODEL"]["BETA_FT"])
            ),
            "ncl": 1e-3,
        }
        ft_loss_dict = trainer.run_epoch(
            ft_outdir,
            hyperparams,
            epoch,
            Total_ft_epoch,
            batch_size,
            Eval_kwargs,
            mode="ft",
        )

        if "baseline" in exp:
            break
        val_loss = torch.stack([i.mean() for i in ft_loss_dict["Val_NLL_mods"]]).sum()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            trainer.save_params(ft_outdir, key=f"ft", epoch=epoch, prev_ckpt=best_epoch)
            best_epoch = epoch
        else:
            if counter == patience:
                logging.info(f"Fine-tune early stopping at epoch {epoch}")
                int_eval_dict = trainer.collect_results(
                    epoch, Eval_kwargs, "ft", plot=False
                )
                break
            else:
                counter += 1
                logging.info(f"Counter raised to {counter}")

    trainer.set_mode("eval")

    loaded_best_epoch = trainer.load_ckpt(ft_outdir, mode="ft")
    print("loaded fine-tune model", loaded_best_epoch)
    # trainer.reset_optimizer()

    if "patchwork" in exp:
        for i in range(len(trainer.decoder.betas)):
            print(f"Norm of beta {i+1}: {torch.norm(trainer.decoder.betas[i], 2)}")
        print(f"Norm of alpha: {torch.norm(trainer.decoder.alpha, 2)}")

    with torch.no_grad():
        # save and/or plot train & test results
        int_eval_dict_train = trainer.inference(
            torch.tensor(
                (ft_set.adata["GEX"].obs["mode"].values == "0").nonzero()[0]
            ).cuda(),
            loaded_best_epoch,
            "train_ft",
            Eval_kwargs,
            plot=True,
            save=True,
        )

        int_eval_dict_test = trainer.inference(
            torch.tensor(
                (ft_set.adata["GEX"].obs["mode"].values == "1").nonzero()[0]
            ).cuda(),
            loaded_best_epoch,
            "test_ft",
            Eval_kwargs,
            plot=True,
            save=True,
        )

        train_ft_adata, test_ft_adata = (
            int_eval_dict_train["train_ft_save"],
            int_eval_dict_test["test_ft_save"],
        )

        adata_ret_train_ft, adata_ret_test_ft = {}, {}

        modalities = {"cite": ["GEX", "ADT"], "multiome": ["GEX", "ATAC"]}

        for seq_type in train_ft_adata["GEX"].obs["dataset"].unique():
            adata_ret_train_ft[seq_type] = collect_int(
                {k: v[v.obs["dataset"] == seq_type] for k, v in train_ft_adata.items()},
                modalities[seq_type],
                exp,
                "moetm",
                agg="concat",
                int_key="moetm",
            )

            try:
                save_int(
                    adata_ret_train_ft[seq_type],
                    seq_type,
                    f"{savedir}/train_ft_{seq_type}_{method}_combine.h5ad",
                    settings,
                )
            except Exception as e:
                print(e)

            for i in adata_ret_train_ft[seq_type]:
                del i
                gc.collect()

            adata_ret_test_ft[seq_type] = collect_int(
                {k: v[v.obs["dataset"] == seq_type] for k, v in test_ft_adata.items()},
                modalities[seq_type],
                exp,
                "moetm",
                agg="concat",
                int_key="moetm",
            )

            try:
                save_int(
                    adata_ret_test_ft[seq_type],
                    seq_type,
                    f"{savedir}/test_ft_{seq_type}_{method}_combine.h5ad",
                    settings,
                )
            except Exception as e:
                print(e)

            for i in adata_ret_test_ft[seq_type]:
                del i
                gc.collect()

        trainer.writer.flush()
        trainer.writer.close()

        return
