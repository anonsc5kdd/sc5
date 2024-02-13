import torch
import dgl
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

import anndata as ad
import time


def append_substring(text, substring_to_append):
    return substring_to_append + "_" + text


class moETM_dataloader:
    def __init__(
        self,
        batch_effect,
        train_adata=None,
        X_mods=None,
        missing_mod_train=None,
        test_adata=None,
        test_X_mods=None,
        missing_mod_test=None,
        mode="train",
        train_dataset=None,
        graph_train=[None],
        graph_test=[None],
    ):
        graph = graph_train
        if mode == "train":
            (
                adata,
                dataset_idx,
                batch_list_mods,
                X_mods,
                mask_mods,
                batch_index_mods,
                batch_list_mods,
                cell_avail,
                graph,
            ) = self.prep_batches_mode(
                train_adata, X_mods, missing_mod_train, batch_effect, graph_train
            )
        elif mode == "test":
            batch_list = train_dataset["batch_list"][0]
            for k in test_adata.keys():
                test_adata[k].obs["batch_indices"] = pd.Categorical(
                    test_adata[k].obs["batch_indices"].values.astype(int)
                    + batch_list.max().item()
                    + 1,
                    categories=np.unique(
                        test_adata[k].obs["batch_indices"].values.astype(int)
                        + batch_list.max().item()
                        + 1
                    ),
                )
                test_adata[k].obs[batch_effect] = (
                    test_adata[k]
                    .obs[batch_effect]
                    .apply(append_substring, substring_to_append="test")
                )

                (
                    adata,
                    dataset_idx,
                    batch_list_mods,
                    X_mods,
                    mask_mods,
                    batch_index_mods,
                    batch_list_mods,
                    cell_avail,
                    graph,
                ) = self.prep_batches_mode(
                    test_adata, test_X_mods, missing_mod_test, batch_effect, graph_test
                )

        elif mode == "fine-tune":
            batch_list = train_adata["GEX"].obs["batch_indices"].values.astype(int)
            # ensure that test batches are separate from train batches

            for k in test_adata.keys():
                test_adata[k].obs["batch_indices"] = pd.Categorical(
                    test_adata[k].obs["batch_indices"].values.astype(int)
                    + batch_list.max().item()
                    + 1,
                    categories=np.unique(
                        test_adata[k].obs["batch_indices"].values.astype(int)
                        + batch_list.max().item()
                        + 1
                    ),
                )

                test_adata[k].obs[batch_effect] = (
                    test_adata[k]
                    .obs[batch_effect]
                    .apply(append_substring, substring_to_append="test")
                )

            if graph_train[0] != None:
                graph = [
                    dgl.batch([graph_train[i], graph_test[i]]).to("cuda")
                    for i in range(len(graph_train))
                ]

            adata = {
                k: ad.concat(
                    [train_adata[k], test_adata[k]],
                    merge="same",
                    uns_merge="first",
                    label="mode",
                )
                for k in test_adata.keys()
            }
            X_mods_ft = [
                torch.vstack((X_mods[i], test_X_mods[i]))
                for i in range(len(test_X_mods))
            ]
            
            missing_mod_ft = {
                k: missing_mod_train[k] + missing_mod_test[k]
                for k, v in missing_mod_train.items()
            }

            (
                adata,
                dataset_idx,
                batch_list_mods,
                X_mods,
                mask_mods,
                batch_index_mods,
                batch_list_mods,
                cell_avail,
                graph,
            ) = self.prep_batches_mode(
                adata, X_mods_ft, missing_mod_ft, batch_effect, graph
            )

        self.adata = adata
        self.modalities = list(adata.keys())
        self.X_mods = X_mods
        self.mask_mods = mask_mods
        self.batch_index_mods = batch_index_mods
        self.batch_list_mods = batch_list_mods
        self.graph = graph
        self.cell_avail = cell_avail
        self.dataset_idx = dataset_idx

        dataset_keys = adata[self.modalities[0]].obs["dataset"].unique()
        self.batch_list_datasets = [
            torch.tensor(
                adata["GEX"][adata["GEX"].obs["dataset"] == dataset_key]
                .obs["batch_indices"]
                .unique()
            ).cuda()
            for dataset_key in dataset_keys
        ]

        """
        for i in range(len(self.X_mods)):
            try:
                assert (
                    self.batch_index_mods[i + 1].nonzero().max() <= self.X_mods[i].shape[0]
                ), "batch index exceeds number of samples"
            except:
                import ipdb ; ipdb.set_trace()
        """

    def prep_batches_mode(self, adata, X_mods, missing_mod, batch_effect, graph):
        modalities = list(adata.keys())
        batch_list = adata[modalities[0]].obs["batch_indices"].values.astype(int)
        batch_index = torch.arange(batch_list.shape[0]).cuda()
        dataset_keys = adata[modalities[0]].obs["dataset"].unique()

        batch_list_mods, batch_list_mods_test = [], []

        for ind, k in enumerate(adata.keys()):
            batch_list_mods.append(
                torch.tensor(adata[k].obs["batch_indices"].values.astype(int)).cuda()
            )
            """
            assert (
                X_mods[ind].shape[0] == batch_list_mods[-1].shape[0]
            ), "batch list does not match features"
            """
            
        batch_list_mods.insert(0, batch_list_mods[0])

        # mask batches based on real/simulated missingness
        all_batches = np.unique(np.array(adata["GEX"].obs["batch_indices"].values))

        batches_missing = {
            i: np.setdiff1d(
                all_batches,
                np.unique(np.array(adata[i].obs["batch_indices"].values)),
            ).tolist()
            for i in modalities
        }
        batches_missing_sim = {i: [] for i in modalities}

        # simulated missingness
        if len(missing_mod) > 0:
            for ind, (mod, batches) in enumerate(missing_mod.items()):
                batches_missing_sim[mod] = (
                    list(
                        adata[mod]
                        .obs["batch_indices"][
                            adata[mod]
                            .obs[batch_effect]
                            .astype(str)
                            .str.contains("|".join(batches))
                        ]
                        .unique()
                    )
                    if len(batches) > 0
                    else batches
                )

        # diictionary of available/missing samples based on configurationi
        cell_avail = {
            "missing": [torch.tensor(v).cuda() for k, v in batches_missing.items()],
            "missing_sim": [
                torch.tensor(v).cuda() for k, v in batches_missing_sim.items()
            ],
        }

        # mask features/batches before training
        mask_mods = [
            (
                ~torch.isin(
                    batch_list_mods[ind + 1],
                    cell_avail["missing_sim"][ind],
                )
            )
            .nonzero()
            .T[0]
            for ind, i in enumerate(X_mods)
        ]

        batch_index_mods = [
            batch_index[
                torch.isin(
                    batch_list_mods[0],
                    torch.tensor(adata[k].obs["batch_indices"].values).cuda(),
                )
                .nonzero()
                .T[0]
                .cuda()
            ]
            for k in adata.keys()
        ]
        batch_index_mods.insert(0, batch_index_mods[0])
        """
        for i in range(len(batch_index_mods)):
            assert (
                batch_index_mods[i].shape[0] == batch_list_mods[i].shape[0]
            ), "Batch indices/lists do not match"
        """
        dataset_idx = [
            adata[modalities[i]].obs["dataset"].values for i in range(len(modalities))
        ]

        return (
            adata,
            dataset_idx,
            batch_list_mods,
            X_mods,
            mask_mods,
            batch_index_mods,
            batch_list_mods,
            cell_avail,
            graph,
        )

    def __len__(self):
        return self.X_mods[0].shape[0]

    def __getitem__(self, batch_index):
        (
            batch_index_mods_minibatch_unmasked,
            batch_index_mods_minibatch,
            batch_index_mods_minibatch_test,
        ) = ([], [], [])
        (
            batch_list_mods_minibatch_unmasked,
            batch_list_mods_minibatch,
            batch_list_mods_minibatch_test,
            batch_index_mods_minibatch_map,
        ) = ([], [], [], [])

        # map global batch indices to omic-specific batches
        for ind in range(len(self.mask_mods)):
            batch_index_chunks = torch.chunk(batch_index, chunks=20)  # 10)
            chunk_arr = []
            # Appending the results of the comparison for each chunk
            for chunk in batch_index_chunks:
                comparison_result = (
                    (self.batch_index_mods[ind + 1] == chunk.unsqueeze(1))
                    .nonzero()
                    .T[1]
                )
                chunk_arr.append(comparison_result)
            batch_index_mods_minibatch_unmasked.append(torch.cat(chunk_arr))

            batch_list_mods_minibatch_unmasked.append(
                self.batch_list_mods[0][batch_index][
                    torch.isin(
                        self.batch_list_mods[0][batch_index],
                        self.batch_list_mods[ind + 1],
                    )
                ]
            )

            # maps minibatch to full
            batch_list_mods_minibatch.append(
                batch_list_mods_minibatch_unmasked[ind][
                    (
                        ~torch.isin(
                            batch_list_mods_minibatch_unmasked[ind],
                            self.cell_avail["missing_sim"][ind],
                        )
                    )
                    .nonzero()
                    .T[0]
                ]
            )

            # maps minibatch to full
            batch_index_mods_minibatch.append(
                batch_index_mods_minibatch_unmasked[ind][
                    (
                        ~torch.isin(
                            batch_list_mods_minibatch_unmasked[ind],
                            self.cell_avail["missing_sim"][ind],
                        )
                    )
                    .nonzero()
                    .T[0]
                ]
            )

        batch_index_mods_minibatch_unmasked.insert(
            0, batch_index_mods_minibatch_unmasked[0]
        )
        batch_index_mods_minibatch.insert(0, batch_index)
        batch_list_mods_minibatch.insert(0, self.batch_list_mods[0][batch_index])
        batch_list_mods_minibatch_unmasked.insert(
            0, self.batch_list_mods[0][batch_index]
        )

        x_minibatch_index_mods = []

        try:
            for ind, i in enumerate(self.X_mods):
                if (
                    batch_index_mods_minibatch[ind + 1].max()
                    > self.X_mods[ind].shape[0]
                ):
                    import ipdb

                    ipdb.set_trace()
                x_minibatch_index_mods.append(
                    (self.X_mods[ind].cuda())[batch_index_mods_minibatch[ind + 1]]
                )
        except:
            import ipdb

            ipdb.set_trace()

        for i in range(len(batch_index_mods_minibatch)):
            try:
                assert (
                    batch_index_mods_minibatch[i].shape[0]
                    == batch_list_mods_minibatch[i].shape[0]
                ), "batch indices and batch list do not match"
                if i == 0:
                    continue
                assert (
                    batch_index_mods_minibatch[i].shape[0]
                    == x_minibatch_index_mods[i - 1].shape[0],
                ), "batch indices and features do not match"
                # assert torch.isin(batch_index_mods_minibatch[i], batch_index_mods_minibatch[0]).min(), "batch indices do not match"
            except:
                import ipdb

                ipdb.set_trace()

        if self.graph[0] is not None:
            graph_batch = [
                dgl.node_subgraph(
                    self.graph[i].to("cuda"), batch_index_mods_minibatch[i + 1]
                )
                for i in range(len(self.graph))
            ]
            """
            for i in range(len(x_minibatch_index_mods)):
                assert (
                    x_minibatch_index_mods[i].shape[0]
                    == graph_batch[i].number_of_nodes()
                )
            """

        else:
            graph_batch = self.graph
        """
        assert(np.array_equal(self.adata['GEX'][batch_index.detach().cpu().numpy()].obs['batch_indices'].values, batch_list_mods_minibatch[0].detach().cpu().numpy()))
        for batch in batch_list_mods_minibatch[0].unique():
            try:
                assert (
                    torch.stack(
                        [
                            (
                                torch.tensor((batch_list_mods_minibatch[i] == batch)
                                .nonzero()
                                .shape[0])
                            )
                            for i in range(1, len(batch_list_mods_minibatch))
                            if batch in batch_list_mods_minibatch[i].unique()
                        ]
                    )
                    .unique()
                    .shape[0]
                    == 1
                ), "batch list does not match across modalities"
            except:
                import ipdb ; ipdb.set_trace()
        """
        return (
            batch_index,
            batch_index_mods_minibatch_unmasked,
            batch_index_mods_minibatch,
            x_minibatch_index_mods,
            batch_list_mods_minibatch_unmasked,
            batch_list_mods_minibatch,
            graph_batch,
        )
