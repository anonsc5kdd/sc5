import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
import gc

import dgl
import dgl.nn as dglnn


def build_moETM(
    input_dims,
    num_batch,
    num_topic=50,
    emd_dim=400,
    lr=1e-3,
    domains=[],
    flavor="original",
):
    from layers import encoder, decoder, graph_encoder, decoder_patchwork

    decoder_all = decoder(
        mod_dims=input_dims,
        z_dim=num_topic,
        emb_dim=emd_dim,
        num_batch=num_batch,
        domains=domains,
    ).cuda()
    if "graph" in flavor and False:
        encoders = nn.ModuleList(
            [
                graph_encoder(x_dim=input_dim_mod, z_dim=num_topic).cuda()
                for input_dim_mod in input_dims
            ]
        )
    else:
        encoders = nn.ModuleList([])
        for input_dim_mod in input_dims:
            encoders.append(encoder(x_dim=input_dim_mod, z_dim=num_topic).cuda())

    if "original" in flavor:
        decoder_all = decoder(
            mod_dims=input_dims,
            z_dim=num_topic,
            emb_dim=emd_dim,
            num_batch=num_batch,
            domains=domains,
        ).cuda()

    elif "patchwork" in flavor:
        decoder_all = decoder_patchwork(
            mod_dims=input_dims,
            z_dim=num_topic,
            emb_dim=emd_dim,
            num_batch=num_batch,
            domains=domains,
        ).cuda()

    PARA = [
        {"params": decoder_all.parameters()},
    ]
    for encoder in encoders:
        PARA.append({"params": encoder.parameters()})

    optimizer = optim.Adam(PARA, lr=lr)

    return encoders, decoder_all, optimizer
