## $SC^5$: Single-Cell Cross-Cohort Cross-Category) integration

### Dataset
The raw NeurIPS 2021 AnnData files can be downloaded from [here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)

### Environment set-up
```
conda create --name sc5 --file environment_droplet.yml
```

### Run patchwork integration & eval
```
chmod +x ./scripts/*

CONFIG=neurips2021_combine_missing_site

EXP=patchwork_COMBINE
RUN_MODE=train_seed_1234

# rename experiment name for visualization purposes
VIS_NAME=patchwork

./scripts/run_exps_model_suite.sh $CONFIG $EXP $RUN_MODE
./scripts/run_eval_model_suite.sh $CONFIG $EXP $VIS_NAME
```

### Code organization
- `data/` - Dataset subdirectories containing .h5ad files
- `configs/` - Configuration files for each dataset
- `scripts/` - Script files
- `output/` - Output directory for checkpointing & visualization
- `eval_classes/` - Evaluation util classes
- `exps.py` - Main python script for running integration task
- `dataloader.py` - Dataloading python script
- `models.py` - Python script for running various integration models
- `contrastive.py` - Graph-based contrastive loss
- `eval.py` - Evaluation suite
- `utils.py` - Util functions (including graph creation)

In the folder `moETM`:
- `moETM/build_model.py` - Encoder/decoder modules
- `moETM/train_moetm.py` - Main moETM training file
- `moETM/trainer_moetm.py` - moETM trainer
- `moETM/trainer_moetm_patchwork.py` - moETM patchwork trainer
