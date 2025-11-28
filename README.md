# Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark

**Title:** *Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark*  
**Conference:** *IEEE Winter Conference on Applications of Computer Vision (WACV) 2026*  
**Paper:** [arXiv](https://arxiv.org/abs/2502.12003v3.pdf)  
**Dataset:** [Zenodo](https://zenodo.org/records/17584629)  
**Model Weights:** [HuggingFace](https://huggingface.co/saadlahrichi/WSTSPlus)  
**Authors:** Saad Lahrichi, Jake Bova, Jesse Johnson, Jordan Malof

---

This repository extends the original **WildfireSpreadTS** benchmark with new models, improved training, and an expanded benchmark dataset, **WSTS+**.

---
## Benchmark Results (AP ± Standard Deviation)

#### Mean Test AP for T = 1 and T = 5 Across Feature Sets

| **Fusion Level** | **Model** | **Input Days** | **Veg** | **Multi** | **All** | **# Params** |
|------------------|-----------|----------------|---------|-----------|---------|-------------|
| **–** | Res18-UNet *(Gerard et al. 2023)* | 1 | 0.328 ± 0.090 | 0.341 ± 0.085 | 0.341 ± 0.086 | 14.3M |
| | Res18-UNet | 1 | **0.455 ± 0.090** | **0.468 ± 0.087** | **0.460 ± 0.084** | 14.3M |
| | Res50-Unet | 1 | **0.457 ± 0.089** | 0.459 ± 0.090 | 0.451 ± 0.093 | 32.5M |
| | SwinUnet | 1 | 0.432 ± 0.088 | 0.437 ± 0.082 | 0.424 ± 0.090 | 27.2M |
| | SegFormer | 1 | 0.433 ± 0.080 | 0.436 ± 0.083 | 0.423 ± 0.087 | 27.5M |
| **Data** | Res18-UNet *(Gerard et al. 2023)* | 5 | 0.333 ± 0.079 | 0.344 ± 0.076 | 0.325 ± 0.108 | 14.4M |
| | Res18-UNet | 5 | **0.472 ± 0.083** | **0.469 ± 0.087** | **0.460 ± 0.084** | 14.4M |
| | SwinUnet | 5 | 0.447 ± 0.087 | 0.453 ± 0.083 | 0.435 ± 0.079 | 27.3M |
| | SegFormer | 5 | 0.439 ± 0.081 | 0.436 ± 0.085 | 0.430 ± 0.082 | 27.7M |
| **Feature** | UTAE *(Gerard et al. 2023)* | 5 | 0.372 ± 0.088 | 0.350 ± 0.113 | 0.321 ± 0.135 | 1.1M |
| | UTAE | 5 | 0.452 ± 0.082 | 0.459 ± 0.088 | 0.433 ± 0.099 | 1.1M |
| | UTAE (Res18) | 5 | **0.478 ± 0.085** | **0.477 ± 0.089** | **0.475 ± 0.091** | 14.6M |

---
## Datasets
### WSTS+ (Extended Benchmark)

- **Name:** WSTS+
- **Years:** 2016–2018; 2021-2023 
- **Link:** https://doi.org/10.48550/arXiv.2502.12003

### Original WSTS Dataset

- **Name:** WildfireSpreadTS (WSTS)  
- **Years:** 2018–2021  
- **Link:** https://doi.org/10.5281/zenodo.8006177  

Both datasets are compatible with the same preprocessing and training code in this repository.

## Model Weights

We release our best T=1 and T=5 models (Res18-Unet and Res18-UTAE) as PyTorch .pth files containing the raw state_dict. They follow a consistent naming convention: ```fold_<foldID>_testAP<value>.pth``` and they are organized in folders by architecture (Res18UNet, Res18UTAE), temporal dimension (T=1 or T=5), and feature set used (Veg, Multi, or All). 

Each model directory contains 12 files: one per cross-validation fold (fold_0 … fold_11). The filename includes the Test AP, allowing for easy identification of best- and worst-performing folds.
**Link:** [HuggingFace](https://huggingface.co/saadlahrichi/WSTSPlus)

### Loading pretraind Models
We provide a utility script ```load_trained_model.py``` to allow for quickly loading the pretrained models. Example calls:

```
python load_trained_model.py \
    --weights_path /path/to/unet/model/fold_X_testAP0.X.pth \
    --model unet
```

Or for UTAE:
```
python load_trained_model.py \
    --weights_path /path/to/utae/model/fold_Y_testAP0.Y.pth \
    --model utae
```

## Model Comparison Table

| Model | Parameters (M) | FLOPs (G) | Inference Time (ms) | GPU Memory (MB) | Model Size (MB) | Training Time (hours) | Test AP |
|-------|---------------:|----------:|---------------------:|----------------:|-----------------:|-----------------------:|--------:|
| ResNet18-UNet | 14.4 | 1.8 | 2.5±0.0 | 70  | 55  | 0.4 | 0.455 |
| ResNet50-UNet | 32.6 | 3.1 | 5.1±0.1 | 375 | 125 | 1.1 | 0.457 |
| SwinUnet | 27.2 | 6.1 | 8.9±0.0 | 526 | 106 | 1.8 | 0.432 |
| SegFormer-B2 | 27.5 | 3.7 | 12.7±0.8 | 865 | 105 | 2.0 | 0.448 |
| UTAE | 1.1 | 10.6 | 9.5±1.0 | 997 | 4 | 1.0 | 0.452 |

---

## WSTS vs. WSTS+ Dataset Comparison

| **Dataset**       | **WSTS**           | **WSTS+**          | **Increase (%)** |
|-------------------|--------------------|---------------------|------------------|
| Years             | 4 (2018–2021)      | 8 (2016–2023)       | +100%            |
| Fire Events       | 607                | 1,005               | +65.6%           |
| Total Images      | 13,607             | 24,462              | +79.8%           |
| Active Fire Pixels| 1,878,679          | 2,638,537           | +40.4%           |

---
### Citation
If you use this fork or the WSTS+ benchmark, please consider citing:

```
@inproceedings{
    lahrichi2026improved,
    title={Improved Wildfire Spread Prediction with Time-Series Data and the WSTS+ Benchmark},
    author={Saad Lahrichi, Jake Bova, Jesse Johnson, Jordan Malof},
    booktitle={IEEE Winter Conference on Applications of Computer Vision (WACV) 2026},
    year={2026},
    url={https://arxiv.org/abs/2502.12003}
}
```

# Highlights from Original README 

## Setup the environment

``` pip3 install -r requirements.txt ```

## Preparing the dataset

The dataset is freely available at [https://doi.org/10.5281/zenodo.8006177](https://doi.org/10.5281/zenodo.8006177) under CC-BY-4.0. For training, you will need to convert them to HDF5 files, which take up twice as much space but allow for much faster training.

To convert the dataset to HDF5, run:
```python3 src/preprocess/CreateHDF5Dataset.py --data_dir YOUR_DATA_DIR --target_dir YOUR_TARGET_DIR```
 substituting the path to your local dataset and where you want the HDF5 version of the dataset to be created. 

If you want to skip this step, and simply pass `--data.load_from_hdf5=False` on the command line, but be aware that you won't be able to perform training at any reasonable speed. 

## Re-running the baseline experiments

We use wandb to log experimental results. This can be turned off by setting the environment variable `WANDB_MODE=disabled`. The results will then be logged to a local directory instead.

Experiments are parameterized via yaml files in the `cfgs` directory. Arguments are parsed via the [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html).

Grid searches and repetitions of experiments were done via WandB sweeps. Those are parameterized via yaml files in the `cfgs` directory prefixed with `wandb_`. WandB configuration files ending in `_repetition` contain the configurations for the runs that were used in the paper. They only needed to be _repeated_ five times with varying seeds to estimate the variance of results. We omit explanations of how to use wandb sweeps to run experiments and refer the readers to the [original documentation](https://docs.wandb.ai/guides/sweeps). To run the same experiments without WandB, the parameters specified in the WandB sweep configuration file can simply be passed via the command line. 

For example, to train the U-net architecture on one day of observations, which is specified in `cfgs/unet/wandb_monotemporal_repetition.yaml`, we could simply copy and paste the WandB parameters to the command line:

```
python3 train.py --config=cfgs/unet/res18_monotemporal.yaml --trainer=cfgs/trainer_single_gpu.yaml --data=cfgs/data_monotemporal_full_features.yaml --data.data_dir YOUR_DATA_DIR
```
where you replace `YOUR_DATA_DIR` with the path to your local HDF5 dataset. Alternatively, you can permanently set the data directory in the respective data config files. Later arguments overwrite previously given arguments, including parameters defined in config files. 
## Citation

```
@inproceedings{
    gerard2023wildfirespreadts,
    title={WildfireSpread{TS}: A dataset of multi-modal time series for wildfire spread prediction},
    author={Sebastian Gerard and Yu Zhao and Josephine Sullivan},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2023},
    url={https://openreview.net/forum?id=RgdGkPRQ03}
}
