# FGTE

This repository contains the implementation of the following paper:
Fine-Grained Texture Enhancement via Dynamic State Space Fusion for Reference-based Image Super-Resolution

## Overview


## Dependencies and Installation

- Python >= 3.8
- PyTorch >= 2.2.1
- CUDA 12.1

## Dataset Preparation

- Train Set: [CUFED Dataset](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I)

Please refer to [Datasets.md](datasets/DATASETS.md) for pre-processing and more details.

## Get Started

### Test

We provide quick test code with the pretrained model.

1. Modify the paths to dataset and pretrained model in the following yaml files for configuration.

    ```bash
    ./options/test/test_gan.yml
    ./options/test/test_mse.yml
    ```

1. Run test code for models trained using **GAN loss**.

    ```bash
    python mmsr/test.py -opt "options/test/test_gan.yml"
    ```

   Check out the results in `./results`.

1. Run test code for models trained using only **reconstruction loss**.

    ```bash
    python mmsr/test.py -opt "options/test/test_mse.yml"
    ```
    
   Check out the results in `./results`


### Train

All logging files in the training process, *e.g.*, log message, checkpoints, and snapshots, will be saved to `./experiments` and `./tb_logger` directory.


1.  Train restoration network.
   ```bash
   # add the path to *pretrain_model_feature_extractor* in the following yaml
   #  prepare the dataset path and pretrained model path in the (./options/train/stage3_restoration_gan.yml). The path to *pretrain_model_feature_extractor* is the model obtained in link 
   python mmsr/train.py -opt "options/train/stage3_restoration_gan.yml"

   # if you wish to train the restoration network with only mse loss
   python mmsr/train.py -opt "options/train/stage3_restoration_mse.yml"
   ```



