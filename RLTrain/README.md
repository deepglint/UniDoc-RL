# RLTrain

This directory contains training utilities and configuration files for RL-based multimodal retrieval-augmented generation experiments.

## Overview

The RL training stage assumes that:

- an SFT model checkpoint is already available,
- the retrieval service is running,
- the answer evaluation service is running,
- the training and validation data have been prepared in parquet format.

## Before running

Update the placeholders in the config files or override them from the command line:

- model paths such as `/path/to/your/policy-model`
- dataset paths such as `/path/to/your/train.parquet`
- service endpoints for retrieval and reward evaluation

## Example environment variables

- `MODEL_PATH`: policy or SFT checkpoint used by `train.sh`
- `TRAIN_FILE`: training parquet file
- `VAL_FILE`: validation parquet file
- `SEARCH_URL`: retrieval service endpoint
- `RM_URL`: reward-model evaluation endpoint
- `CUDA_VISIBLE_DEVICES`: GPUs used for training

## How to run

### 1. Start the required tool services

Before launching RL training, make sure the following services are available:

- retrieval service from [tools/search_engine](../tools/search_engine)
- evaluation service from [tools/model_eval](../tools/model_eval)

By default, the example launcher expects endpoints such as:

- `SEARCH_URL=http://127.0.0.1:9001/search`
- `RM_URL=http://127.0.0.1:8003/eval`

### 2. Prepare the training environment

Install the required dependencies for this module:

```bash
pip install -r requirements.txt
```

### 3. Set the runtime variables

An example setup is shown below:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_PATH=/path/to/your/sft-checkpoint
export TRAIN_FILE=/path/to/your/train.parquet
export VAL_FILE=/path/to/your/val.parquet
export SEARCH_URL=http://127.0.0.1:9001/search
export RM_URL=http://127.0.0.1:8003/eval
```

### 4. Launch RL training

Run the provided launcher script:

```bash
bash train.sh
```