# LayoutDM: Discrete Diffusion Model for Controllable Layout Generation (CVPR2023)
This repository is an official implementation of the paper titled above.
Please refer to [project page](https://cyberagentailab.github.io/layout-dm/) or [paper](https://arxiv.org/abs/2303.08137) for more details.

## Setup
Here we describe the setup required for the model training and evaluation.

### Requirements
We check the reproducibility under this environment.
- Python3.7
- CUDA 11.3
- [PyTorch](https://pytorch.org/get-started/locally/) 1.12

We recommend using Poetry (all settings and dependencies in [pyproject.toml](pyproject.toml)).
Pytorch-geometry provides independent pre-build wheel for a *combination* of PyTorch and CUDA version (see [PyG:Installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
) for details). If your environment does not match the one above, please update the dependencies.


### How to install
1. Install poetry (see [official docs](https://python-poetry.org/docs/)). We recommend to make a virtualenv and install poetry inside it.

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies (it may be slow..)

```bash
poetry install
```

3. Download resources and unzip

``` bash
wget https://github.com/CyberAgentAILab/layout-dm/releases/download/v1.0.0/layoutdm_starter.zip
unzip layoutdm_starter.zip
```

The data is decompressed to the following structure:
```
download
- clustering_weights
- datasets
- fid_weights
- pretrained_weights
```

## Experiment
**Important**: we find some critical errors that cannot be fixed quickly in using multiple GPUs. Please set `CUDA_VISIBLE_DEVICES=<GPU_ID>` to force the model use a single GPU.

Note: our main framework is based on [hydra](https://hydra.cc/). It is convenient to handle dozens of arguments hierarchically but may require some additional efforts if one is new to hydra.

### Demo
Please run a jupyter notebook in [notebooks/demo.ipynb](notebooks/demo.ipynb). You can get and render the results of six layout generation tasks on two datasets (Rico and PubLayNet).

### Training
You can also train your own model from scratch, for example by

```bash
bash bin/train.sh rico25 layoutdm
```

, where the first and second argument specifies the dataset ([choices](src/trainer/trainer/config/dataset)) and the type of experiment ([choices](src/trainer/trainer/config/experiment)), respectively.
Note that for training/testing, style of the arguments is `key=value` because we use hydra, unlike popular `--key value` (e.g., [argparse](https://docs.python.org/3/library/argparse.html)).

### Testing

```bash
poetry run python3 -m src.trainer.trainer.test \
    cond=<COND> \
    job_dir=<JOB_DIR> \
    result_dir=<RESULT_DIR> \
    <ADDITIONAL_ARGS>
```
`<COND>` can be: (unconditional, c, cwh, partial, refinement, relation)

For example, if you want to test the provided LayoutDM model on `C->S+P`, the command is as follows:
```
poetry run python3 -m src.trainer.trainer.test cond=c dataset_dir=./download/datasets job_dir=./download/pretrained/layoutdm_rico result_dir=tmp/dummy_results
```

Please refer to [TestConfig](src/trainer/trainer/hydra_configs.py#L12) for more options available.
Below are some popular options for <ADDITIONAL_ARGS>
- `is_validation=true`: used to evaluate the generation performance on validation set instead of test set. This must be used when tuning the hyper-parameters.
- `sampling=top_p top_p=<TOP_P>`: use top-p sampling with p=<TOP_P>　instead of default sampling.

### Evaluation
```bash
poetry run python3 eval.py <RESULT_DIR>
```

## Citation

If you find this code useful for your research, please cite our paper:

```
@inproceedings{inoue2023layout,
  title={{LayoutDM: Discrete Diffusion Model for Controllable Layout Generation}},
  author={Naoto Inoue and Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
  pages={10167-10176},
}
```
