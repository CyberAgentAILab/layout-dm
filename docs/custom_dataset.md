# Training on custom dataset

Note: Please run all the scripts at the root of this project to make sure to invoke `poetry` command.

### 1. Make a config file for the dataset

Make a yaml file describing the dataset and put it under [this directory](../src/trainer/trainer/config/dataset/).
This yaml is parsed and used as the input for [hydra.utils.instantiate](https://hydra.cc/docs/advanced/instantiate_objects/overview/) to initialize the dataset class.
For example, the config for Rico dataset ([rico25.yaml](../src/trainer/trainer/config/dataset/rico25.yaml)) is currently as follows:

```yaml
_target_: trainer.datasets.rico.Rico25Dataset
_partial_: true
dir: ???
max_seq_length: 25
```

### 2. Implement a dataset class

Implement a dataset class that is defined above. It should inherit `BaseDataset` in [base.py](../src/trainer/trainer/datasets/base.py) and override `preprocess` function to conduct dataset-specific preprocessing and train-val-test split. For example, please refer to `Rico25Dataset` in [rico.py](../src/trainer/trainer/datasets/rico.py).

Modify `DATASET_DIR` in [global_config.py](../src/trainer/trainer/global_configs.py), so that your dataset is used.
`DATASET_DIR` should have the following structure.
```
DATASET_DIR
    - raw
    - processed
```
`DATASET_DIR/raw` contains raw dataset files.
`DATASET_DIR/processed` contains the preprocessed splits and meta information that is auto-generated.

### 3. Train a layout classifier for FID computation

Following [LayoutGAN++](https://arxiv.org/abs/2108.00871), we compute FID during and after the training of LayoutDM for validation and testing, respectively. In order to do so, we first train a Transformer-based model that can extract discriminative layout features, which is used to compute the FID. This is done by:

```bash
poetry run python3 src/trainer/trainer/fid/train.py <DATASET_YAML_PATH> --out_path <FID_WEIGHT_DIR>
```

After the training, modify `FID_WEIGHT_DIR` in [global_config.py](../src/trainer/trainer/global_configs.py), so that the trained weights are used for FID computation later.

### (4. Clustering coordinates of layouts)
If one wants to apply adaptive quantization for position and size tokens, please first conduct clustering.
```bash
poetry run python3 preprocess/clustering_coordinates.py <DATASET_YAML_PATH> <ALGORITHM> --result_dir <KMEANS_WEIGHT_ROOT>
```

After the clustering, modify `KMEANS_WEIGHT_ROOT` in [global_config.py](../src/trainer/trainer/global_configs.py), so that the cluster centroids are loaded later.

### 5. Train your own model
```bash
bash bin/train.sh rico25 layoutdm
```
