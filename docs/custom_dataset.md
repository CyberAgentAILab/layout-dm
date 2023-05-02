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
poetry run python3 bin/clustering_coordinates.py <DATASET_YAML_PATH> <ALGORITHM> --result_dir <KMEANS_WEIGHT_ROOT>
```

After the clustering, modify `KMEANS_WEIGHT_ROOT` in [global_config.py](../src/trainer/trainer/global_configs.py), so that the cluster centroids are loaded later.

### 5. Train your own model
```bash
bash bin/train.sh rico25 layoutdm
```

# Testing on custom dataset
If you want to feed a hand-made layout to LayoutDM, the quickest way is to instantiate [Data](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data).

```python
from torch_geometric.data import Data

# [xc, yc, w, h] format in 0~1 normalized coordinates
bboxes = torch.FloatTensor([
    [0.4985, 0.0968, 0.4990, 0.0153],
    [0.4986, 0.5134, 0.8288, 0.0285],
    [0.4986, 0.2918, 0.8289, 0.3573],
])
# see .labels of each dataset class for name-index correspondense
labels = torch.LongTensor([0, 0, 3])
assert bboxes.size(0) == labels.size(0) and bboxes.size(1) == 4

# set some optional attributes by a dummy value (False)
attr = {k: torch.full((1,), fill_value=False) for k in ["filtered", "has_canvas_element", "NoiseAdded"]}

data = Data(x=bboxes, y=labels, attr=attr)  # can be used as an alternative for `dataset[target_index]` in demo.ipynb
```
