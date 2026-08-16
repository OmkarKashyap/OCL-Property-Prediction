# OCL-Property-Prediction

# To Run
Activate the Conda environment and set the dataset path:  (See [Environment Setup](#environment-setup))
```bash
conda activate object_centric_lib310
export OBJECT_CENTRIC_LIB_DATA=/data/omkar/object-centric-library/datasets
```

Set the model_name in  `checkpoints/oc_eval/train_config.yaml`  (See [Model Configuration](#model-configuration))

Run the downstream property prediction evaluation:
```bash
python eval_downstream_prediction.py downstream_model=linear checkpoint_path=/data/omkar/OCL-Property-Prediction/checkpoints/oc_eval
```

# To Create

## File setup
Create the file structure like:
```
checkpoints/
└── oc_eval/
    ├── evaluation/
    ├── logs/
    └── output/
```

## Model Configuration
In `checkpoints/oc_eval/train_config.yaml`, change the model name to the desired model type. For example:

### 1. DINOv2

In `train_config.yaml`, set `model_name` to `"dinov2"`:

```yaml
model_name: "dinov2"
```

### 2. FT-DINOSAUR

To load and configure the Ft-DINOSAUR module for inference:

1. Clone the repository and rename it:

```bash
git clone https://github.com/rw-ocrl/ftdinosaur-inference.git
rename to ftdinosaur_inference
```

Update the import paths in the respective files:

ftdinosaur_inference/ftdinosaur_inference/build_dinosaur.py: 
```python
from ftdinosaur_inference.ftdinosaur_inference import utils
from ftdinosaur_inference.ftdinosaur_inference.modules import dinosaur
```

ftdinosaur_inference/ftdinosaur_inference/modules/dinosaur.py

```python
from ftdinosaur_inference.ftdinosaur_inference.modules import vit
from ftdinosaur_inference.ftdinosaur_inference.modules.decoding import PatchDecoder
from ftdinosaur_inference.ftdinosaur_inference.modules.helpers import build_mlp, build_two_layer_mlp
from ftdinosaur_inference.ftdinosaur_inference.modules.slot_attention import RandomSlotInitialization,SlotAttentionGrouping
```

In `train_config.yaml`, set `model_name` to `"ft-dinosaur"`:

```yaml
model_name: "ft-dinosaur"
```

### 3. FT-DINOSAUR Patch Avg

To load and configure the Ft-DINOSAUR module for inference:

1. Clone the repository and rename it:

```bash
git clone https://github.com/rw-ocrl/ftdinosaur-inference.git
rename to ftdinosaur_inference
```

Update the import paths in the respective files:

ftdinosaur_inference/ftdinosaur_inference/build_dinosaur.py: 
```python
from ftdinosaur_inference.ftdinosaur_inference import utils
from ftdinosaur_inference.ftdinosaur_inference.modules import dinosaur
```

ftdinosaur_inference/ftdinosaur_inference/modules/dinosaur.py

```python
from ftdinosaur_inference.ftdinosaur_inference.modules import vit
from ftdinosaur_inference.ftdinosaur_inference.modules.decoding import PatchDecoder
from ftdinosaur_inference.ftdinosaur_inference.modules.helpers import build_mlp, build_two_layer_mlp
from ftdinosaur_inference.ftdinosaur_inference.modules.slot_attention import RandomSlotInitialization,SlotAttentionGrouping
```

In `train_config.yaml`, set `model_name` to `"ft-dinosaur-patch-avg"`:

```yaml
model_name: "ft-dinosaur-patch-avg"
```

### 4. DINSOAUR

In `train_config.yaml`, set `model_name` to `"dinosaur"`:

```yaml
model_name: "dinosaur"
```

## Dataset Setup

1. In a folder, add the COCO train2017, val2017 images and the annotations with a structure like:
```
coco/
    train2017/
    val2017/
    annotations/
```
In `config/dataset/coco.yaml`, add the path to the `coco/` directory containing the COCO dataset in `dataset_path`


2. Create the `datasets/` directory. Add the .h5 file for the coco train and coco val as provided.

In `data/datasets.py`, modify `cache_path_train` and `cache_path_val`. Set these paths to the appropriate .h5 file for the COCO train and validation dataset caches.

## Environment Setup
Activate the Conda environment and set the dataset path:
```bash
conda activate object_centric_lib310
export OBJECT_CENTRIC_LIB_DATA=/data/omkar/object-centric-library/datasets
```

## Run
Run the downstream property prediction evaluation:
```bash
python eval_downstream_prediction.py downstream_model=linear checkpoint_path=/data/omkar/OCL-Property-Prediction/checkpoints/oc_eval
```



# Changes done 
## Added

- `train_config.yaml`
- `config/dataset/coco.yaml`
- COCO dataset support in `data/datasets.py`

## Modified

- `config/eval_downstream_prediction.yaml`
- `eval_downstream_prediction.py`
- `data/datasets.py`
- `models/utils.py`
- `evaluation/feature_prediction/models.py`
- `evaluation/feature_prediction/core.py`
- `evaluation/feature_prediction/shared.py`