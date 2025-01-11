# fmp

## Fingerspelling5 Training (Lit)
```
python scripts/train_fingerspelling5_litcli.py fit --config configs/train_fingerspelling5.yaml
```

### Predictions (Lit)

Simply use training data for prediction:
```
python scripts/train_fingerspelling5_litcli.py predict \
    --return_predictions true \
    --ckpt_path lightning_logs/version_22/checkpoints/epoch\=17-step\=36.ckpt \
    --config lightning_logs/version_22/config.yaml
```

Overwrite datamodule to use other input (and practice overwriting :P):
```
python scripts/train_fingerspelling5_litcli.py predict \
    --config lightning_logs/version_22/config.yaml \
    --config configs/predict_fingerspelling5.yaml \
    --ckpt_path lightning_logs/version_22/checkpoints/epoch=17-step=36.ckpt
```

### Compute Fingerspelling5 Metrics

```
python scripts/compute_fingerspelling5_metrics.py predict \
    --config configs/metric_computation.yaml
```

## Synthetic Data (Or how to process a dataset)

### Sequence of Manual Steps

#### Create random dummy data which matches fingerspelling5 + mediapipe hand landmark dataset properties.
```
python scripts/create_fingerspelling5_dummy.py \
    --dir-dest=data/fingerspelling5 \
    --dataset-name=fingerspelling5_dummy \
    --num-persons=3 \
    --num-samples=4
```

#### Create datasplits for fingerspelling5 mediapipe data.
Grouped N-fold split using fingerspelling5 persons as group.
```
python scripts/create_fingerspelling5_splits.py \
    --dataset-dir=data/fingerspelling5/fingerspelling5_dummy
```


#### Create dataquality file for dummy data.
For actual fingerspelling5 dataset manual labeling can be performed using the following [script](apps/visualize_videos.py).

Create fake label to test or play with pipeline:
```
python scripts/create_fingerspelling5_dummy_dataquality.py \
    --dataset-dir=data/fingerspelling/fingerspelling_dummy 
```

#### Compute descriptive statistics for dataset.
##### Normalized Landmarks
Compute metrics [scaled](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.NormalizeScale.html#torch_geometric.transforms.NormalizeScale) data.
```
python scripts/compute_fingerspelling5_metrics.py predict \
    --config configs/examples/fingerspelling5_dummy_metrics_scaled.yaml
```

##### Un-normalized Landmarks
Compute metrics for 'raw' recorded data.
```
python scripts/compute_fingerspelling5_metrics.py predict \
    --config configs/examples/fingerspelling5_dummy_metrics_unscaled.yaml
```

#### Train model using dummy data.
```
python scripts/train_fingerspelling5_litcli.py fit \
     --config configs/examples/fingerspelling5_dummy_training.yaml
```

#### Make prediction using trained model.
(Currently I'm not happy with the way how prediction is called(input args etc; (previous) train config + (current) pred config + model ckpt)) (too much redundancy?)
```
python scripts/train_fingerspelling5_litcli.py predict \
    --config logs/examples/version_0/config.yaml \
    --config configs/examples/fingerspelling5_dummy_prediction.yaml \
    --ckpt_path logs/examples/version_0/checkpoints/epoch=8-step=9.ckpt
```

### Pipelines

#### Create dataset with all related files (dataquality, splits)
```
python pipelines/fingerspelling5/create_dummy_data.py
```

#### Compute metrics (scaled and unscaled)
```
python pipelines/fingerspelling5/compute_metrics.py \
     --dataset-path data/fingerspelling5_dummy/ \
     --output-path metrics/
```

#### Train Model and Predict
```
python pipelies/fingerspelling5/train_eval.py \
    --train-config configs/examples/fingerspelling5_dummy_training.yaml \
    --predict-config configs/examples/fingerspelling5_dummy_prediction.yaml
```

# Links
## Mediapipe
- [Visualization of landmarks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Remarks on landmarks](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)

## Docker

### General
- [Cheat Sheet](https://docs.docker.com/get-started/docker_cheatsheet.pdf)

### VSCode
- [VSCode Settings](https://code.visualstudio.com/docs/getstarted/settings)
- [VSCode devcontainer](https://code.visualstudio.com/docs/devcontainers/containers)
- [Tutorial devcontainer](https://code.visualstudio.com/docs/devcontainers/tutorial)
- [Advanced devcontainer](https://code.visualstudio.com/remote/advancedcontainers/overview)
- [Sharing Git credentials](https://code.visualstudio.com/remote/advancedcontainers/sharing-git-credentials)
- [devcontainer.json](https://code.visualstudio.com/docs/devcontainers/create-dev-container)
- [Devcontainer reference](https://containers.dev/implementors/json_reference/)
- Devcontainer Features
    - [Repo](https://github.com/devcontainers/features/tree/main/src)
    - [VSCode mention](https://code.visualstudio.com/docs/devcontainers/containers#_dev-container-features)
