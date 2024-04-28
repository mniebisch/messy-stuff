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

Create random dummy data which matches fingerspelling5 + mediapipe hand landmark dataset properties.
```
python scripts/create_fingerspelling5_dummy.py \
    --dir-dest=data/fingerspelling5 \
    --dataset-name=fingerspelling5_dummy \
    --num-persons=3 \
    --num-samples=4
```

Create datasplits for fingerspelling5 mediapipe data.
Grouped N-fold split using fingerspelling5 persons as group.
```
python scripts/create_fingerspelling5_splits.py \
    --dataset-dir=data/fingerspelling5/fingerspelling5_dummy
```

Compute descriptive statistics for dataset.
```
python scripts/compute_fingerspelling5_metrics.py predict \
    --config configs/examples/fingerspelling5_dummy_metrics.yaml
```

Train model using dummy data.
```
python scripts/train_fingerspelling5_litcli.py fit \
     --config configs/examples/fingerspelling5_dummy_training.yaml
```

Make prediction using trained model.
```
python scripts/train_fingerspelling5_litcli.py predict \
    --config logs/examples/version_0/config.yaml \
    --config configs/examples/fingerspelling5_dummy_prediction.yaml \
    --ckpt_path logs/examples/version_0/checkpoints/epoch=8-step=9.ckpt
```