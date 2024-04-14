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