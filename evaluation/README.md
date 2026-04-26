# Evaluation Pipeline

Dataset-agnostic, multi-model evaluation framework for radar point cloud generation.

## Quick Start

```bash
python evaluation/scripts/evaluate.py --config evaluation/configs/truckscenes_eval.yaml
```

## Configuration

All settings live in a YAML config file. See `evaluation/configs/truckscenes_eval.yaml`.

## Metrics

| Metric | Scope | Description |
|--------|-------|-------------|
| Chamfer Distance (xy) | per-sample, per-box | Spatial accuracy |
| Chamfer Distance (normalized) | per-sample, per-box | All attributes normalized to [0,1] |
| IoU | per-sample | At 1.0m threshold |
| MMD (xy, rcs, doppler) | per-sample | Distribution similarity |
| Recall / Precision / F1 | per-sample | Optimal one-to-one matching with distance + attribute thresholds |
| Density Similarity | per-box | min(N, M) / max(N, M) ratio between synthetic and GT points |
| Hit / Miss / FP rates | per-box | Box detection statistics |
| Invariant MMD per class | aggregated | Class-level distribution comparison in box-local coords |

## Adding a New Model

1. Implement `ModelWrapper` (see `evaluation/protocols.py`) and register it with `@register_model("my_model")` in a new file under `evaluation/models/`. See `evaluation/models/radargen_wrapper.py` as an example.
2. Import it in `evaluation/models/__init__.py`.
3. Add it to the config under `models`. See `evaluation/configs/truckscenes_eval.yaml` as an example.
