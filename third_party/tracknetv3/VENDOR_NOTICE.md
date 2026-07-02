# Vendored: TrackNetV3

This directory contains a shallow clone of the **TrackNetV3** codebase:

- **Upstream:** https://github.com/qaz812345/TrackNetV3 (branch `master`)
- **License:** MIT — see [`LICENSE`](LICENSE) (`Copyright (c) 2024 qaz812345`)
- **Paper:** *TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations
  and Trajectory Rectification* (ACM MMSPA'23)

## Why it is vendored (not pip-installed)

TrackNetV3 has no PyPI package, and — critically — we **fine-tune the upstream
pretrained weights**. The model architecture (`model.py`), the weighted-BCE loss
(`utils/metric.py`) and the heat-map target conventions (`utils/general.py`)
must match the checkpoint **byte-for-byte** for `load_state_dict` to succeed
with zero key mismatch. Vendoring guarantees that compatibility; reimplementing
risked subtle channel/shape mismatches that would make the pretrained weights
unloadable.

Verified: `TrackNet_best.pt` loads into `get_model('TrackNet', seq_len=8,
bg_mode='concat')` with **0 missing / 0 unexpected** keys.

## What we actually use from here

Only the weight-critical pieces, re-exported through `src/tracknet/`:

| File | Used for |
|------|----------|
| `model.py` | `TrackNet` (2D UNet, heatmap prediction) + `InpaintNet` (1D UNet, trajectory rectification) |
| `utils/general.py` | `get_model`, constants `HEIGHT=288 / WIDTH=512 / SIGMA=2.5`, helpers |
| `utils/metric.py` | `WBCELoss` (the TrackNetV2 weighted binary cross-entropy) |

## What we do NOT use (and why it's still here)

`dataset.py`, `train.py`, `test.py`, `predict.py`, `preprocess.py`,
`generate_mask_data.py`, `correct_label.py`, `error_analysis.py` and the
`corrected_test_label/` fixtures are kept for reference and upstream Sync, but
are **not** imported by our pipeline. They assume a specific "rally/match"
directory layout and a hardcoded `data_dir='data'`; we replace them with our own
adapters tailored to our padel ball data:

- `scripts/build_tracknet_dataset.py` — our padel ball data → TrackNet label CSVs
- `src/train_tracknet.py` — fine-tune loop built on the vendored `TrackNet`/`WBCELoss`
- `src/ball_tracker.py` — inference wrapper consumed by `PadelAnalyzer`

## Updating

```bash
cd third_party/tracknetv3 && git pull          # then re-verify weight compat:
python -c "import torch; from src.tracknet import get_model, pretrained_tracknet_path; \
  c=torch.load(pretrained_tracknet_path(),map_location='cpu'); \
  m=get_model('TrackNet',c['param_dict']['seq_len'],c['param_dict']['bg_mode']); \
  print('compat', m.load_state_dict(c['model'],strict=False))"
```

## Pretrained weights

Not stored here (too large for git) — downloaded by the setup step into
`data/models/tracknet_pretrained/ckpts/` (`TrackNet_best.pt`, `InpaintNet_best.pt`)
from the Google Drive link in the upstream README.
