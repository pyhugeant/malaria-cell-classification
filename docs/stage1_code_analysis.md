# Stage-1 Rebalanced Experiment: Code Analysis and Planned Changes

## Scope
This note analyzes the uploaded implementation files and maps the requested changes to the current codebase.

Requested changes:
1. Keep the ANA branch, but switch ANA loss to weighted BCE with `lambda_ana = 0.1` so that META remains the primary optimization target.
2. Increase crop size to `256` and crop jitter to `32`.
3. Increase model capacity to `base_channels = 128` and `ConvLSTM hidden_channels = 256`.
4. Train with `epochs = 50` and `batch_size = 8`.

## File-by-file analysis

### `src/losses/losses.py`
The current implementation already supports the requested loss formulation.

Relevant functions:
- `bce_with_logits_loss(...)`
- `phase_gated_loss(...)`

Important observations:
- `phase_gated_loss(...)` already supports `ana_loss_type = "bce"`.
- `phase_gated_loss(...)` already supports `lambda_ana`.
- `phase_gated_loss(...)` already supports `ana_bce_pos_weight`.
- META loss is still computed independently from ANA loss, and the final loss is:
  - `total = meta_loss + lambda_ana * ana_loss`

Conclusion:
- No structural code change is required in `losses.py` for the requested experiment.
- The experiment can be enabled through configuration, as long as the training loop passes:
  - `ana_loss_type = "bce"`
  - `ana_bce_pos_weight = <scalar>`
  - `lambda_ana = 0.1`

Recommended starting point:
- `ana_bce_pos_weight = 5.0`

Reason:
- The ANA branch should remain auxiliary.
- A moderate positive-class weight is safer than an aggressive one, because `lambda_ana` is intentionally reduced to prevent ANA from dominating training.

### `src/datasets/event_dataset.py`
The requested crop and jitter settings are already controlled by configuration.

Relevant fields:
- `self.patch = int(ds_cfg["patch_size"])`
- `self.jitter = int(ds_cfg.get("jitter_px", 0)) if train else 0`

Important observations:
- Crop size is controlled by `dataset.patch_size`.
- Crop jitter is controlled by `dataset.jitter_px`.
- No code change is required for:
  - `patch_size = 256`
  - `jitter_px = 32`

Conclusion:
- This request is configuration-only.

### `src/datasets/seq_augment.py`
No code change is required.

Reason:
- Crop size and jitter are handled in `event_dataset.py`, not in sequence augmentation.
- Sequence augmentation only applies flips, `rot90`, and brightness scaling consistently across frames and targets.

### `src/models/model.py`
The requested model scaling is already supported by constructor arguments.

Relevant constructor:
- `MitosisNet(base_channels: int = 64, temporal_type: str = "convlstm", hidden_channels: int = 128)`

Important observations:
- `base_channels` can be increased from `64` to `128` directly.
- `hidden_channels` can be increased from `128` to `256` directly.
- `temporal_type = "convlstm"` is already supported.

Conclusion:
- No structural code change is required.
- The experiment can be enabled through configuration.

### `src/models/convlstm.py`
No code change is required.

Reason:
- `ConvLSTM(in_ch, hid_ch)` already supports arbitrary hidden channel size.
- The requested change is fully covered by `hidden_channels = 256` in model configuration.

### Training configuration
The training script was not included in the uploaded files, so the exact integration point is not visible here.

However, for this experiment, the training configuration must pass the following values into:
- dataset construction
- model construction
- loss construction
- dataloader/training loop

Required configuration values:
- `dataset.patch_size = 256`
- `dataset.jitter_px = 32`
- `model.base_channels = 128`
- `model.temporal_type = "convlstm"`
- `model.hidden_channels = 256`
- `loss.ana_loss_type = "bce"`
- `loss.ana_bce_pos_weight = 5.0`
- `loss.lambda_ana = 0.1`
- `train.epochs = 50`
- `train.batch_size = 8`

## Risk assessment

### Memory risk
This planned experiment is substantially heavier than the current baseline because it changes all of the following at once:
- crop size: `256` instead of a smaller patch
- encoder width: `128` instead of `64`
- ConvLSTM hidden size: `256` instead of `128`
- batch size: `8`

This can significantly increase GPU memory usage.

Recommendation:
- Keep `epochs = 50`.
- Try `batch_size = 8` first only if GPU memory is sufficient.
- If out-of-memory occurs, reduce to `batch_size = 4` and use gradient accumulation to preserve an effective batch size of `8`.

### Optimization risk
The ANA branch is intentionally kept weak in this plan.

Reason:
- The current project goal is META localization first.
- ANA is only an auxiliary signal for downstream timing.
- A small `lambda_ana = 0.1` is aligned with the project goal.

## Implementation summary
For the uploaded code, the requested experiment is primarily a **configuration update**, not a source-code rewrite.

In other words:
- `losses.py`: already compatible
- `event_dataset.py`: already compatible
- `model.py`: already compatible
- `convlstm.py`: already compatible
- the main required change is the experiment configuration file and the branch/commit structure

## Proposed branch name
`exp/stage1-rebalanced-wbce-c256-b8`

## Proposed commit message
`Configure stage-1 rebalanced experiment with weighted BCE, larger crop, and higher-capacity temporal model`
