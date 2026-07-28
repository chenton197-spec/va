# robotfm

Extensible robot learning framework with **flow matching**. PushT is the first simulation backend; the data protocol and policy stack are designed for multi-camera observations, variable state/action dimensions, and future real-robot integration.

## Environment

Use the existing conda env:

```bash
conda activate lerobot
cd /home/casbotskill/ct/va
pip install -e .
```

## Quick start (PushT)

### 1. Collect demonstrations (mouse teleop)

```bash
conda activate lerobot
cd /home/casbotskill/ct/va
python scripts/collect.py --config configs/pusht_fm.yaml --target-episodes 10
```

Controls:
- **Mouse**: target end-effector position
- **R**: reset / discard current episode
- **S**: save current episode early
- **Q / Esc**: quit

Data is saved to `data/demos/pusht_demos/episodes/ep_*.npz` with `meta.json` and `stats.json`.

### 2. Train flow matching

Default policy stack: **ImageNet-pretrained ResNet-18** (frame-diff + BN, fine-tuned at 0.1× lr), **ConditionalUnet1D + FiLM**, OT flow-matching. Training uses random 84×84 crops; `horizon=8` matches `n_action_steps`.

```bash
python scripts/train_flow_matching.py --config configs/pusht_fm.yaml
```

SlowFast-R50 video backbone (Kinetics pretrained; `n_obs_steps=8`, resize 256 → crop 224):

```bash
python scripts/train_flow_matching.py --config configs/pusht_slowfast_fm.yaml
```

A2A / N-A2A（Action-to-Action flow matching，依赖 `torchcfm`；历史动作为 flow 起点）：

```bash
pip install torchcfm
python scripts/train_flow_matching.py --config configs/pusht_a2a.yaml
python scripts/train_flow_matching.py --config configs/pusht_n_a2a.yaml
```

Checkpoints go to `outputs/fm_pusht_pretrained/` (see `output_dir` in the yaml).

### 3. Evaluate in simulation

```bash
python scripts/eval_flow_matching.py --config configs/pusht_fm.yaml --save-video
```

Reports success rate, avg reward, and mean max coverage. Eval uses a fixed center crop.
### 4. Visualize dataset

```bash
python scripts/visualize_dataset.py --run-dir data/demos/pusht_demos --episode 0
```

## Data format

Each episode NPZ stores:
- `images/<camera>`: `(T, H, W, 3)` uint8
- `state`: `(T, state_dim)` float32
- `action`: `(T, action_dim)` float32
- `reward`, `done`, `success`, `task`

`meta.json` defines cameras and dimensions for the whole run.

## Extending to real robots

1. Subclass `RealRobotEnv` in `robotfm/envs/real_robot.py`
2. Register backend in `robotfm/envs/registry.py`
3. Implement a teleop driver under `robotfm/collect/drivers/`
4. Add a yaml config with your `cameras`, `state_dim`, `action_dim`

No changes to `FlowMatchingPolicy` are required if dimensions stay consistent within a run.

## Roadmap

- **Phase 1 (current)**: PushT sim + flow matching BC（预训练 ResNet + UNet1D）
- **Phase 2**: Multitask DiT with language (`policies/multitask_dit.py` stub)
- **Phase 3**: RL fine-tuning (`rl/base.py` stub)
