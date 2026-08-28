# Does Finetuning on Wrist-Rich Data Help Octo Use Its Wrist Camera?

This repo is a research experiment built on top of [Octo-PyTorch](https://github.com/emb-ai/octo-pytorch), a PyTorch reimplementation of [Octo](https://octo-models.github.io/), a generalist vision-language-action robot policy.

**The question:** Octo's paper reports that combining a wrist camera with the third-person camera often hurt performance *during finetuning*, attributed to only 27% of pretraining data including a wrist camera. We asked a narrower, directly testable version of that: does finetuning the already-pretrained model specifically on wrist-camera-rich data help it use that camera better on one task?

**The answer, from a paired A/B/C comparison on `berkeley_fanuc_manipulation`:**

| Condition | Overall MSE |
|---|---|
| Pretrained, primary camera only | 2.181 |
| Pretrained, primary + real wrist camera | **1.192** |
| Finetuned (3,000 steps), primary + real wrist camera | 1.245 |

The pretrained model already makes strong zero-shot use of the wrist camera (~45% lower error than without it). Finetuning specifically on wrist-rich data did **not** clearly improve on that — a mixed, slightly-worse-overall result, driven mostly by one action dimension (pitch) regressing while others improved.

Full methodology, per-dimension results, and honest caveats about what this does and doesn't support: [WRIST_CAMERA_EXPERIMENT.md](WRIST_CAMERA_EXPERIMENT.md) ([LaTeX version](WRIST_CAMERA_EXPERIMENT.tex)).

## What's in this repo

- [`WRIST_CAMERA_EXPERIMENT.md`](WRIST_CAMERA_EXPERIMENT.md) — the experiment writeup.
- [`eval_abc.py`](eval_abc.py) — the paired A/B/C evaluation script (held-out split, repeated diffusion sampling, normalized-space MSE).
- [`scripts/configs/fanuc_config.py`](scripts/configs/fanuc_config.py) — finetuning config for `berkeley_fanuc_manipulation`.
- A fix to [`berkeley_fanuc_dataset_transform`](octo/data/oxe/oxe_standardization_transforms.py) — this dataset has no raw `language_instruction` field, and the transform was missing the placeholder every other language-less dataset transform in that file already injects.
- Dependency pins in `requirements.txt` for `transformers`, `wandb`, `tensorflow-metadata`, `protobuf`, and `accelerate` — the upstream repo's unbounded version constraints break on a fresh install; these are pinned to a verified-working set.

The underlying model/training code is otherwise the upstream Octo-PyTorch codebase, kept intact so the experiment is reproducible against the real finetuning pipeline rather than a simplified reimplementation.

## Reproducing this

```bash
conda create -n octo_pt python=3.10
conda activate octo_pt
pip install -e .
pip install -r requirements.txt
pip install torch torchvision torchaudio
```

Note: `tensorflow==2.15.0` only ships wheels for Python 3.9–3.11, so Python must be ≤3.11.

Finetune on `berkeley_fanuc_manipulation` (needs a CUDA GPU — the script wraps the model in `DistributedDataParallel` unconditionally, so it must be launched via `torchrun` even for one GPU):

```bash
torchrun --nproc_per_node 1 scripts/finetune_pt.py \
  --config=scripts/configs/fanuc_config.py:full,image_conditioned \
  --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 \
  --config.save_dir=<your_save_dir>
```

Then run the A/B/C evaluation:

```bash
python eval_abc.py \
  --pretrained-path hf://rail-berkeley/octo-small-1.5 \
  --finetuned-checkpoint <path_to_saved_checkpoint> \
  --config-module configs.fanuc_config \
  --config-string full,image_conditioned
```

## Credit

This builds on [emb-ai/octo-pytorch](https://github.com/emb-ai/octo-pytorch), which reimplements [Octo](https://octo-models.github.io/) (Ghosh et al., 2024) in PyTorch. See that repo for the general-purpose model API, inference examples, and architecture customization guides.
