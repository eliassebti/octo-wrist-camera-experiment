# Does the Wrist Camera Help Octo-Small, and Does Finetuning on Wrist-Rich Data Help It Help More?

**Date:** 2026-08-27
**Model:** Octo-Small (`hf://rail-berkeley/octo-small-1.5`), PyTorch port
**Dataset:** `berkeley_fanuc_manipulation` (Open X-Embodiment), 415 episodes, 224×224 real wrist camera
**Compute:** RunPod A100 SXM 80GB

## Motivation

Octo's own paper (arXiv 2405.12213) states that combining a wrist camera with the
third-person camera *during finetuning* often performed worse than using the
third-person camera alone, and attributes this to only 27% of pretraining data
including a wrist camera. The paper does not test this claim about
force/contact-blind manipulation specifically — that hypothesis (that Octo's total
lack of a force/torque input is a real limitation for contact-rich tasks) is ours,
derived from reading the architecture, not from the paper. We shelved the
force-input question as a larger, separate project and instead ran a cheaper,
directly testable question: **does finetuning the pretrained model specifically on
wrist-camera-rich data help it use that camera better on one task?**

This is a narrower question than the paper's own claim (which was about
pretraining-time data exposure, not finetuning). We test it because it's
tractable without modifying the architecture.

## Method

**Three conditions**, same held-out data, same evaluation code, run in the same
process for a true paired comparison:

- **A** — pretrained Octo-Small, `image_primary` only. `image_wrist` is zeroed and
  its `pad_mask` set to `False`, matching exactly how the data pipeline represents
  a genuinely wrist-less camera (verified against `bridge_dataset`, which has no
  wrist camera at all, before this experiment — see Appendix).
- **B** — pretrained Octo-Small, `image_primary` + real `image_wrist`, untouched.
- **C** — Octo-Small after finetuning 3,000 steps on `berkeley_fanuc_manipulation`
  (default hyperparameters: batch size 128, cosine LR schedule, peak 3e-4), then
  evaluated with `image_primary` + real `image_wrist`.

**Held-out set:** `berkeley_fanuc_manipulation` has no separate validation split,
so the pipeline's existing 95/5 fallback (`train[:95%]` / `train[95%:]`) was used.
40 held-out examples, none seen during finetuning.

**Metric (decided before seeing results):** mean squared error between the
model's raw action-head output and ground-truth actions, in the dataset's own
normalized space. Comparing in normalized space (rather than unnormalizing)
avoids needing `berkeley_fanuc`'s statistics to exist in the *pretrained* model's
bundled dataset statistics — they don't, since this dataset wasn't part of
Octo's original pretraining mix.

**Handling stochasticity:** Octo's action head is a diffusion model — a single
sample proves nothing. Each of the 40 held-out examples was sampled 8 times per
condition, with a different random seed each time. Combined with the 4-step
action horizon, each condition's result is aggregated over 1,280 error values.

## Results

| Condition | Overall MSE | x | y | z | roll | pitch | yaw | gripper |
|---|---|---|---|---|---|---|---|---|
| A — pretrained, primary only | 2.181 | 2.795 | 3.338 | 2.011 | 1.620 | 3.626 | 1.604 | 0.274 |
| B — pretrained, + real wrist | 1.192 | 1.527 | 1.201 | 1.071 | 1.416 | 1.030 | 2.039 | 0.062 |
| C — finetuned, + real wrist | 1.245 | 1.669 | 0.910 | 0.888 | 1.226 | 2.855 | 1.096 | 0.073 |

## Findings

**1. Adding the real wrist camera to the untouched pretrained model cut error by
~45% (A → B).** This is a large, clean effect in this specific test. It is *not*
a contradiction of the paper's claim — the paper's finding was about combining
cameras hurting during *finetuning*; this test compares zero-shot use of an
already-present camera, a different question.

**2. Finetuning specifically on wrist-rich data did not improve on the
already-strong zero-shot wrist result — it was very slightly worse overall
(B → C: 1.192 → 1.245).** This is the direct answer to the question we set out
to test, and the honest answer from this run is **no**. The per-dimension
breakdown shows this isn't a uniform failure: y, z, and yaw all improved
substantially under finetuning, but pitch got much worse (1.030 → 2.855) and
pulled the aggregate down. Finetuning changed *what* the model got wrong, not
simply whether it got things wrong.

## What this does and does not support

- It supports: on this one task, with this one 3,000-step finetuning run, the
  pretrained model already makes good zero-shot use of the wrist camera, and
  additional finetuning did not clearly help by this metric.
- It does **not** support a general claim that "finetuning on wrist-rich data
  never helps Octo use the wrist camera." This was one finetuning run at default
  hyperparameters. The diffusion-head *sampling* stochasticity was properly
  handled (8 samples/example, decided in advance), but a second finetuning run
  with a different seed or a different hyperparameter (learning rate, step
  count) could plausibly land differently — especially given how volatile the
  pitch dimension was between B and C.
- Normalized-space MSE is a reasonable proxy but is not the same as real robot
  task success rate. A lower-MSE model could still fail the task, or vice versa.
- This result is specific to `berkeley_fanuc_manipulation`. It says nothing
  directly about Bridge, about contact-rich insertion tasks, or about the
  separate force/torque-input question that was shelved.

## Suggested next step

Before treating the B-vs-C result as settled, rerun condition C with a different
random seed (and optionally a different step count) to see whether the pitch
regression is a real property of finetuning on this data or an artifact of one
run's initialization/data order.

## Appendix: reproducibility notes

- Evaluation script: `eval_abc.py` (repo root on the RunPod pod at time of run;
  not yet committed — the finetuning config `scripts/configs/fanuc_config.py` and
  a real bug fix to `berkeley_fanuc_dataset_transform` (missing
  `language_instruction` placeholder, needed because this dataset has no raw
  language field) are committed on branch `fix/dependency-pins` at
  `eliassebti/octo-pytorch`).
- Environment: Python 3.10, `transformers==4.44.2`, `tensorflow==2.15.0`,
  `tensorflow-metadata==1.14.0`, `protobuf==3.20.3`, `wandb==0.16.6`,
  `accelerate==0.33.0` — all pinned in `requirements.txt` on the same branch
  after several rounds of version-drift debugging.
- Finetuning invoked via `torchrun --nproc_per_node 1` (required — the script
  wraps the model in `DistributedDataParallel` unconditionally, which needs a
  real process group even for one GPU).
- Checkpoint: `/workspace/runs/fanuc_wrist/octo_finetune/fanuc_wrist_20260827_183157/`
  on the RunPod pod (not persisted anywhere else as of this writeup).
