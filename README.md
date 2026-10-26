# Deadline-aware wireless channel selection

An original packet-level simulator and Double DQN project for the RL mini-project assignment. A transmitter must deliver packets before their deadlines using eight channels, with limited ACK feedback, costly retuning, and changing interference. All experiments run in software; radio hardware is unnecessary.

Team: **Ayush Singh (B23EE1008)** and **Tashir Ahmad (B23ME1074)**. Repository: [DeadlineHop](https://github.com/Ayush-Singh-as/DeadlineHop).

**Complete core implementation:** simulator, six baselines, Double DQN, 34 passing tests, checkpoint continuation, frozen scenarios, evaluation, figures, demo, and two-page report plus references. Both history variants were trained with three seeds each and 250,000 steps per seed (1.5 million total steps); 4,800 held-out episodes were evaluated. A fresh wheel installation exactly reproduced five recorded episodes.

Measured structured-scenario delivery: **H=16: 90.80%**, **H=1: 94.06%**, **history predictor: 94.56%**. The strongest baseline wins overall; H=1 exceeds it in the shorter-deadline stress condition. These results do not establish that longer history is necessary. See [full results and interpretation](RESULTS.md), [the report](report/report.pdf), and [raw aggregate values](artifacts/figures/overall.json).

Training ran on the exposed Intel Core i7-13650HX CPU with one PyTorch thread per process. Individual runs took 162–309 seconds including validation; the sum of process durations was 1,417.7 seconds, with some runs overlapping. The RTX 4060 was not accessible to this execution environment and was not used for these measurements.

## Installation

Use Python 3.12 (tested: 3.12.14). Create and activate a virtual environment in this directory. Examples below use Linux/macOS paths; on Windows use `.venv\Scripts\python.exe`.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip install -e '.[dev,report]'
.venv/bin/python -m pytest -q
```

The CPU lock records the exact package set tested here. Install the CPU torch wheel first so its `+cpu` version is already satisfied when resolving the lock. If using `uv`, `uv pip install --python .venv/bin/python ...` can replace the pip commands. In this workspace, `.venv` is already installed and usable.

For the RTX 4060, install the CUDA-enabled PyTorch build selected for your driver by the [official installation selector](https://pytorch.org/get-started/locally/), then install `-e '.[dev,report]'` without applying the CPU lock. Record the resulting versions. Confirm `torch.cuda.is_available()` and a real tensor calculation before starting a long run. `--device cuda` fails clearly if unavailable; `--device auto` selects CUDA if exposed, otherwise CPU. This execution environment exposes only CPU, so its released results use CPU. Small networks can train efficiently on CPU.

## Quick evaluation and demo

Run from the project root. Evaluation never retrains and refuses to overwrite an existing result file.

```bash
.venv/bin/python -m radio_rl.evaluate --checkpoint artifacts/checkpoints/h16_seed11.pt --suite artifacts/scenarios/test.json --limit 5 --output quick_check.csv
.venv/bin/python -m radio_rl.demo --checkpoint artifacts/checkpoints/h16_seed11.pt --output demo_check
```

The short evaluation checks functionality; it is not the complete statistical experiment. The demo uses the first switching validation scenario, chosen independently of test scores. Its red interference background is evaluator-only information unavailable to the policy.

## Reproduce the complete experiment

Released checkpoints, scenario manifests, baseline settings, and compact training logs are sufficient to regenerate comparisons without training. The following runs 400 episodes for each of six learned checkpoints and six baselines, then rebuilds figures:

```bash
.venv/bin/python scripts/reproduce.py --output reproduced
```

An empty output directory is required. Compare its episode counts and metrics against `artifacts/results`; same-platform deterministic evaluation should agree. No internet or external dataset is needed after installing dependencies. Training-seed differences are shown as individual dots/ranges, not presented as hundreds of independent training repetitions.

## Train from scratch

The checked-in scenario manifests are already frozen. Do not regenerate them over the originals. To explore a separate experiment, generate into a new directory and keep it distinct from the reported experiment.

```bash
.venv/bin/python -m radio_rl.train --config configs/history_ddqn.yaml --seed 11 --device auto --output runs/my_history_seed11
.venv/bin/python -m radio_rl.train --config configs/single_observation_ddqn.yaml --seed 11 --device auto --output runs/my_single_seed11
.venv/bin/python scripts/campaign.py --device auto
```

The campaign runs the predefined three-seed matrix and copies selected checkpoints plus learning logs to `artifacts`. It skips runs only when their recorded step budget matches. It resumes interrupted runs when a complete resume file exists. To continue one run explicitly:

```bash
.venv/bin/python -m radio_rl.train --config configs/history_ddqn.yaml --seed 11 --device cpu --output runs/my_history_seed11 --resume runs/my_history_seed11/resume.pt
```

Only resume files produced by this project and trusted by you should be loaded: full resume uses serialized replay/environment objects. Released inference models use `weights_only=True`. Exact continuation requires matching code, seed, configuration, and validation suite; it restores optimizer/target/replay/environment/history and random states. A previous atomic resume checkpoint is retained. Cross-device/library-version bitwise equivalence is not promised.

To redo conventional-policy selection, write to a new settings file:

```bash
.venv/bin/python -m radio_rl.tune_baselines --output runs/new_baseline_settings.json
```

Baseline tuning sees training and validation only. The fitted predictor uses 60 training episodes for each candidate context length. Fixed channel, UCB window/bonus, and predictor context are selected on validation. Do not use final test scores to retune them.

## Formulation and methods

The observation has 48 values: tuned channel (8), eight queue rows (24), clocks (2), previous action (9), previous outcome (4), and validity (1). H=16 stacks 16 observations into 768 inputs. H=1 still sees the previous event; it is not memory-free. Actions 0–7 select a channel and 8 waits. A changed channel consumes one slot before another action can transmit. EDF packet scheduling is fixed for all policies.

Reward: `delivered - 2*(expired+overflow) - 0.02*retunes - 0.01*(attempts+0.2*retunes)`. The mission has 500 arrival slots plus 12 drain slots. Deadlines are inclusive. Current-slot interference is fixed before the action; a detected transmission influences a follower only after its positive delay.

Double DQN uses two 128-unit ReLU layers, a separate target network, compact episode-safe replay, Huber loss, gamma 0.99, and Adam. Six baselines are fixed channel, random hopping, cyclic hopping, stay-on-success/switch-on-failure, sliding-window UCB, and a smoothed history-conditioned myopic predictor. Hopping policies commit through retuning to an attempt. All policies see identical feedback; privileged ground truth is confined to diagnostic traces.

Read [the exact environment contract](docs/environment_spec.md), [evaluation protocol](docs/experiment_protocol.md), [research notes](docs/literature_notes.md), and [implementation decisions](docs/decisions.md). The full original [implementation plan](IMPLEMENTATION_PLAN.md) also records optional extensions; recurrent models and real hardware are not implemented in this core.

## Report and submission

The report builder reads measured result files and training logs. Put actual team members/IDs and repository URL into `report/metadata.json`, then generate the final two-column report:

```bash
.venv/bin/python scripts/build_report.py
```

Until those details are provided, use `--draft`; the output clearly identifies missing submission metadata and is not submission-ready. The builder produces two pages of main content plus a reference page, and refuses a final report when metadata is missing. Verify page rendering and course requirements before submitting.

For a private repository, grant access to `shubhambagwari` and `Divyaanshmertia`. Submission through the course platform is a separate step; the reproduction and release scripts do not submit coursework.

## Repository history and release

Published and release checkouts use normal Git. In the original execution workspace only, the existing `.git` was protected and empty, so genuine incremental commits were preserved separately:

```bash
git --git-dir=.development-history --work-tree=. log --oneline
.venv/bin/python scripts/release.py
```

The release creates `dist/radio-rl` with a normal `.git` directory, and a ZIP containing the source, artifacts, report, and actual history. Publish that checkout to the team's repository after reviewing it. Neither installed environments, full replay buffers, source paper PDFs, nor unrelated credential directories are included. See [provenance and assistance disclosure](docs/provenance.md).

## Limitations

Synthetic packet-level dynamics; ideal coordinated receiver and reliable ACKs; fixed packet sizes and powers; one link; eight channels; scripted rather than learned jammer. No calibrated milliseconds/watts, RF validation, or generalization to a different number of channels is claimed. An RL advantage is an experimental result, not an assumption. The final report must retain strong-baseline wins and failures rather than cherry-picking seeds or conditions.
