# AdaptiveQueue

A lightweight Python project for experimenting with adaptive background job
scheduling. Includes a discrete-time, single-worker simulator, FIFO, SPT, EDF,
Priority, and seeded Random baselines, and a tabular Q-Learning meta-scheduler.

Requires Python 3.10 or newer. Simulation and learning have no runtime
dependencies; benchmark figures use the optional Matplotlib extra.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,benchmark]'
```

Run an episode with 60 jobs:

```sh
adaptivequeue --workload mixed --policy edf --seed 42
```

The CLI reports priority-weighted scheduling cost (lower is better), deadline miss
rate, average and 95th-percentile waiting time, and average flow time. Workloads
are `stable`, `bursty`, or `mixed`; policies are `fifo`, `spt`, `edf`, `priority`,
or `random`. Use the same seed and workload settings to compare policies fairly.

Train a meta-scheduler that chooses FIFO, SPT, EDF, or Priority at each decision,
then evaluate its frozen Q-table on held-out workload seeds:

```sh
adaptivequeue train --workload mixed --episodes 1000 --seed 42 --output policy.json
adaptivequeue evaluate --workload mixed --episodes 20 --model policy.json
```

Evaluation defaults to seeds following the saved training range, with exploration
disabled. Use `--seed` to choose a different disjoint range. Both commands accept
the workload options below; `train --help` lists learning and exploration settings.

Run the fixed paired benchmark (three independently trained models and all five
baselines, across three workloads and 30 evaluation seeds):

```sh
adaptivequeue benchmark --output-dir benchmark-results
```

Models are cached in the output directory; `--retrain` regenerates them.
See [experimental results and reproduction details](docs/experiments.md) for
the observed comparisons and CSV definitions.

```sh
python -m adaptivequeue --workload bursty --policy spt --seed 7 \
  --jobs 60 --arrival-rate 0.3 --deadline-factor 3
python -m pytest
ruff check .
ruff format --check .
mypy
```

See [simulation rules and parameters](docs/simulation.md) or run
`adaptivequeue --help`. The Python API also exposes per-job execution records.
The current scope is simulation, classical baselines, CPU-only tabular
Q-Learning, and reproducible benchmarking.
