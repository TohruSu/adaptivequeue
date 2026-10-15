# AdaptiveQueue

A lightweight Python project for experimenting with adaptive background job
scheduling. The first milestone provides a discrete-time, single-worker simulator
and FIFO, SPT, EDF, Priority, and seeded Random baselines.

Requires Python 3.10 or newer. There are no runtime dependencies.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run an episode with 60 jobs:

```sh
adaptivequeue --workload mixed --policy edf --seed 42
```

The CLI reports priority-weighted scheduling cost (lower is better), deadline miss
rate, average and 95th-percentile waiting time, and average flow time. Workloads
are `stable`, `bursty`, or `mixed`; policies are `fifo`, `spt`, `edf`, `priority`,
or `random`. Use the same seed and workload settings to compare policies fairly.

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
Reinforcement learning is planned for a later milestone; the current scope is
simulation and conventional scheduling baselines.
