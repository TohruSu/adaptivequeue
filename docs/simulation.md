# Simulation

## Rules

Time starts at zero and uses integer units. A job specifies a unique string
`job_id`, nonnegative `arrival_time`, positive `duration`, nonnegative absolute
`deadline`, positive `priority_weight`, and descriptive `job_type`. An already
unachievable deadline is allowed. Specifications are immutable, so one workload
can be reused across policies.

One worker runs one job to completion without preemption. Whenever it is idle,
all jobs with arrival time at or before the current time enter the waiting queue.
The policy receives only an immutable snapshot of that queue and must select a
waiting job. It cannot observe future arrivals. Jobs arriving during execution
wait until completion; arrivals exactly at completion participate in the next
decision. The engine skips to completion or the next arrival rather than stepping
through ticks where no scheduling decision is possible. It runs until every job
finishes, including late jobs, without deliberately idling when work is available.

## Policies

| CLI name | Selects |
| --- | --- |
| `fifo` | Earliest arrival |
| `spt` | Shortest processing duration |
| `edf` | Earliest absolute deadline |
| `priority` | Highest priority weight |
| `random` | Uniformly random waiting job, using a local seeded RNG |

The first four break ties by earliest arrival, then lexicographically smallest
job ID. Random's queue is populated in arrival/ID order, making seeded results
independent of input ordering. Recreate a Random policy with the same seed for
each replay. Workload generation and policy selection use separate RNGs and do
not modify Python's global random state.

## Workloads

Every generator takes an explicit seed and an optional `WorkloadConfig`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `num_jobs` / `--jobs` | 60 | Jobs in an episode; zero gives an empty episode |
| `arrival_rate` / `--arrival-rate` | 0.28 | Target mean jobs per time unit |
| `deadline_factor` / `--deadline-factor` | 4.0 | Relative deadline budget; larger is looser |
| `duration_range` (Python API) | (1, 6) | Inclusive integer duration bounds |
| `burst_size` / `--burst-size` | 6 | Jobs per burst, with a smaller final burst if needed |
| `burst_window` / `--burst-window` | 3 | Burst offsets range from 0 to window minus 1 |

`stable` uses independent exponential interarrival gaps with a constant rate and
the first arrival at zero. Cumulative arrival times are rounded down to integers,
so several jobs can arrive together. `bursty` places batches within short windows;
burst starts have exponential gaps with mean `burst_size / arrival_rate`. Bursts
can overlap. `mixed` uses stable arrivals with interactive jobs (weight 4, deadline
multiplier 0.5), reports (weight 2, multiplier 1), and maintenance jobs (weight 1,
multiplier 2), chosen with equal probability.

Durations are uniform within the configured range. Stable and bursty workloads
choose weights uniformly from 1, 2, and 4, and descriptive types from email,
image, and backup. All deadline budgets are
`max(1, ceil(duration * deadline_factor * type_multiplier * U(0.75, 1.25)))`;
the absolute deadline is arrival plus budget. Stable and bursty type multipliers
are 1. Increasing arrival rate increases contention. The default rate gives
nominal utilization `0.28 * 3.5 = 0.98`; finite episodes vary by seed.

## Records and evaluation

`simulate(jobs, policy)` returns `SimulationResult`. Its `jobs` tuple holds
`JobResult` records in execution order, including each job, start time, completion
time, and derived waiting time, flow time, and tardiness:

- Waiting time = start minus arrival.
- Flow time = completion minus arrival.
- Tardiness = max(0, completion minus absolute deadline).
- Primary cost `J = mean(priority_weight * (tardiness + 0.1 * flow_time))`.
- Deadline miss rate = fraction of jobs with positive tardiness. Completion on
  the deadline counts as on time.

Lower primary cost is better. The result also reports average waiting time,
95th-percentile waiting time, and average flow time. The percentile linearly
interpolates sorted waiting times at index `0.95 * (n - 1)` (zero-based), and
equals the lone value for one job. Every aggregate metric is zero for an empty
episode. Cost is averaged over jobs, without normalizing by total priority weight.

Compare policies on identical generated jobs and settings. Seeds control both
workloads and Random decisions independently; several episode seeds give a more
useful comparison than a single run.
