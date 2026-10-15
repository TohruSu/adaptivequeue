# Scheduling experiments

## Setup and reproduction

Three independent Q-Learning models train on `mixed` for 1,000 episodes each,
starting at seeds 42, 2042, and 4042 (workload seed ranges 42–1041, 2042–3041,
and 4042–5041). Learning defaults are unchanged: alpha=0.1, gamma=1,
epsilon=max(0.05, 0.995**episode). Each model is evaluated greedily with no
exploration, learning, or Q-table insertion.

Evaluation uses `stable`, `bursty`, and `mixed`, each at seeds 20000–20029.
The generators produce 60 jobs with duration 1–6, weights 1/2/4, arrival rate
0.28, deadline factor 4, burst size 6, and burst window 3. Each workload/seed
is generated once and the same immutable job specifications go to all eight
policy instances: five baselines and three trained models. Random uses a separate
local RNG seeded with that episode's evaluation seed.

Before the final run, seeds 7 and 42 were checked on all three workloads. All five
baselines produced distinct execution orders in each episode. On FIFO's queue
snapshots, 54–59 decisions had multiple jobs waiting and the deterministic
heuristics disagreed at 48–57 of them. The final protocol above was then kept fixed;
there was no hyperparameter tuning or selection of a preferred trained model.

```sh
python -m pip install -e '.[benchmark]'
adaptivequeue benchmark --output-dir benchmark-results --retrain
```

Omit `--retrain` to reuse the three saved models. Cached models must match the
training seed, episode count, workload configuration, and learning settings.
The committed [reference outputs](../benchmarks/default) were generated with
Python 3.12.14 and Matplotlib 3.11.2.

## Output definitions

- `per_episode.csv`: 720 rows, with all five scheduling metrics for each
  workload/evaluation-seed/policy/training-seed combination. Baselines have a
  blank training seed and are evaluated once, not repeated for each RL run.
- `summary.csv`: 24 rows, one per workload/policy/training run. Metrics are
  arithmetic means of the 30 **episode metrics**. In particular, mean P95 waiting
  time averages episode percentiles; it is not a percentile over pooled jobs.
  The standalone `evaluate` command pools jobs across episodes instead.
  `std_primary_cost` is the sample standard deviation over evaluation episodes,
  not training variability or a confidence interval.
- `mean_cost_delta_vs_*` is mean paired `J(policy) - J(baseline)` using matching
  evaluation seeds; negative favors the policy. `cost_win_rate_vs_*` is the
  fraction of those 30 paired episodes with strictly lower cost; ties are not wins.
- `action_distribution.csv`: 36 rows, counting each underlying action separately
  for each workload and trained model. `fraction` covers all 1,800 decisions;
  `contended_fraction` covers only decisions with at least two jobs waiting.
- `figures/scheduling_cost.png` and `figures/deadline_miss_rate.png`: episode
  distributions, with median lines and mean triangles. Each panel uses its own
  vertical scale. Models remain separate; 30 episodes per box.
- `models/` and `protocol.json`: the three Q-tables and complete experiment settings.

## Observed results

Mean primary cost J over the same 30 evaluation episodes; lower is better:

| Policy / training seed | Stable | Bursty | Mixed |
| --- | ---: | ---: | ---: |
| FIFO | 16.531 | 52.744 | 19.360 |
| SPT | 10.415 | 31.018 | 12.651 |
| EDF | 11.441 | 39.999 | 9.328 |
| Priority | 9.852 | 30.867 | **9.212** |
| Random | 18.647 | 56.171 | 22.390 |
| Q-Learning / 42 | 9.270 | 28.804 | 9.284 |
| Q-Learning / 2042 | 10.164 | 31.471 | 10.402 |
| Q-Learning / 4042 | **9.210** | **27.130** | 9.900 |

Priority has the lowest mean cost among the classical baselines in all three
workloads. Models 42 and 4042 improve on it in stable and bursty workloads:
their paired mean differences are -0.582/-0.642 on stable and -2.062/-3.736 on
bursty. They win 23/22 of 30 stable episodes and 24/26 bursty episodes respectively.
Model 2042 is worse than Priority in both means (+0.312 and +0.605), despite
winning 19/30 paired episodes in each. Mean improvement and win frequency answer
different questions when cost differences vary substantially between episodes.

On `mixed`, the training workload, **all three models lose to Priority in mean
cost**: +0.072, +1.190, and +0.687 respectively. Model 42 narrowly beats EDF
in mean (-0.044), while models 2042 and 4042 lose to EDF (+1.075/+0.572).
Every model beats FIFO and Random in mean on every workload, but those comparisons
alone would overstate the value relative to stronger heuristics.

SPT has the lowest mean deadline-miss rate in every workload: 14.94%, 35.00%, and
20.72%. The best RL miss rates are 22.17%, 44.06%, and 24.22%. J penalizes weighted
tardiness and flow time rather than the unweighted count of misses, so the cost
ranking does not imply the same ranking on deadline-miss rate or waiting time.
All secondary metrics are retained in the CSVs.

Action mixes also differ across training runs. On bursty workloads, model 42
selects FIFO/SPT/EDF/Priority for 2.0%/53.8%/17.6%/26.6% of all decisions;
model 2042 uses 18.3%/41.2%/23.8%/16.8%, and model 4042 uses
8.2%/45.9%/18.8%/27.1%. Choices at single-job queues cannot affect that dispatch;
the CSV includes contended-only frequencies to distinguish these cases.

These runs show useful cost reductions for two training seeds on stable and
bursty loads, but no consistent advantage over a strong heuristic, even on mixed
loads. Three training seeds and one load/deadline setting do not establish broad
superiority. The models share the same 30 evaluation workloads per family, so their
results should not be pooled as 90 independent evaluation episodes. The coarse
state also maps different queues to the same observation; see the
[state design](simulation.md#tabular-q-learning-meta-scheduler).
