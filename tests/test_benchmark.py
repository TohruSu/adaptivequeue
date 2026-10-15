import copy
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import pytest

from adaptivequeue import Job, Metrics, SimulationResult, WorkloadConfig, benchmark, simulate
from adaptivequeue.benchmark import EpisodeResult, run_benchmark, summarize
from adaptivequeue.policies import POLICY_NAMES, Policy
from adaptivequeue.qlearning import QLearningAgent, train
from adaptivequeue.workloads import WORKLOADS


def test_fresh_and_cached_benchmarks_are_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = {
        "training_seeds": (42, 2042),
        "training_episodes": 2,
        "evaluation_seeds": (20000, 20001),
        "workloads": ("stable", "mixed"),
        "plots": False,
    }
    global_state = random.getstate()
    first, second = tmp_path / "first", tmp_path / "second"
    results = run_benchmark(first, **settings)
    assert results == run_benchmark(second, **settings)
    assert len(results) == 2 * 2 * (5 + 2)
    for name in ("per_episode.csv", "summary.csv", "action_distribution.csv", "protocol.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    def fail(**kwargs: object) -> QLearningAgent:
        pytest.fail("a cached benchmark must not retrain")

    monkeypatch.setattr(benchmark, "train", fail)
    assert run_benchmark(first, **settings) == results
    assert random.getstate() == global_state
    with (first / "action_distribution.csv").open() as stream:
        actions = list(csv.DictReader(stream))
    assert len(actions) == 2 * 2 * 4
    for workload in settings["workloads"]:
        for seed in settings["training_seeds"]:
            rows = [
                r for r in actions if r["workload"] == workload and int(r["training_seed"]) == seed
            ]
            assert sum(int(r["selections"]) for r in rows) == 2 * 60
            assert sum(float(r["fraction"]) for r in rows) == pytest.approx(1.0)
            assert sum(float(r["contended_fraction"]) for r in rows) == pytest.approx(1.0)
            assert all(int(r["contended_selections"]) <= int(r["selections"]) for r in rows)


def test_every_policy_receives_the_same_generated_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated: dict[int, list[Job]] = {}
    observed: list[tuple[Job, ...]] = []

    def generate(seed: int, config: WorkloadConfig) -> list[Job]:
        assert seed not in generated
        generated[seed] = WORKLOADS["mixed"](seed, config)
        return generated[seed]

    def observe(jobs: Iterable[Job], policy: Policy) -> SimulationResult:
        assert jobs is generated[20000]
        observed.append(tuple(jobs))
        return simulate(jobs, policy)

    monkeypatch.setattr(benchmark, "WORKLOADS", {"mixed": generate})
    monkeypatch.setattr(benchmark, "simulate", observe)
    run_benchmark(
        tmp_path,
        training_seeds=(42,),
        training_episodes=2,
        evaluation_seeds=(20000,),
        workloads=("mixed",),
        plots=False,
    )
    assert len(generated) == 1
    assert len(observed) == 6
    assert all(jobs == observed[0] for jobs in observed)
    assert tuple(generated[20000]) == observed[0]


def test_aggregation_keeps_runs_separate_and_pairs_evaluation_seeds() -> None:
    results = []
    for seed, cost in ((1, 10.0), (2, 20.0)):
        for policy in POLICY_NAMES:
            results.append(EpisodeResult("mixed", seed, policy, None, Metrics(cost, 0.2, 2, 4, 6)))
        for training_seed, rl_cost in ((42, 8.0 if seed == 1 else 22.0), (2042, cost + 4)):
            metrics = Metrics(rl_cost, 0.1 * seed, 2 * seed, 10 * seed, 5 * seed)
            results.append(EpisodeResult("mixed", seed, "qlearning", training_seed, metrics))
    summary = summarize(results)
    assert len(summary) == 7
    runs = {row["training_seed"]: row for row in summary if row["policy"] == "qlearning"}
    assert runs[42]["episodes"] == 2
    assert runs[42]["mean_primary_cost"] == 15
    assert runs[42]["std_primary_cost"] == pytest.approx(98**0.5)
    assert runs[42]["mean_cost_delta_vs_fifo"] == 0
    assert runs[42]["cost_win_rate_vs_fifo"] == 0.5
    assert runs[42]["mean_deadline_miss_rate"] == pytest.approx(0.15)
    assert runs[42]["mean_p95_waiting_time"] == 15
    assert runs[2042]["mean_primary_cost"] == 19
    assert runs[2042]["mean_cost_delta_vs_fifo"] == 4
    assert runs[2042]["cost_win_rate_vs_fifo"] == 0


def test_benchmark_evaluation_is_frozen_and_counts_greedy_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = train(episodes=2, seed=42)
    table_before = copy.deepcopy(agent.q_table)
    rng_before = agent._rng.getstate()
    expected = simulate(WORKLOADS["mixed"](20000, WorkloadConfig()), agent.policy).metrics

    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("benchmark evaluation must not explore or learn")

    monkeypatch.setattr(agent, "choose_action", fail)
    monkeypatch.setattr(agent, "update", fail)
    monkeypatch.setattr(benchmark, "_load_or_train", lambda *args, **kwargs: {42: agent})
    rows = run_benchmark(
        tmp_path,
        training_seeds=(42,),
        training_episodes=2,
        evaluation_seeds=(20000,),
        workloads=("mixed",),
        plots=False,
    )
    assert rows[-1].metrics == expected
    assert agent.q_table == table_before
    assert agent._rng.getstate() == rng_before
    with (tmp_path / "action_distribution.csv").open() as stream:
        actions = list(csv.DictReader(stream))
    assert sum(int(row["selections"]) for row in actions) == 60


def test_cached_model_configuration_must_match(tmp_path: Path) -> None:
    settings = {
        "training_seeds": (42,),
        "training_episodes": 1,
        "evaluation_seeds": (20000,),
        "workloads": ("mixed",),
        "plots": False,
    }
    run_benchmark(tmp_path, **settings)
    path = tmp_path / "models" / "qlearning_seed_42.json"
    data = json.loads(path.read_text())
    data["training"]["episodes"] = 999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="cached model configuration"):
        run_benchmark(tmp_path, **settings)


def test_plot_files_are_generated(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    run_benchmark(
        tmp_path,
        training_seeds=(42,),
        training_episodes=1,
        evaluation_seeds=(20000, 20001),
        workloads=("mixed",),
    )
    for name in ("scheduling_cost.png", "deadline_miss_rate.png"):
        assert (tmp_path / "figures" / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
