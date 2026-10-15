import random
from dataclasses import replace

import pytest

from adaptivequeue import WorkloadConfig, make_policy, simulate
from adaptivequeue.workloads import WORKLOADS


@pytest.mark.parametrize("name", WORKLOADS)
def test_workload_and_episode_are_reproducible_without_global_rng_changes(name: str) -> None:
    generate = WORKLOADS[name]
    config = WorkloadConfig()
    global_state = random.getstate()
    jobs = generate(42, config)
    assert jobs == generate(42, config)
    assert jobs != generate(7, config)
    assert simulate(jobs, make_policy("random", seed=42)) == simulate(
        generate(42, config), make_policy("random", seed=42)
    )
    assert random.getstate() == global_state
    assert len(jobs) == len({job.job_id for job in jobs}) == 60
    assert all(1 <= job.duration <= 6 and job.priority_weight in (1, 2, 4) for job in jobs)
    assert all(job.deadline > job.arrival_time and job.job_type for job in jobs)


@pytest.mark.parametrize("name", WORKLOADS)
def test_workload_configuration_controls_size_load_and_deadlines(name: str) -> None:
    generate = WORKLOADS[name]
    config = WorkloadConfig(num_jobs=13, duration_range=(2, 2), burst_size=4)
    jobs = generate(42, config)
    faster = generate(42, replace(config, arrival_rate=1.0))
    looser = generate(42, replace(config, deadline_factor=8.0))
    assert len(jobs) == 13
    assert all(job.duration == 2 for job in jobs)
    assert max(job.arrival_time for job in faster) < max(job.arrival_time for job in jobs)
    assert [job.arrival_time for job in jobs] == [job.arrival_time for job in looser]
    assert all(a.deadline < b.deadline for a, b in zip(jobs, looser))
    assert generate(42, WorkloadConfig(num_jobs=0)) == []


def test_burst_windows_and_mixed_job_classes() -> None:
    config = WorkloadConfig(num_jobs=6, burst_size=6, burst_window=3)
    burst = WORKLOADS["bursty"](42, config)
    assert all(0 <= job.arrival_time < 3 for job in burst)
    mixed = WORKLOADS["mixed"](42, WorkloadConfig())
    assert {(job.job_type, job.priority_weight) for job in mixed} == {
        ("interactive", 4),
        ("report", 2),
        ("maintenance", 1),
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"num_jobs": -1},
        {"arrival_rate": 0},
        {"arrival_rate": float("nan")},
        {"deadline_factor": -1},
        {"deadline_factor": float("inf")},
        {"duration_range": (3, 2)},
        {"burst_size": 0},
        {"burst_window": 0},
    ],
)
def test_invalid_configuration_is_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        WorkloadConfig(**changes)
