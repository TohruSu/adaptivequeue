from typing import Sequence

import pytest

from adaptivequeue import Job, make_policy, simulate
from adaptivequeue.policies import spt


def test_arrivals_do_not_preempt_and_idle_time_is_skipped() -> None:
    jobs = [
        Job("late", 20, 1, 25, 1, "test"),
        Job("boundary", 8, 2, 10, 2, "test"),
        Job("short", 3, 1, 4, 4, "test"),
        Job("long", 2, 6, 20, 1, "test"),
    ]
    result = simulate(jobs, make_policy("spt", seed=0))
    assert [(r.job.job_id, r.start_time, r.completion_time) for r in result.jobs] == [
        ("long", 2, 8),
        ("short", 8, 9),
        ("boundary", 9, 11),
        ("late", 20, 21),
    ]
    assert len(result.jobs) == len(jobs)
    assert all(r.completion_time - r.start_time == r.job.duration for r in result.jobs)


def test_policy_sees_available_jobs_and_current_time_at_each_decision() -> None:
    jobs = [
        Job("running", 0, 5, 20, 1, "test"),
        Job("waiting", 1, 4, 20, 1, "test"),
        Job("at-completion", 5, 1, 20, 1, "test"),
        Job("future", 30, 1, 40, 1, "test"),
    ]
    observations = []

    def observe(waiting: Sequence[Job], current_time: int) -> Job:
        assert isinstance(waiting, tuple)
        observations.append((current_time, tuple(job.job_id for job in waiting)))
        return spt(waiting, current_time)

    result = simulate(jobs, observe)
    assert observations == [
        (0, ("running",)),
        (5, ("waiting", "at-completion")),
        (6, ("waiting",)),
        (30, ("future",)),
    ]
    assert [r.start_time for r in result.jobs] == [0, 5, 6, 30]


def test_duplicate_ids_are_rejected() -> None:
    job = Job("duplicate", 0, 1, 5, 1, "test")
    with pytest.raises(ValueError, match="unique"):
        simulate([job, job], make_policy("fifo", seed=0))


def test_policy_cannot_select_a_future_job() -> None:
    jobs = [Job("available", 0, 2, 10, 1, "test"), Job("future", 3, 1, 10, 1, "test")]
    with pytest.raises(ValueError, match="waiting queue"):
        simulate(jobs, lambda waiting, current_time: jobs[1])


@pytest.mark.parametrize(
    ("arrival", "duration", "deadline", "weight"),
    [(-1, 1, 5, 1), (0, 0, 5, 1), (0, 1, -1, 1), (0, 1, 5, 0), (0.5, 1, 5, 1)],
)
def test_invalid_job_values_are_rejected(
    arrival: int, duration: int, deadline: int, weight: int
) -> None:
    with pytest.raises(ValueError):
        Job("invalid", arrival, duration, deadline, weight, "test")
