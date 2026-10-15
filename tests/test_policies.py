import pytest

from adaptivequeue import Job, make_policy, simulate
from adaptivequeue.policies import POLICIES


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("fifo", ["a", "b", "c", "d"]),
        ("spt", ["c", "b", "d", "a"]),
        ("edf", ["c", "d", "a", "b"]),
        ("priority", ["b", "d", "c", "a"]),
    ],
)
def test_scheduling_order(name: str, expected: list[str]) -> None:
    jobs = [
        Job("d", 0, 2, 4, 4, "test"),
        Job("b", 0, 2, 9, 4, "test"),
        Job("a", 0, 4, 7, 1, "test"),
        Job("c", 0, 1, 3, 2, "test"),
    ]
    result = simulate(jobs, make_policy(name, seed=1))
    assert [record.job.job_id for record in result.jobs] == expected


@pytest.mark.parametrize("name", POLICIES)
def test_ties_use_arrival_then_job_id(name: str) -> None:
    jobs = [
        Job("b", 1, 2, 9, 2, "test"),
        Job("a", 1, 2, 9, 2, "test"),
        Job("z", 0, 2, 9, 2, "test"),
    ]
    policy = make_policy(name, seed=1)
    assert policy(jobs, 1).job_id == "z"
    assert policy(jobs[:2], 1).job_id == "a"


def test_random_policy_replays_seeded_choices() -> None:
    jobs = [Job(str(i), 0, 1, 10, 1, "test") for i in range(8)]
    first = simulate(jobs, make_policy("random", seed=42))
    replay = simulate(reversed(jobs), make_policy("random", seed=42))
    other = simulate(jobs, make_policy("random", seed=7))
    assert first == replay
    assert [record.job.job_id for record in first.jobs] == ["1", "0", "7", "4", "3", "2", "5", "6"]
    assert first.jobs != other.jobs


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("missing", seed=0)
