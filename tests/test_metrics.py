import pytest

from adaptivequeue import Job, make_policy, simulate


def test_job_times_and_metrics_match_hand_calculation() -> None:
    jobs = [
        Job("a", 0, 3, 3, 1, "test"),
        Job("b", 1, 2, 4, 4, "test"),
        Job("c", 5, 1, 5, 2, "test"),
    ]
    result = simulate(jobs, make_policy("fifo", seed=0))
    assert [(r.start_time, r.completion_time) for r in result.jobs] == [(0, 3), (3, 5), (5, 6)]
    assert [r.waiting_time for r in result.jobs] == [0, 2, 0]
    assert [r.flow_time for r in result.jobs] == [3, 4, 1]
    assert [r.tardiness for r in result.jobs] == [0, 1, 1]

    metrics = result.metrics
    # Per-job costs: 0.3, 4 * (1 + 0.4), 2 * (1 + 0.1).
    assert metrics.primary_cost == pytest.approx(2.7)
    assert metrics.deadline_miss_rate == pytest.approx(2 / 3)
    assert metrics.average_waiting_time == pytest.approx(2 / 3)
    assert metrics.p95_waiting_time == pytest.approx(1.8)
    assert metrics.average_flow_time == pytest.approx(8 / 3)


def test_empty_episode_has_zero_metrics_and_never_calls_policy() -> None:
    def fail_if_called(waiting: object) -> Job:
        pytest.fail("an empty episode must not request a scheduling decision")

    result = simulate([], fail_if_called)
    assert result.jobs == ()
    assert result.metrics.primary_cost == 0
    assert result.metrics.deadline_miss_rate == 0
    assert result.metrics.average_waiting_time == 0
    assert result.metrics.p95_waiting_time == 0
    assert result.metrics.average_flow_time == 0


def test_single_job_finishing_on_deadline_is_not_a_miss() -> None:
    result = simulate([Job("a", 10, 2, 12, 4, "test")], make_policy("fifo", seed=0))
    assert result.metrics.deadline_miss_rate == 0
    assert result.metrics.p95_waiting_time == 0
    assert result.metrics.primary_cost == pytest.approx(0.8)
