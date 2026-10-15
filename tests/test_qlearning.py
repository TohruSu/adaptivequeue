import copy
import json
import random
from pathlib import Path

import pytest

from adaptivequeue import Job, Simulation, WorkloadConfig, simulate
from adaptivequeue.metrics import calculate_metrics
from adaptivequeue.qlearning import (
    ACTION_NAMES,
    LearningConfig,
    QLearningAgent,
    encode_state,
    evaluate,
    step_action,
    train,
)
from adaptivequeue.workloads import WORKLOADS


@pytest.mark.parametrize(("size", "bucket"), [(1, 0), (2, 1), (3, 1), (4, 2), (7, 2), (8, 3)])
def test_queue_length_buckets(size: int, bucket: int) -> None:
    jobs = [Job(str(i), 0, 4, 30, 1, "test") for i in range(size)]
    assert encode_state(jobs, 10) == (bucket, 3, 0, False)


@pytest.mark.parametrize(("slack", "bucket"), [(-1, 0), (0, 1), (2, 1), (3, 2), (6, 2), (7, 3)])
def test_slack_subtracts_current_time_and_processing_duration(slack: int, bucket: int) -> None:
    job = Job("a", 0, 4, 10 + 4 + slack, 1, "test")
    assert encode_state([job], 10)[1] == bucket


@pytest.mark.parametrize(
    ("short", "total", "bucket"), [(0, 3, 0), (1, 4, 0), (1, 3, 1), (1, 2, 1), (2, 3, 2), (3, 3, 2)]
)
def test_short_job_proportion_buckets(short: int, total: int, bucket: int) -> None:
    jobs = [Job(str(i), 0, 2 if i < short else 3, 30, 1, "test") for i in range(total)]
    assert encode_state(jobs, 0)[2] == bucket


def test_state_uses_minimum_slack_and_detects_high_priority() -> None:
    jobs = [Job("a", 0, 4, 14, 1, "test"), Job("b", 0, 6, 14, 4, "test")]
    assert encode_state(jobs, 10) == (1, 0, 0, True)
    with pytest.raises(ValueError, match="nonempty"):
        encode_state([], 0)


def test_one_transition_executes_one_job_and_returns_its_cost() -> None:
    jobs = [
        Job("first", 2, 4, 5, 4, "test"),
        Job("arriving", 3, 1, 10, 1, "test"),
        Job("future", 20, 1, 25, 1, "test"),
    ]
    simulation = Simulation(jobs)
    assert simulation.current_time == 2
    assert simulation.waiting_jobs == (jobs[0],)
    transition = step_action(simulation, 1)
    assert (transition.job.start_time, transition.job.completion_time) == (2, 6)
    assert transition.reward == pytest.approx(-4 * (1 + 0.1 * 4))
    assert not transition.done
    assert transition.next_state == encode_state([jobs[1]], 6)
    assert simulation.waiting_jobs == (jobs[1],)
    assert len(simulation.result().jobs) == 1
    step_action(simulation, 0)
    assert simulation.current_time == 20
    assert simulation.waiting_jobs == (jobs[2],)
    terminal = step_action(simulation, 3)
    assert terminal.done and terminal.next_state is None
    assert terminal.job.completion_time == 21
    with pytest.raises(ValueError, match="completed"):
        step_action(simulation, 0)


@pytest.mark.parametrize(("action", "expected"), [(0, "a"), (1, "b"), (2, "c"), (3, "d")])
def test_action_indices_invoke_the_corresponding_baseline(action: int, expected: str) -> None:
    jobs = [
        Job("a", 0, 5, 20, 1, "test"),
        Job("b", 0, 1, 30, 1, "test"),
        Job("c", 0, 3, 4, 2, "test"),
        Job("d", 0, 4, 25, 4, "test"),
    ]
    transition = step_action(Simulation(jobs), action)
    assert transition.job.job.job_id == expected


@pytest.mark.parametrize("action", [-1, 4])
def test_invalid_action_does_not_execute_a_job(action: int) -> None:
    simulation = Simulation([Job("a", 0, 2, 10, 1, "test")])
    with pytest.raises(ValueError, match="index"):
        step_action(simulation, action)
    assert simulation.current_time == 0
    assert simulation.result().jobs == ()


def test_q_updates_use_next_max_and_no_bootstrap_at_terminal() -> None:
    agent = QLearningAgent(seed=0, config=LearningConfig(learning_rate=0.5, gamma=0.9))
    state = (0, 1, 2, False)
    next_state = (1, 1, 0, True)
    agent.q_table[state] = [1.0, 0.0, 0.0, 0.0]
    agent.q_table[next_state] = [2.0, 4.0, 3.0, 1.0]
    agent.update(state, 0, -2.0, next_state)
    assert agent.q_table[state][0] == pytest.approx(1 + 0.5 * (-2 + 0.9 * 4 - 1))
    agent.update(state, 0, -2.0, None)
    assert agent.q_table[state] == pytest.approx([-0.35, 0.0, 0.0, 0.0])


def test_training_updates_the_single_terminal_job() -> None:
    config = WorkloadConfig(num_jobs=1, duration_range=(2, 2), deadline_factor=10)
    agent = train(
        episodes=1,
        seed=7,
        workload_config=config,
        learning_config=LearningConfig(learning_rate=1, epsilon_start=0, epsilon_end=0),
    )
    job = WORKLOADS["mixed"](7, config)[0]
    state = encode_state([job], 0)
    assert agent.q_table[state] == pytest.approx([-job.priority_weight * 0.2, 0, 0, 0])


def test_decay_is_per_episode_and_stops_at_floor() -> None:
    config = LearningConfig(epsilon_start=0.8, epsilon_end=0.1, epsilon_decay=0.5)
    assert [config.epsilon(i) for i in range(5)] == pytest.approx([0.8, 0.4, 0.2, 0.1, 0.1])


def test_training_is_reproducible_and_does_not_use_global_rng() -> None:
    global_state = random.getstate()
    config = WorkloadConfig(num_jobs=15)
    first = train(episodes=12, seed=42, workload_config=config)
    replay = train(episodes=12, seed=42, workload_config=config)
    other = train(episodes=12, seed=43, workload_config=config)
    assert first.q_table == replay.q_table
    assert first.training == replay.training
    assert first.q_table != other.q_table
    assert random.getstate() == global_state


def test_save_load_preserves_predictions_metadata_and_metrics(tmp_path: Path) -> None:
    agent = train(episodes=10, seed=42, workload_config=WorkloadConfig(num_jobs=12))
    path = tmp_path / "policy.json"
    agent.save(path)
    loaded = QLearningAgent.load(path)
    assert loaded.q_table == agent.q_table
    assert loaded.config == agent.config
    assert loaded.training == agent.training
    assert all(loaded.greedy_action(state) == agent.greedy_action(state) for state in agent.q_table)
    assert loaded.greedy_action((3, 3, 2, False)) == agent.greedy_action((3, 3, 2, False))
    assert evaluate(loaded, episodes=3) == evaluate(agent, episodes=3)
    data = json.loads(path.read_text())
    assert data["actions"] == list(ACTION_NAMES)
    assert data["state_encoding"]["slack"] == "deadline - current_time - duration"
    assert data["training"]["seed"] == 42


def test_evaluation_is_greedy_frozen_and_uses_held_out_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = train(episodes=3, seed=42, workload_config=WorkloadConfig(num_jobs=8))
    jobs = [Job("long", 0, 5, 30, 1, "test"), Job("short", 0, 1, 30, 1, "test")]
    state = encode_state(jobs, 0)
    agent.q_table[state] = [0.0, 2.0, 1.0, 0.0]
    assert simulate(jobs, agent.policy).jobs[0].job.job_id == "short"
    unseen = (3, 3, 2, False)
    assert agent.greedy_action(unseen) == 0
    before = copy.deepcopy(agent.q_table)
    rng_before = agent._rng.getstate()

    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("evaluation must not explore or learn")

    monkeypatch.setattr(agent, "choose_action", fail)
    monkeypatch.setattr(agent, "update", fail)
    metrics = evaluate(agent, episodes=2, workload_config=WorkloadConfig(num_jobs=8))
    expected = [
        record
        for seed in (45, 46)
        for record in simulate(
            WORKLOADS["mixed"](seed, WorkloadConfig(num_jobs=8)), agent.policy
        ).jobs
    ]
    assert metrics == calculate_metrics(expected)
    assert agent.q_table == before
    assert agent._rng.getstate() == rng_before
    assert unseen not in agent.q_table
    with pytest.raises(ValueError, match="overlap"):
        evaluate(agent, seed=43, episodes=1)


def test_episode_reward_is_negative_total_cost() -> None:
    jobs = WORKLOADS["bursty"](5, WorkloadConfig(num_jobs=12))
    simulation = Simulation(jobs)
    reward = 0.0
    while not simulation.done:
        reward += step_action(simulation, 2).reward
    assert reward == pytest.approx(-len(jobs) * simulation.result().metrics.primary_cost)


def test_incompatible_saved_formulation_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    QLearningAgent(seed=0).save(path)
    data = json.loads(path.read_text())
    data["actions"].reverse()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incompatible"):
        QLearningAgent.load(path)
