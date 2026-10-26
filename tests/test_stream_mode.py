"""Stream mode on a test-local engine and contract (teia:core/datamodule.md#stream-mode)."""

import pytest
import torch

from teia.base.task import Tensor, TaskContract
from teia.core.datamodule import TeiaDataModule
from teia.core.datamodule.interactive._cycle import Cycle

STACK = "teia.node.data.collate.tensor.Stack"


class StreamTask(TaskContract):
    batch = {
        "s_obs": Tensor("B *"),
        "s_next": Tensor("B *"),
        "s_act": Tensor("B *"),
        "s_val": Tensor("B *"),
        "s_rew": Tensor("B *", target=True),
        "s_term": Tensor("B *", target=True),
        "s_trunc": Tensor("B *", target=True),
        "s_epi": Tensor("B *", target=True),
        "s_w": Tensor("B *"),
        "s_idx": Tensor("B *"),
        "s_disc": Tensor("B *"),
        "s_ret": Tensor("B *"),
        "s_adv": Tensor("B *"),
    }
    capture = {"policy.action": Tensor("B *"), "policy.value": Tensor("B")}
    feedback = {"policy.action": "s_act", "policy.value": "s_val"}


class PolicyStub:
    """Stands in for the netmodule: acts through ``act_forward`` only."""

    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls = 0
        self.deterministic: list[bool] = []

    def act_forward(self, batch, *, deterministic, generator):
        self.calls += 1
        self.deterministic.append(deterministic)
        n = batch.s_obs.shape[0]
        return {"policy.action": torch.ones(n, 1), "policy.value": torch.full((n,), 0.5)}

    def __call__(self, *args):
        raise AssertionError("the data graph must not run the full forward")


def _collate(field, key, dtype="float32"):
    return {"_target_": STACK, "dtype": dtype, "in": f"item.{key}", "out": f"batch.{field}"}


def _dm(*, reset_policy="per_env", runner="sync", buffer="replay", total=18, stats="fixed"):
    out = ["item.obs", "item.reward", "item.term", "item.trunc"] + (["item.valid"] if reset_policy == "all_done" else [])
    roles = {"reward": "item.reward", "terminated": "item.term", "truncated": "item.trunc"}
    if reset_policy == "all_done":
        roles["valid"] = "item.valid"
    if buffer == "replay":
        roll = {
            "_target_": "teia.core.datamodule.interactive.replay.ReplayBuffer",
            "warmup_steps": 3, "minibatch_size": 4, "capacity": 60,
            "in": roles, "out": ["item.s_w", "item.s_idx", "item.s_disc"], "stage": "stream",
        }
    else:
        roll = {
            "_target_": "teia.core.datamodule.interactive.rollout.RolloutBuffer",
            "rollout_steps": 4, "minibatch_size": 4, "update_epochs": 1,
            "in": {**roles, "value": "item.s_val"}, "out": ["item.s_ret", "item.s_adv"], "stage": "stream",
        }
    runner_cfg = {"total_env_steps": total, "eval_episodes": 3, "episode_id": "item.episode", "return_key": "item.reward"}
    runner_cfg["_target_"] = (
        "teia.core.datamodule.interactive.sync.SyncRunner" if runner == "sync" else "teia.core.datamodule.interactive.async_runner.AsyncRunner"
    )
    if runner == "async":
        runner_cfg["parallel"] = "inline"
    dm = TeiaDataModule(
        env={"_target_": "_fake_engine.FakeEngine", "num_envs": 3, "length": 3, "reset_policy": reset_policy, "out": out},
        norm={"_target_": "teia.node.data.transform.normalize.Normalize", "stats": stats, "mean": 0.0, "std": 1.0, "in": "item.obs", "out": "item.obs"}
        if stats == "fixed"
        else {"_target_": "teia.node.data.transform.normalize.Normalize", "stats": stats, "in": "item.obs", "out": "item.obs"},
        roll=roll,
        s_obs=_collate("s_obs", "obs"),
        s_next=_collate("s_next", "next_obs"),
        s_act=_collate("s_act", "s_act"),
        s_val=_collate("s_val", "s_val"),
        s_rew=_collate("s_rew", "reward"),
        s_term=_collate("s_term", "term"),
        s_trunc=_collate("s_trunc", "trunc"),
        s_epi=_collate("s_epi", "episode", "int64"),
        runner=runner_cfg,
        **{f: _collate(f, f) for f in (("s_w", "s_idx", "s_disc") if buffer == "replay" else ("s_ret", "s_adv"))},
    )
    dm.bind_contract(StreamTask())
    dm.bind_module(PolicyStub())
    return dm


def _cycle(dm, *, training=True, deterministic=False):
    dm._plan()
    ctx = dm._regime.context(dm._policy_module(), training=training)
    return Cycle(ctx, deterministic=deterministic, return_key="item.reward", episode_id="item.episode")


def test_a_row_holds_the_acting_state_the_outcome_and_the_final_successor():
    dm = _dm()
    cycle = _cycle(dm)
    rows = [cycle.step() for _ in range(4)]
    third = rows[2]
    assert third["item.obs"][0].tolist() == [2.0, 2.0]
    assert third["item.next_obs"][0].tolist() == [3.0, 3.0]
    assert third["item.term"].tolist() == [True, False, False]
    assert third["item.reward"].tolist() == [1.0, 1.0, 1.0]
    assert rows[3]["item.obs"][0].tolist() == [0.0, 0.0]


def test_feedback_writes_the_captured_atoms_and_episode_ids_follow_the_boundary():
    cycle = _cycle(_dm())
    rows = [cycle.step() for _ in range(4)]
    assert rows[0]["item.s_act"].tolist() == [[1.0]] * 3
    assert rows[0]["item.s_val"].tolist() == [0.5] * 3
    assert rows[2]["item.episode"].tolist() == [0, 1, 2]
    assert rows[3]["item.episode"].tolist() == [3, 1, 2]


def test_acting_calls_only_the_capture_slice_once_per_step():
    dm = _dm()
    cycle = _cycle(dm)
    for _ in range(5):
        cycle.step()
    assert dm._policy_module().calls == 6


def test_all_done_idles_finished_environments_until_everyone_ends():
    cycle = _cycle(_dm(reset_policy="all_done"))
    rows = [cycle.step() for _ in range(6)]
    valid = torch.stack([row["item.valid"] for row in rows])
    assert valid[:, 0].tolist() == [True, True, True, False, False, True]
    assert rows[4]["item.term"].tolist() == [False, False, True]


def test_the_probe_publishes_shapes_without_a_network():
    meta = _dm().batch_meta()
    assert meta["s_obs"] == (2,) and meta["s_next"] == (2,) and meta["s_act"] == (1,)
    assert meta["s_rew"] == () and meta["s_w"] == () and "s_adv" not in meta


def test_the_runner_yields_collated_minibatches_and_reports_episodes():
    dm = _dm(total=24)
    batches = list(dm.train_dataloader())
    assert batches and batches[0].s_obs.shape == (4, 2)
    metrics = dm._runner.metrics()
    assert metrics["episode/count"] >= 1 and metrics["episode/return"] == pytest.approx(3.8)


def test_the_inline_async_runner_matches_the_sync_contract():
    batches = list(_dm(runner="async", total=15).train_dataloader())
    assert batches[0].s_obs.shape == (4, 2) and batches[0].s_disc.shape == (4,)


def test_the_rollout_buffer_bootstraps_from_the_captured_value():
    batches = list(_dm(buffer="rollout", total=12).train_dataloader())
    assert batches[0].s_adv.shape == (4,) and batches[0].s_ret.shape == (4,)


def test_eval_runs_on_the_open_engine_deterministically_and_leaves_training_state_alone():
    dm = _dm(stats="running")
    node = dm._nodes["norm"]
    list(dm.train_dataloader())
    seen = node.params()["mean"]
    engine = dm._engine_node
    batches = list(dm.val_dataloader())
    assert engine.opened == 1
    assert dm._policy_module().deterministic[-1] is True
    assert node.params()["mean"] == seen and node.training is True
    assert all(b.s_epi.shape == (3,) for b in batches)
    assert sum(int(b.s_term.sum()) for b in batches) >= 3


def test_an_export_chain_ends_at_a_key_the_runner_supplies():
    from teia.core.export.preprocess_slice import resolve_data_graph_chain

    dm = _dm()
    steps, payload_key, _ = resolve_data_graph_chain(dm, "s_next")
    assert payload_key == "next_obs" and [s.owner for s in steps] == ["Stack"]


def test_an_episode_aggregate_over_a_capture_with_no_ended_episode_reports_zero():
    import numpy as np

    from teia.node.eval.metric.aligned.episode_aggregate import EpisodeAggregate

    out = EpisodeAggregate().from_source(None, reward=np.ones(2), terminated=np.zeros(2), truncated=np.zeros(2), episode_id=np.zeros(2))
    assert out["episodes"] == 0 and out["rows"] == []
