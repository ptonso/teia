from __future__ import annotations

import unittest
from collections import namedtuple
from types import SimpleNamespace

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset

from teia.core.callbacks import (
    GradSafeRichModelSummary,
    _safe_parse_model_summary_shape,
)
from teia.core.module.net import TeiaNetModule
from teia.callbacks import LossWeightWarmupCallback


RawOutput = namedtuple("RawOutput", ["tensor", "metadata", "nested"])


class ShapeNoneMetadata:
    shape = None


class StructuredSummaryModule(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)
        self.example_input_array = torch.zeros(1, 3)

    def forward(self, x: torch.Tensor) -> RawOutput:
        out = self.layer(x)
        return RawOutput(
            tensor=out,
            metadata=ShapeNoneMetadata(),
            nested={"optional": None, "callable": lambda: None},
        )

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        x = batch[0]
        return self.layer(x).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class MeanLoss(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean()


LossBatch = namedtuple("LossBatch", ["x"])


class LossWeightWarmupCallbackTests(unittest.TestCase):
    def _module(self, *, terms: list[str] | None = None, weights: dict[str, float] | None = None):
        terms = ["loss.kld"] if terms is None else terms
        weights = {key: 8.0 for key in terms} if weights is None else weights
        return SimpleNamespace(
            _loss_records=[SimpleNamespace(name="loss_kld", out_key=terms)],
            _pipeline_records=[],
            _activation_records=[],
            _loss_term_weight=dict(weights),
        )

    def test_scales_target_loss_from_start_to_final_weight(self) -> None:
        module = self._module()
        callback = LossWeightWarmupCallback(loss_attr="loss_kld", warmup_epochs=20)

        callback.on_train_epoch_start(SimpleNamespace(current_epoch=0), module)
        self.assertEqual(module._loss_term_weight["loss.kld"], 0.0)
        callback.on_train_epoch_start(SimpleNamespace(current_epoch=10), module)
        self.assertEqual(module._loss_term_weight["loss.kld"], 4.0)
        callback.on_train_epoch_start(SimpleNamespace(current_epoch=20), module)
        self.assertEqual(module._loss_term_weight["loss.kld"], 8.0)

    def test_scales_every_output_from_the_target_loss_node(self) -> None:
        module = self._module(terms=["loss.a", "loss.b"], weights={"loss.a": 2.0, "loss.b": 4.0})
        callback = LossWeightWarmupCallback(loss_attr="loss_kld", warmup_epochs=4, start_scale=0.25)

        callback.on_train_epoch_start(SimpleNamespace(current_epoch=2), module)

        self.assertEqual(module._loss_term_weight["loss.a"], 1.25)
        self.assertEqual(module._loss_term_weight["loss.b"], 2.5)

    def test_power_curve_scales_non_linearly(self) -> None:
        module = self._module()
        callback = LossWeightWarmupCallback(
            loss_attr="loss_kld",
            warmup_epochs=10,
            curve="power",
            power=2.0,
        )

        callback.on_train_epoch_start(SimpleNamespace(current_epoch=5), module)

        self.assertEqual(module._loss_term_weight["loss.kld"], 2.0)

    def test_train_end_restores_final_weight(self) -> None:
        module = self._module()
        callback = LossWeightWarmupCallback(loss_attr="loss_kld", warmup_epochs=20)

        callback.on_train_epoch_start(SimpleNamespace(current_epoch=10), module)
        callback.on_train_end(SimpleNamespace(), module)

        self.assertEqual(module._loss_term_weight["loss.kld"], 8.0)

    def test_fails_fast_for_bad_loss_alias(self) -> None:
        callback = LossWeightWarmupCallback(loss_attr="missing")
        with self.assertRaisesRegex(AttributeError, "no loss node alias"):
            callback.on_train_epoch_start(SimpleNamespace(current_epoch=0), self._module())

    def test_fails_fast_when_alias_is_not_a_loss_node(self) -> None:
        module = SimpleNamespace(
            _loss_records=[],
            _pipeline_records=[SimpleNamespace(name="head")],
            _activation_records=[],
            _loss_term_weight={},
        )
        callback = LossWeightWarmupCallback(loss_attr="head")
        with self.assertRaisesRegex(TypeError, "not a loss node"):
            callback.on_train_epoch_start(SimpleNamespace(current_epoch=0), module)

    def test_fails_fast_for_unrouted_loss_terms(self) -> None:
        callback = LossWeightWarmupCallback(loss_attr="loss_kld")
        with self.assertRaisesRegex(ValueError, "unrouted loss terms"):
            callback.on_train_epoch_start(SimpleNamespace(current_epoch=0), self._module(weights={}))

    def test_generated_loss_total_and_route_sums_use_live_weight_map(self) -> None:
        module = TeiaNetModule(
            loss={
                "_target_": "unused.MeanLoss",
                "in": "batch.x", "out": {"loss.mean": None}, "weight": 3.0,
            },
        )
        module.loss = MeanLoss()
        batch = LossBatch(x=torch.tensor([2.0]))

        loss_log = module._generated_losses(batch, module._pred_type())
        self.assertEqual(float(loss_log["loss"]), 6.0)
        self.assertEqual(float(module._route_sums(loss_log)["default"]), 6.0)

        module._loss_term_weight["loss.mean"] = 1.5
        loss_log = module._generated_losses(batch, module._pred_type())

        self.assertEqual(float(loss_log["loss"]), 3.0)
        self.assertEqual(float(module._route_sums(loss_log)["default"]), 3.0)


class GradSafeRichModelSummaryTests(unittest.TestCase):
    def test_safe_shape_parser_preserves_tensor_shapes(self) -> None:
        self.assertEqual(_safe_parse_model_summary_shape(torch.zeros(2, 3)), [2, 3])

    def test_safe_shape_parser_treats_shape_none_as_unknown(self) -> None:
        from lightning.pytorch.utilities.model_summary import model_summary as lightning_model_summary

        self.assertEqual(_safe_parse_model_summary_shape(ShapeNoneMetadata()), lightning_model_summary.UNKNOWN_SIZE)

    def test_safe_shape_parser_handles_nested_structured_outputs(self) -> None:
        parsed = _safe_parse_model_summary_shape(
            RawOutput(
                tensor=torch.zeros(1, 2),
                metadata=ShapeNoneMetadata(),
                nested={"optional": None, "callable": lambda: None},
            )
        )

        self.assertEqual(parsed[0], [1, 2])
        self.assertEqual(parsed[1], "?")
        self.assertEqual(parsed[2]["optional"], "?")
        self.assertEqual(parsed[2]["callable"], "?")

    def test_summary_callback_handles_shape_less_structured_outputs(self) -> None:
        trainer = L.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            accelerator="cpu",
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[GradSafeRichModelSummary(max_depth=1)],
        )
        loader = DataLoader(TensorDataset(torch.ones(2, 3)), batch_size=1)

        trainer.fit(StructuredSummaryModule(), train_dataloaders=loader)


if __name__ == "__main__":
    unittest.main()
