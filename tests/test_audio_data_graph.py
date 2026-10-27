"""End-to-end audio data graph: ``batch.audio``/``batch.waveform`` resolve, ``num_classes`` publishes,
and augment is train-split-only. Mirrors ``test_tabular_data_graph.py``'s direct-node-dict pattern."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from teia.core.datamodule import TeiaDataModule

try:
    import soundfile as sf
    import torchaudio  # noqa: F401

    _HAS_TORCHAUDIO = True
except ImportError:
    _HAS_TORCHAUDIO = False


def _write_sine(path: Path, *, sample_rate: int = 16000, seconds: float = 1.0, freq: float = 440.0) -> None:
    t = torch.arange(int(sample_rate * seconds), dtype=torch.float32) / sample_rate
    wave = (0.5 * torch.sin(2 * math.pi * freq * t)).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, sample_rate)


def _build_class_tree(root: Path) -> None:
    for split, classes in (("train", ["cat", "dog"]), ("val", ["cat", "dog"]), ("test", ["cat", "dog"])):
        for i, cls in enumerate(classes):
            for j in range(2):
                _write_sine(root / split / cls / f"{cls}_{j}.wav", freq=220.0 * (i + 1) * (j + 1))


def _mel_nodes() -> dict:
    return {
        "auds": {
            "_target_": "teia.node.data.reader.audio.AudioFolderReader",
            "out": ["item.audio_path", "item.label_name"],
        },
        "enc": {
            "_target_": "teia.node.data.transform.labels.EncodeLabels",
            "in": "item.label_name",
            "out": "item.cls",
            "phase": "target",
        },
        "dec": {
            "_target_": "teia.node.data.transform.decode.DecodeAudio",
            "sample_rate": 16000,
            "duration": 0.5,
            "in": "item.audio_path",
            "out": "item.waveform",
        },
        "mel": {
            "_target_": "teia.node.data.transform.stft.MelSpectrogram",
            "sample_rate": 16000,
            "n_mels": 32,
            "in": "item.waveform",
            "out": "item.audio",
        },
        "aug": {
            "_target_": "teia.node.data.transform.augment.SpecAugment",
            "p": 1.0,
            "in": ["item.audio"],
            "out": ["item.audio"],
            "phase": "augment",
        },
        "audio_stack": {
            "_target_": "teia.node.data.collate.tensor.Stack",
            "in": "item.audio",
            "out": "batch.audio",
        },
        "cls_stack": {
            "_target_": "teia.node.data.collate.tensor.ClassLabels",
            "in": "item.cls",
            "out": "batch.cls",
        },
    }


@unittest.skipUnless(_HAS_TORCHAUDIO, "torchaudio/soundfile not installed")
class AudioMelGraphTest(unittest.TestCase):
    def test_mel_graph_batches_and_publishes_num_classes(self) -> None:
        with _tmp_dir() as root:
            _build_class_tree(root)
            dm = TeiaDataModule(
                data_root=str(root),
                batch_size=2,
                num_workers=0,
                **_mel_nodes(),
            )
            dm.setup()

            self.assertEqual(dm.meta()["num_classes"], 2)
            self.assertEqual(sorted(dm.meta()["class_names"]), ["cat", "dog"])

            batch = next(iter(dm.train_dataloader()))
            self.assertEqual(batch.audio.shape[0], 2)
            self.assertEqual(batch.audio.shape[1], 1)
            self.assertEqual(batch.audio.shape[2], 32)
            self.assertEqual(tuple(batch.cls.shape), (2,))

            val_batch = next(iter(dm.val_dataloader()))
            self.assertEqual(val_batch.audio.shape[1:], batch.audio.shape[1:])

    def test_augment_is_train_split_only(self) -> None:
        # Determinism check: an eval-split item run twice through the item pipeline must be
        # bit-identical (no augment applied), while the augment node itself is proven to mutate
        # data in test_audio_data_components.py::SpecAugmentTest.
        with _tmp_dir() as root:
            _build_class_tree(root)
            dm = TeiaDataModule(
                data_root=str(root),
                batch_size=2,
                num_workers=0,
                **_mel_nodes(),
            )
            dm.setup()
            val_ds = dm.val_dataloader().dataset
            first = val_ds[0]["item.audio"].clone()
            second = val_ds[0]["item.audio"].clone()
            self.assertTrue(torch.equal(first, second))


def _mel_multi_nodes() -> dict:
    nodes = _mel_nodes()
    nodes["auds"] = {
        "_target_": "teia.node.data.reader.audio.AudioFolderReader",
        "multi_label": True,
        "out": ["item.audio_path", "item.label_names"],
    }
    nodes["enc"] = {
        "_target_": "teia.node.data.transform.labels.MultiHotLabels",
        "in": "item.label_names",
        "out": "item.cls",
        "phase": "target",
    }
    nodes["cls_stack"] = {
        "_target_": "teia.node.data.collate.tensor.MultiHotClasses",
        "in": "item.cls",
        "out": "batch.cls",
    }
    return nodes


@unittest.skipUnless(_HAS_TORCHAUDIO, "torchaudio/soundfile not installed")
class AudioMultiLabelGraphTest(unittest.TestCase):
    def test_multi_hot_batch_shape(self) -> None:
        with _tmp_dir() as root:
            # Same-stem files across two class dirs -> a single multi-labeled item.
            _write_sine(root / "train" / "cat" / "shared.wav", freq=220.0)
            _write_sine(root / "train" / "dog" / "shared.wav", freq=440.0)
            _write_sine(root / "val" / "cat" / "a.wav", freq=220.0)
            _write_sine(root / "val" / "dog" / "b.wav", freq=440.0)
            dm = TeiaDataModule(
                data_root=str(root),
                batch_size=1,
                num_workers=0,
                **_mel_multi_nodes(),
            )
            dm.setup()
            self.assertEqual(dm.num_classes, 2)
            batch = next(iter(dm.train_dataloader()))
            self.assertEqual(tuple(batch.cls.shape), (1, 2))
            self.assertEqual(float(batch.cls.sum()), 2.0)  # the shared item carries both labels


def _wave_nodes() -> dict:
    nodes = _mel_nodes()
    del nodes["mel"], nodes["aug"]
    nodes["audio_stack"] = {
        "_target_": "teia.node.data.collate.tensor.Stack",
        "in": "item.waveform",
        "out": "batch.waveform",
    }
    return nodes


@unittest.skipUnless(_HAS_TORCHAUDIO, "torchaudio/soundfile not installed")
class AudioWaveformGraphTest(unittest.TestCase):
    def test_waveform_field_resolves_through_stack_collate(self) -> None:
        with _tmp_dir() as root:
            _build_class_tree(root)
            dm = TeiaDataModule(
                data_root=str(root),
                batch_size=2,
                num_workers=0,
                **_wave_nodes(),
            )
            dm.setup()
            batch = next(iter(dm.train_dataloader()))
            self.assertEqual(tuple(batch.waveform.shape), (2, 1, 8000))  # 0.5s @ 16kHz


def _tmp_dir():
    import tempfile

    class _Ctx:
        def __enter__(self_inner):
            self_inner._tmp = tempfile.TemporaryDirectory()
            return Path(self_inner._tmp.name)

        def __exit__(self_inner, *exc):
            self_inner._tmp.cleanup()

    return _Ctx()


if __name__ == "__main__":
    unittest.main()
