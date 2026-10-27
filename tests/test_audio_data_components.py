"""Unit tests for the audio data components: reader, decode/mel/augment transforms, writer."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import torch

from teia.node.data.reader.audio import AudioFolderReader
from teia.node.data.transform.decode import DecodeAudio
from teia.node.data.transform.stft import MelSpectrogram
from teia.node.data.transform.augment import SpecAugment
from teia.node.data.writer.class_dir import ClassDirWriter

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


def _wire(node, in_key: list[str], out_key: list[str], *, phase: str = "preprocess", stage: str = "item"):
    node.in_key, node.out_key = in_key, out_key
    node.phase, node.stage = phase, stage
    return node


@unittest.skipUnless(_HAS_TORCHAUDIO, "torchaudio not installed")
class AudioFolderReaderTest(unittest.TestCase):
    def test_scans_split_and_class_dirs_single_label(self) -> None:
        with _tmp_class_tree() as root:
            reader = AudioFolderReader(root=str(root))
            reader.data_root, reader.project_dir = None, None
            train = dict(reader.iter_split("train"))
            val = dict(reader.iter_split("val"))
            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 1)
            _, (path, cls) = next(iter(train.items()))
            self.assertTrue(Path(path).exists())
            self.assertIn(cls, {"cat", "dog"})

    def test_multi_label_groups_same_stem_across_class_dirs(self) -> None:
        with _tmp_class_tree(multi=True) as root:
            reader = AudioFolderReader(root=str(root), multi_label=True)
            reader.data_root, reader.project_dir = None, None
            train = dict(reader.iter_split("train"))
            self.assertEqual(len(train), 1)
            _, (path, names) = next(iter(train.items()))
            self.assertEqual(sorted(names), ["cat", "dog"])


@unittest.skipUnless(_HAS_TORCHAUDIO, "torchaudio not installed")
class DecodeAudioTest(unittest.TestCase):
    def test_decodes_resamples_and_pads_to_fixed_window(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            _write_sine(path, sample_rate=8000, seconds=0.5)
            node = _wire(DecodeAudio(sample_rate=16000, duration=2.0), ["item.audio_path"], ["item.waveform"])
            wave = node(str(path))
            self.assertEqual(wave.shape, (1, 32000))

    def test_crops_when_longer_than_duration(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            _write_sine(path, sample_rate=16000, seconds=3.0)
            node = _wire(DecodeAudio(sample_rate=16000, duration=1.0), ["item.audio_path"], ["item.waveform"])
            wave = node(str(path))
            self.assertEqual(wave.shape, (1, 16000))

    def test_bypasses_decoding_for_precomputed_pt(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.pt"
            precomputed = torch.randn(1, 64, 100)
            torch.save(precomputed, path)
            node = _wire(DecodeAudio(sample_rate=16000, duration=1.0), ["item.audio_path"], ["item.waveform"])
            out = node(str(path))
            self.assertTrue(torch.equal(out, precomputed))


@unittest.skipUnless(_HAS_TORCHAUDIO, "torchaudio not installed")
class MelSpectrogramTest(unittest.TestCase):
    def test_produces_grid_shape(self) -> None:
        node = _wire(MelSpectrogram(sample_rate=16000, n_mels=64), ["item.waveform"], ["item.audio"])
        wave = torch.zeros(1, 16000)
        mel = node(wave)
        self.assertEqual(mel.shape[0], 1)
        self.assertEqual(mel.shape[1], 64)
        self.assertGreater(mel.shape[2], 0)


class SpecAugmentTest(unittest.TestCase):
    def test_masks_bands_when_triggered(self) -> None:
        node = _wire(
            SpecAugment(p=1.0, freq_mask_param=16, time_mask_param=20, n_freq_masks=4, n_time_masks=4),
            ["item.audio"],
            ["item.audio"],
            phase="augment",
        )
        # Non-constant values: masking overwrites with the spectrogram's mean, a no-op on a
        # constant tensor, so distinct values are needed to detect a mask actually landed.
        spec = torch.arange(1 * 16 * 20, dtype=torch.float32).reshape(1, 16, 20)
        out = node(spec)
        self.assertEqual(out.shape, spec.shape)
        self.assertFalse(torch.equal(out, spec))

    def test_noop_when_probability_zero(self) -> None:
        node = _wire(SpecAugment(p=0.0), ["item.audio"], ["item.audio"], phase="augment")
        spec = torch.ones(1, 16, 20)
        out = node(spec)
        self.assertTrue(torch.equal(out, spec))


class ClassDirWriterTest(unittest.TestCase):
    def test_copies_files_into_predicted_class_folders(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "src"
            src_dir.mkdir()
            src_files = [src_dir / f"clip{i}.wav" for i in range(2)]
            for f in src_files:
                f.write_bytes(b"RIFF....")

            out_root = Path(tmp) / "out"
            ClassDirWriter()([0, 1], [str(p) for p in src_files], ["cat", "dog"], dst=out_root)

            self.assertTrue((out_root / "cat" / "clip0.wav").exists())
            self.assertTrue((out_root / "dog" / "clip1.wav").exists())


def _tmp_class_tree(*, multi: bool = False):
    import tempfile

    class _Ctx:
        def __enter__(self_inner):
            self_inner._tmp = tempfile.TemporaryDirectory()
            root = Path(self_inner._tmp.name)
            if not multi:
                _write_sine(root / "train" / "cat" / "a.wav")
                _write_sine(root / "train" / "dog" / "b.wav")
                _write_sine(root / "val" / "cat" / "c.wav")
            else:
                _write_sine(root / "train" / "cat" / "a.wav")
                _write_sine(root / "train" / "dog" / "a.wav")
            return root

        def __exit__(self_inner, *exc):
            self_inner._tmp.cleanup()

    return _Ctx()


if __name__ == "__main__":
    unittest.main()
