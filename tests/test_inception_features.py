"""InceptionPool3: pinned-weight verification and pool3 output shape (extra ``fid``)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("pytorch_fid")

from teia.node.eval.metric.unpaired import inception  # noqa: E402


def test_corrupted_cached_weights_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / Path(inception.FID_WEIGHTS_URL).name).write_bytes(b"not the weights")
    with pytest.raises(ValueError, match="sha256"):
        inception.ensure_weights()


@pytest.mark.skipif(not (Path(torch.hub.get_dir()) / "checkpoints" / Path(inception.FID_WEIGHTS_URL).name).exists() and not os.environ.get("TEIA_NETWORK_TESTS"), reason="FID weights not cached; set TEIA_NETWORK_TESTS=1 to download")
def test_pool3_features_shape():
    extractor = inception.InceptionPool3().eval()
    with torch.no_grad():
        features = extractor(torch.rand(2, 3, 32, 32))
    assert features.shape == (2, 2048)
    assert inception.InceptionPool3.identity.endswith(inception.SHA256)
