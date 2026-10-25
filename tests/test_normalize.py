import pytest
import torch

from teia.node.data.transform.normalize import Normalize


def _legacy_image(value, *, pixel_norm):
    x = value.to(torch.float32) / 255.0
    if pixel_norm == "signed":
        x = x * 2.0 - 1.0
    if pixel_norm == "imagenet":
        m = torch.tensor((0.485, 0.456, 0.406)).reshape(-1, 1, 1)
        s = torch.tensor((0.229, 0.224, 0.225)).reshape(-1, 1, 1)
        x = (x - m) / s
    return x


@pytest.mark.parametrize("pixel_norm", ["unit", "signed", "imagenet"])
def test_pixel_norm_presets_match_the_former_image_kernel(pixel_norm):
    batch = torch.randint(0, 256, (4, 3, 8, 8), dtype=torch.uint8)
    out = Normalize(pixel_norm=pixel_norm)(batch)
    torch.testing.assert_close(out, _legacy_image(batch, pixel_norm=pixel_norm), atol=1e-6, rtol=0)


def test_fixed_stats_broadcast_along_the_axis_and_clip():
    out = Normalize(axis=1, mean=[1.0, 2.0], std=[1.0, 2.0], clip=1.0)(torch.tensor([[[3.0], [10.0]]]))
    assert out.flatten().tolist() == [1.0, 1.0]


def test_fit_uses_the_train_stream_and_persists_through_state():
    node = Normalize(stats="fit")
    node.fit([torch.tensor([1.0, 10.0]), torch.tensor([3.0, 30.0])])
    out = node(torch.tensor([[2.0, 20.0]]))
    assert out.abs().max() < 1e-6
    clone = Normalize(stats="fit")
    clone.load_state(node.state())
    assert clone.params() == node.params()


def test_running_stats_update_only_while_training_and_match_batch_moments():
    node = Normalize(stats="running")
    x = torch.randn(64, 3) * 5 + 2
    node(x[:32])
    node(x[32:])
    mean = torch.tensor(node.params()["mean"], dtype=torch.float64)
    torch.testing.assert_close(mean, x.double().mean(0))
    node.training = False
    node(x + 100)
    assert node.params()["mean"] == mean.tolist()


def test_unfitted_running_stats_pass_through_and_export_params_are_plain():
    node = Normalize(stats="running")
    node.training = False
    x = torch.randn(2, 3)
    torch.testing.assert_close(node(x), x)
    assert node.params()["mean"] is None


def test_literal_stats_are_rejected_when_the_data_derives_them():
    with pytest.raises(ValueError, match="derives"):
        Normalize(stats="fit", mean=0.0, std=1.0)
    with pytest.raises(ValueError, match="stats"):
        Normalize(stats="sometimes")
