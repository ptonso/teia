from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from teia.node.data.reader.extension.node_images import NodeImages
from teia.node.data.reader.tu import TuReader
from teia.node.data.transform.decode_node_images import DecodeNodeImages
from teia.task.graph_node_cls import GraphNodeCls
from teia.task.vision_graph_node_cls import VisionGraphNodeCls


def write_graphs(root: Path, images: bool = True) -> Path:
    """Two graphs (nodes 1-3 and 4-5) with a solid-colour image per node."""
    folder = root / "train"
    (folder / "nodes").mkdir(parents=True)
    (folder / "T_A.txt").write_text("1, 2\n2, 3\n4, 5\n")
    (folder / "T_graph_indicator.txt").write_text("1\n1\n1\n2\n2\n")
    (folder / "T_node_labels.txt").write_text("0\n1\n0\n1\n1\n")
    (folder / "T_node_attributes.txt").write_text("0.0\n1.0\n2.0\n3.0\n4.0\n")
    for node_id in range(1, 6 if images else 5):
        Image.new("RGB", (6, 4), (node_id * 40, 0, 0)).save(folder / "nodes" / f"{node_id}.png")
    return root


def read(root: Path) -> dict:
    return dict(TuReader(root=str(root), extension=NodeImages()).iter_split("train"))


def test_node_images_are_listed_per_graph_in_node_order(tmp_path):
    items = read(write_graphs(tmp_path))
    paths = items[1][3]
    assert [Path(p).name for p in paths] == ["1.png", "2.png", "3.png"]
    assert [Path(p).name for p in items[2][3]] == ["4.png", "5.png"]


def test_missing_node_image_raises_naming_the_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="5.png"):
        read(write_graphs(tmp_path, images=False))


def test_decode_stacks_square_uint8_nodes(tmp_path):
    paths = read(write_graphs(tmp_path))[1][3]
    nodes = DecodeNodeImages(size=8)(paths)
    assert nodes.shape == (3, 3, 8, 8) and nodes.dtype == torch.uint8
    assert nodes[:, 0].float().mean(dim=(1, 2)).tolist() == [40.0, 80.0, 120.0]


def test_contract_adds_only_the_node_image():
    assert set(VisionGraphNodeCls.batch) - set(GraphNodeCls.batch) == {"node_image"}
    assert VisionGraphNodeCls.capture == GraphNodeCls.capture


def test_decode_without_size_keeps_stored_pixels_exactly(tmp_path):
    pixels = torch.randint(0, 256, (2, 5, 5, 3), dtype=torch.uint8)
    paths = []
    for i, image in enumerate(pixels):
        Image.fromarray(image.numpy()).save(tmp_path / f"{i}.png")
        paths.append(str(tmp_path / f"{i}.png"))
    assert torch.equal(DecodeNodeImages()(paths), pixels.permute(0, 3, 1, 2))


def test_decode_without_size_rejects_mixed_sizes(tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    Image.new("RGB", (6, 4)).save(tmp_path / "b.png")
    with pytest.raises(ValueError, match="differ in size"):
        DecodeNodeImages()([str(tmp_path / "a.png"), str(tmp_path / "b.png")])
