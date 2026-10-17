from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from teia.node.data.collate.graph import FlatNodeClasses, NodeMask, PadAdjacency, PadNodeBoxes, StackNodeImages
from teia.node.data.reader.coco import CocoReader
from teia.node.data.reader.extension.coco_graph import CocoGraph
from teia.node.data.reader.extension.labelme_graph import LabelmeGraph
from teia.node.data.reader.labelme import LabelmeReader
from teia.node.data.transform.adjacency import EdgeAdjacency, KnnAdjacency


BOARDS = Path(__file__).parent / "graph-runs" / "fetch_data" / "boards.py"


@pytest.fixture(scope="module")
def boards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("boards")
    subprocess.run([sys.executable, str(BOARDS), "--target-dir", str(root)], check=True)
    return root


def coco(boards, **kw):
    reader = CocoReader(annotations={"train": "coco/train.json"}, root=str(boards), **kw)
    return dict(reader.iter_split("train"))


def test_coco_reader_emits_boxes_and_labels(boards):
    items = coco(boards)
    path, boxes, labels = items["train_00.png"]
    assert path.endswith("images/train_00.png")
    assert boxes.shape == (len(labels), 4) and set(labels) <= {"present", "missing", "background"}


def test_coco_graph_extension_appends_edges(boards):
    items = coco(boards, extension=CocoGraph())
    _, boxes, _, edges = items["train_00.png"]
    assert edges.shape[1] == 2 and int(edges.max()) < boxes.shape[0]


def test_labelme_and_coco_agree_on_edges(boards):
    coco_items = coco(boards, extension=CocoGraph())
    labelme = dict(LabelmeReader(root=str(boards / "labelme"), extension=LabelmeGraph()).iter_split("train"))
    assert torch.equal(labelme["train_00"][3], coco_items["train_00.png"][3])
    assert torch.equal(labelme["train_00"][1], coco_items["train_00.png"][1])


def test_coco_graph_unknown_id_raises(boards, tmp_path):
    graph = json.loads((boards / "graph.json").read_text())
    graph["train_00.png"]["edges"].append({"src": 0, "dst": 10**6})
    (tmp_path / "graph.json").write_text(json.dumps(graph))
    reader = CocoReader(
        annotations={"train": "coco/train.json"}, root=str(boards), extension=CocoGraph(path=str(tmp_path / "graph.json"))
    )
    with pytest.raises(ValueError, match="not a node id"):
        list(reader.iter_split("train"))


def test_coco_graph_missing_image_raises(boards, tmp_path):
    (tmp_path / "graph.json").write_text("{}")
    reader = CocoReader(
        annotations={"train": "coco/train.json"}, root=str(boards), extension=CocoGraph(path=str(tmp_path / "graph.json"))
    )
    with pytest.raises(KeyError, match="train_00.png"):
        list(reader.iter_split("train"))


def test_coco_class_map_rejects_unknown_label(boards):
    with pytest.raises(KeyError):
        coco(boards, class_map={"present": "present"})


def test_edge_adjacency_symmetric_with_self_loops():
    adj = EdgeAdjacency()(torch.tensor([[0, 2]]), torch.zeros(3, 4))
    assert adj.tolist() == [[True, False, True], [False, True, False], [True, False, True]]


def test_knn_adjacency_symmetric_with_self_loops():
    boxes = torch.tensor([[0.0, 0, 1, 1], [2, 0, 1, 1], [50, 0, 1, 1], [52, 0, 1, 1]])
    adj = KnnAdjacency(k=1)(boxes)
    assert torch.equal(adj, adj.T) and adj.diagonal().all()
    assert adj[0, 1] and adj[2, 3] and not adj[0, 2]


def test_collates_pad_to_batch_max_and_align_rows():
    boxes = [torch.rand(3, 4), torch.rand(5, 4)]
    adj = [torch.eye(3, dtype=torch.bool), torch.eye(5, dtype=torch.bool)]
    assert PadNodeBoxes()(boxes).shape == (2, 5, 4)
    padded = PadAdjacency()(adj)
    assert padded.shape == (2, 5, 5) and padded[0, 3:].sum() == 0 and padded[0, :, 3:].sum() == 0
    mask = NodeMask()(boxes)
    assert mask.sum(1).tolist() == [3, 5]
    assert StackNodeImages()([torch.zeros(3, 3, 8, 8), torch.zeros(5, 3, 8, 8)]).shape == (8, 3, 8, 8)
    cls = FlatNodeClasses()([torch.arange(3), torch.arange(5)])
    assert cls.shape[0] == int(mask.sum())
