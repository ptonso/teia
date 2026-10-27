from __future__ import annotations

from pathlib import Path

import pytest
import torch

from teia.node.data.collate.graph import FlatNodeClasses, NodeBatchIndex, NodeMask, PadAdjacency, PadNodeRows, StackNodeImages
from teia.node.data.reader.extension.base import ReaderExtension
from teia.node.data.reader.tu import TuReader
from teia.node.data.transform.adjacency import EdgeAdjacency, KnnAdjacency


def write_tu(folder: Path, prefix: str = "TOY") -> Path:
    """Two graphs: nodes 1-3 (path 1-2, 2-3) and nodes 4-5 (edge 4-5), 2 attributes and a label per node."""
    folder.mkdir(parents=True)
    (folder / f"{prefix}_A.txt").write_text("1, 2\n2, 1\n2, 3\n3, 2\n4, 5\n5, 4\n")
    (folder / f"{prefix}_graph_indicator.txt").write_text("1\n1\n1\n2\n2\n")
    (folder / f"{prefix}_node_labels.txt").write_text("0\n1\n0\n1\n1\n")
    (folder / f"{prefix}_node_attributes.txt").write_text("0.0, 1.0\n1.0, 2.0\n2.0, 3.0\n3.0, 4.0\n4.0, 5.0\n")
    return folder


def read(root: Path, split: str = "train") -> dict:
    return dict(TuReader(root=str(root)).iter_split(split))


def test_tu_reader_emits_one_item_per_graph(tmp_path):
    write_tu(tmp_path / "train")
    items = read(tmp_path)
    assert sorted(items) == [1, 2]
    feat, edges, labels = items[1]
    assert feat.shape == (3, 2) and feat.dtype == torch.float32 and labels == ["0", "1", "0"]
    assert sorted(edges.tolist()) == [[0, 1], [1, 0], [1, 2], [2, 1]]
    _, edges2, labels2 = items[2]
    assert sorted(edges2.tolist()) == [[0, 1], [1, 0]] and labels2 == ["1", "1"]


def test_tu_reader_extension_appends_fields_keyed_by_node_id(tmp_path):
    class Ids(ReaderExtension):
        def extend(self, key, record, source):
            return {**record, "ids": record["node_ids"], "source": source.name}

    write_tu(tmp_path / "train")
    items = dict(TuReader(root=str(tmp_path), extension=Ids()).iter_split("train"))
    assert items[1][3] == [1, 2, 3] and items[2][3] == [4, 5] and items[1][4] == "TOY_A.txt" and len(items[1]) == 5


def test_tu_reader_missing_split_yields_nothing(tmp_path):
    write_tu(tmp_path / "train")
    assert read(tmp_path, "test") == {}


def test_tu_reader_missing_file_raises(tmp_path):
    folder = write_tu(tmp_path / "train")
    (folder / "TOY_node_labels.txt").unlink()
    with pytest.raises(FileNotFoundError, match="node_labels"):
        read(tmp_path)


def test_tu_reader_needs_exactly_one_dataset(tmp_path):
    write_tu(tmp_path / "train")
    (tmp_path / "train" / "OTHER_graph_indicator.txt").write_text("")
    with pytest.raises(ValueError, match="exactly one"):
        read(tmp_path)


def test_tu_reader_keeps_named_labels(tmp_path):
    folder = write_tu(tmp_path / "train")
    (folder / "TOY_node_labels.txt").write_text("present\nmissing\npresent\nbackground\nmissing\n")
    items = read(tmp_path)
    assert items[1][2] == ["present", "missing", "present"] and items[2][2] == ["background", "missing"]


def test_tu_reader_selects_an_edge_variant(tmp_path):
    folder = write_tu(tmp_path / "train")
    (folder / "TOY_A_knn1.txt").write_text("1, 3\n3, 1\n")
    items = dict(TuReader(root=str(tmp_path), edges="A_knn1").iter_split("train"))
    assert sorted(items[1][1].tolist()) == [[0, 2], [2, 0]] and items[2][1].shape == (0, 2)
    assert sorted(read(tmp_path)[1][1].tolist()) == [[0, 1], [1, 0], [1, 2], [2, 1]]


def test_tu_reader_rejects_edge_across_graphs(tmp_path):
    folder = write_tu(tmp_path / "train")
    (folder / "TOY_A.txt").write_text("1, 4\n")
    with pytest.raises(ValueError, match="two different graphs"):
        read(tmp_path)


def test_edge_adjacency_symmetric_with_self_loops():
    adj = EdgeAdjacency()(torch.tensor([[0, 2]]), torch.zeros(3, 4))
    assert adj.tolist() == [[True, False, True], [False, True, False], [True, False, True]]


def test_knn_adjacency_symmetric_with_self_loops():
    boxes = torch.tensor([[0.0, 0, 1, 1], [2, 0, 1, 1], [50, 0, 1, 1], [52, 0, 1, 1]])
    adj = KnnAdjacency(k=1)(boxes)
    assert torch.equal(adj, adj.T) and adj.diagonal().all()
    assert adj[0, 1] and adj[2, 3] and not adj[0, 2]


def test_collates_pad_to_batch_max_and_align_rows():
    rows = [torch.rand(3, 7), torch.rand(5, 7)]
    adj = [torch.eye(3, dtype=torch.bool), torch.eye(5, dtype=torch.bool)]
    assert PadNodeRows()(rows).shape == (2, 5, 7)
    padded = PadAdjacency()(adj)
    assert padded.shape == (2, 5, 5) and padded[0, 3:].sum() == 0 and padded[0, :, 3:].sum() == 0
    mask = NodeMask()(rows)
    assert mask.sum(1).tolist() == [3, 5]
    assert NodeBatchIndex()(rows).tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert StackNodeImages()([torch.zeros(3, 3, 8, 8), torch.zeros(5, 3, 8, 8)]).shape == (8, 3, 8, 8)
    cls = FlatNodeClasses()([torch.arange(3), torch.arange(5)])
    assert cls.shape[0] == int(mask.sum())
