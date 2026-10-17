from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from teia.node.net.many_to_one.agnostic.node_pack import NodePack
from teia.node.net.many_to_one.agnostic.node_unpack import NodeUnpack
from teia.node.net.many_to_one.graph.graph_encoder import GraphEncoder, shortest_path

D = 16


def board(n: int, generator: torch.Generator):
    nodes = torch.randn(n, D, generator=generator)
    adj = (torch.rand(n, n, generator=generator) > 0.6).float()
    return nodes, ((adj + adj.T + torch.eye(n)) > 0).float()


def pad(nodes, adj, size: int):
    n = nodes.shape[0]
    out_nodes, out_adj = torch.zeros(size, D), torch.zeros(size, size)
    out_nodes[:n], out_adj[:n, :n] = nodes, adj
    mask = torch.zeros(size, dtype=torch.bool)
    mask[:n] = True
    return out_nodes, out_adj, mask


LEAVES = sorted((Path(__file__).parents[1] / "src/teia/conf/node/net/encoder").glob("graph_*.yaml"))


def leaf(path: Path, **overrides):
    cfg = {k: v for k, v in yaml.safe_load(path.read_text()).items() if k != "_target_"}
    return GraphEncoder(in_shape=[(0, D), (0, 0), (0,)], **{**cfg, "heads": 2, "dropout": 0.0, **overrides}).eval()


@pytest.mark.parametrize("path", LEAVES, ids=lambda p: p.stem)
def test_padding_invariance(path):
    encoder = leaf(path)
    generator = torch.Generator().manual_seed(0)
    nodes, adj = board(7, generator)
    outs = []
    for size in (7, 20):
        n, a, m = pad(nodes, adj, size)
        outs.append(encoder(n[None], a[None], m[None])[0, :7])
    assert torch.allclose(outs[0], outs[1], atol=1e-5)


@pytest.mark.parametrize("path", LEAVES, ids=lambda p: p.stem)
def test_padded_rows_are_zero_and_finite(path):
    encoder = leaf(path)
    nodes, adj = board(4, torch.Generator().manual_seed(1))
    n, a, m = pad(nodes, adj, 9)
    out = encoder(n[None], a[None], m[None])
    assert torch.isfinite(out).all() and out[0, 4:].abs().sum() == 0


@pytest.mark.parametrize("path", LEAVES, ids=lambda p: p.stem)
def test_permutation_equivariance(path):
    encoder = leaf(path)
    nodes, adj = board(6, torch.Generator().manual_seed(2))
    perm = torch.randperm(6)
    n, a, m = pad(nodes, adj, 6)
    out = encoder(n[None], a[None], m[None])[0]
    n2, a2, m2 = pad(nodes[perm], adj[perm][:, perm], 6)
    assert torch.allclose(encoder(n2[None], a2[None], m2[None])[0], out[perm], atol=1e-5)


@pytest.mark.parametrize("path", [p for p in LEAVES if p.stem != "graph_graphormer"], ids=lambda p: p.stem)
def test_graph_information_flows_only_along_edges(path):
    encoder = leaf(path, num_layers=1)
    nodes = torch.randn(1, 3, D)
    adj = torch.eye(3)[None]
    adj[0, 0, 1] = adj[0, 1, 0] = 1
    mask = torch.ones(1, 3, dtype=torch.bool)
    base = encoder(nodes, adj, mask)
    moved = nodes.clone()
    moved[0, 2] += 5
    assert torch.allclose(encoder(moved, adj, mask)[0, :2], base[0, :2], atol=1e-6)


def test_pack_unpack_round_trip():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    rows = torch.arange(5.0).unsqueeze(-1).repeat(1, 4)
    pack = NodePack(in_shape=[(4,), (3, 4), (3,)], box_dim=2)
    out = pack(rows, torch.zeros(2, 3, 4), mask)
    assert out.shape == (2, 3, 6) and out[0, 2].abs().sum() == 0
    assert torch.equal(out[..., 0][mask], rows[:, 0])
    assert torch.equal(NodeUnpack(in_shape=[(3, 4), (3,)])(out, mask), out[mask])


def test_shortest_path_caps_far_and_unreachable_pairs():
    chain = torch.zeros(1, 4, 4, dtype=torch.bool)
    chain[0, 0, 1] = chain[0, 1, 0] = chain[0, 1, 2] = chain[0, 2, 1] = True
    assert shortest_path(chain, 1)[0, 0].tolist() == [0, 1, 2, 2]
    assert shortest_path(chain, 3)[0, 0].tolist() == [0, 1, 2, 4]


def test_unknown_choice_fails_at_build():
    with pytest.raises(ValueError, match="aggregate"):
        GraphEncoder(in_shape=[(0, D), (0, 0), (0,)], aggregate="max")


class PreviousBlock(nn.Module):
    """The pre-LN block of the graph transformer that boards v5 trained with, restricted to attention over edges."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.edge_bias = nn.Parameter(torch.zeros(heads))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, h, edges, blocked, node_mask):
        b, n, d = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        mask = (edges * self.edge_bias.view(1, -1, 1, 1)).masked_fill(blocked | ~node_mask[:, None, None, :], float("-inf"))
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        h = h + self.out(attended.transpose(1, 2).reshape(b, n, d))
        h = h + self.ffn(self.norm2(h))
        return h * node_mask.unsqueeze(-1)


def test_graph_transformer_leaf_matches_the_previous_graph_transformer():
    encoder = leaf(next(p for p in LEAVES if p.stem == "graph_transformer"))
    blocks = nn.ModuleList(PreviousBlock(D, 2) for _ in encoder.layers).eval()
    for block, layer in zip(blocks, encoder.layers):
        block.qkv.load_state_dict(layer.aggregate.qkv.state_dict())
        block.out.load_state_dict(layer.aggregate.out.state_dict())
        block.norm1.load_state_dict(layer.combine.norm1.state_dict())
        block.norm2.load_state_dict(layer.combine.norm2.state_dict())
        block.ffn.load_state_dict(layer.combine.ffn.state_dict())

    generator = torch.Generator().manual_seed(3)
    boards = [pad(*board(n, generator), 11) for n in (11, 6, 3)]
    nodes, adj, mask = (torch.stack(x) for x in zip(*boards))
    edges = (adj > 0).unsqueeze(1)
    blocked = ~edges & mask[:, None, :, None]
    h = nodes * mask.unsqueeze(-1)
    for block in blocks:
        h = block(h, edges.float(), blocked, mask)

    assert torch.allclose(encoder(nodes, adj, mask), h, atol=1e-5)


def test_norm_order_changes_the_output():
    path = next(p for p in LEAVES if p.stem == "graph_transformer")
    nodes, adj = board(6, torch.Generator().manual_seed(4))
    n, a, m = pad(nodes, adj, 6)
    pre, post = leaf(path), leaf(path, norm_order="post")
    post.load_state_dict(pre.state_dict())
    assert not torch.allclose(pre(n[None], a[None], m[None]), post(n[None], a[None], m[None]))


def test_unknown_norm_order_fails_at_build():
    with pytest.raises(ValueError, match="norm_order"):
        GraphEncoder(in_shape=[(0, D), (0, 0), (0,)], norm_order="mid")


def test_sage_normalize_fixes_the_output_norm_and_off_follows_the_input():
    path = next(p for p in LEAVES if p.stem == "graph_sage")
    nodes, adj = board(6, torch.Generator().manual_seed(5))
    n, a, m = pad(nodes, adj, 6)
    torch.manual_seed(0)
    on, off = leaf(path), leaf(path, normalize=False)
    torch.testing.assert_close(on(n[None], a[None], m[None]).norm(dim=-1), torch.ones(1, 6), atol=1e-5, rtol=0)
    small, large = off(n[None], a[None], m[None]).norm(dim=-1), off(10 * n[None], a[None], m[None]).norm(dim=-1)
    assert (large > 5 * small).all()


def test_normalize_false_needs_the_concat_combine():
    with pytest.raises(ValueError, match="normalize"):
        GraphEncoder(in_shape=[(0, D), (0, 0), (0,)], combine="residual", normalize=False)
