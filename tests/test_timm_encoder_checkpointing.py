from __future__ import annotations

import pytest
import torch

from teia.node.net.one_to_one.grid.timm import TimmEncoder

MODEL = dict(model_name="vit_tiny_patch16_224", pretrained=False, freeze=False)


def encoder(**kw) -> TimmEncoder:
    return TimmEncoder(in_shape=(3, 224, 224), **MODEL, **kw)


def test_checkpointing_is_off_by_default():
    assert not encoder()._model.grad_checkpointing


def test_checkpointing_changes_neither_outputs_nor_gradients():
    plain, checkpointed = encoder(), encoder(grad_checkpointing=True)
    checkpointed._model.load_state_dict(plain._model.state_dict())
    assert checkpointed._model.grad_checkpointing
    x = torch.randn(2, 3, 224, 224)
    grads = []
    for model in (plain, checkpointed):
        model.train()
        model._model.zero_grad()
        for module in model._model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        out = model(x)
        out.square().mean().backward()
        grads.append((out.detach(), [p.grad.clone() for p in model._model.parameters() if p.grad is not None]))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-5)
    assert len(grads[0][1]) == len(grads[1][1])
    for a, b in zip(grads[0][1], grads[1][1]):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_chunked_encoding_changes_neither_outputs_nor_gradients():
    plain, chunked = encoder(), encoder(chunk_size=3)
    chunked._model.load_state_dict(plain._model.state_dict())
    x = torch.randn(7, 3, 224, 224)
    results = []
    for model in (plain, chunked):
        model.train()
        model._model.zero_grad()
        for module in model._model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        out = model(x)
        out.square().mean().backward()
        results.append((out.detach(), [p.grad.clone() for p in model._model.parameters() if p.grad is not None]))
    assert results[1][0].shape == (7, 192)
    assert torch.allclose(results[0][0], results[1][0], atol=1e-5)
    assert len(results[0][1]) == len(results[1][1])
    for a, b in zip(results[0][1], results[1][1]):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_chunked_encoding_in_eval_is_the_plain_forward():
    plain, chunked = encoder(), encoder(chunk_size=2)
    chunked._model.load_state_dict(plain._model.state_dict())
    x = torch.randn(5, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(plain.eval()(x), chunked.eval()(x), atol=1e-6)


@pytest.mark.parametrize("mode", ["eval", "frozen"])
def test_chunking_bounds_the_batch_the_model_sees_outside_training(mode):
    model = TimmEncoder(in_shape=(3, 224, 224), **{**MODEL, "freeze": mode == "frozen"}, chunk_size=2)
    sizes = []
    plain_encode = model._encode
    model._encode = lambda chunk: sizes.append(len(chunk)) or plain_encode(chunk)
    with torch.no_grad():
        out = model.eval()(torch.randn(5, 3, 224, 224))
    assert sizes == [2, 2, 1] and out.shape == (5, 192)


@pytest.mark.parametrize("kw", [dict(chunk_size=0), dict(chunk_size=4, out_indices=(1,))])
def test_chunk_size_rejects_invalid_use(kw):
    with pytest.raises(ValueError):
        encoder(**kw)
