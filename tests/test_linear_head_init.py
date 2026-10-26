import torch

from teia.node.net.one_to_one.agnostic.linear import LinearHead


def test_init_gain_sets_an_orthogonal_weight_with_that_gain_and_a_zero_bias():
    head = LinearHead((8,), (3,), init_gain=0.01)
    weight = head.linear.weight
    assert torch.allclose(weight @ weight.T, 1e-4 * torch.eye(3), atol=1e-6)
    assert head.linear.bias.abs().max() == 0


def test_without_init_gain_the_default_initialisation_stays():
    assert LinearHead((8,), (3,)).linear.bias.abs().max() > 0
