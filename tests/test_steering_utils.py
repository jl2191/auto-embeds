# %%
import os
from auto_embeds.metrics import calc_cos_sim_acc, evaluate_accuracy

from auto_embeds.data import filter_word_pairs, tokenize_word_pairs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pytest
import torch as t
import torch.testing as tt
from torch.utils.data import DataLoader, TensorDataset, random_split
import transformer_lens as tl

from auto_embeds.embed_utils import (
    initialize_transform_and_optim,
    initialize_loss,
    train_transform,
)

np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
d_model = model.cfg.d_model
device = model.cfg.device
batch = 10



# %%
@pytest.mark.parametrize(
    "transformation,expected_optim_type",
    [
        ("identity", None),
        ("translation", t.optim.Adam),
        ("linear_map", t.optim.Adam),
        ("biased_linear_map", t.optim.Adam),
        ("uncentered_linear_map", t.optim.Adam),
        ("biased_uncentered_linear_map", t.optim.Adam),
        ("rotation", t.optim.Adam),
        ("biased_rotation", t.optim.Adam),
        ("uncentered_rotation", t.optim.Adam),
    ],
)
def test_initialize_transform_and_optim_types_general(
    transformation, expected_optim_type
):
    transform_module, optim = initialize_transform_and_optim(d_model, transformation)
    assert_msg = f"Failed to initialize transform for {transformation}"
    assert transform_module is not None, assert_msg

    if expected_optim_type is None:
        assert_msg = f"Expected no optimizer for {transformation}, got {type(optim)}"
        assert optim is None, assert_msg
    else:
        assert_msg = (
            f"Expected {expected_optim_type} for {transformation}, "
            f"got {type(optim)}"
        )
        assert isinstance(optim, expected_optim_type), assert_msg


def test_initialize_transform_and_optim_types_mean_diff():
    mean_diff = t.rand(d_model) - t.rand(d_model)
    transform_module, optim = initialize_transform_and_optim(
        d_model, "mean_translation", mean_diff=mean_diff
    )
    assert_msg = "Failed to initialize transform for mean_translation"
    assert transform_module is not None, assert_msg
    assert optim is None, "Expected no optimizer for mean_translation"


def test_identity_transformation():
    identity_transform, _ = initialize_transform_and_optim(d_model, "identity")
    input = t.rand((d_model), device=device)
    actual = identity_transform(input)
    # The identity transformation should not change the input tensor
    tt.assert_close(input, actual)
    assert t.allclose(input, actual)


def test_translation_transformation():
    translation_transform, _ = initialize_transform_and_optim(d_model, "translation")
    # Test translation transformation by adding a vector of ones to input tensor.
    translation_transform.translation.data = t.ones(d_model, device=device)
    input = t.randn((batch, d_model), device=device)
    expected = input + t.ones_like(input)
    actual = translation_transform(input)
    assert t.allclose(expected, actual)
    tt.assert_close(actual, expected)


def test_linear_map_transformation():
    linear_map_transformation, _ = initialize_transform_and_optim(d_model, "linear_map")
    # Test the linear map transformation by doubling the input tensor values.
    linear_map_transformation.weight.data = t.eye(d_model, device=device) * 2
    input = t.rand((batch, d_model), device=device)
    expected = input * 2
    actual = linear_map_transformation(input)
    tt.assert_close(actual, expected)


def test_uncentered_linear_map_transformation():
    # Test the uncentered linear map transformation by adding a tensor of ones to the
    # input, doubling it and taking away a tensor of ones and seeing if the results
    # match the module.
    uncentered_linear_map_transformation, _ = initialize_transform_and_optim(
        d_model, "uncentered_linear_map"
    )
    # Setting the linear transformation to double the input and translation to ones
    uncentered_linear_map_transformation.linear_map.weight.data = (
        t.eye(d_model, device=device) * 2
    )
    uncentered_linear_map_transformation.center.data = t.ones(d_model, device=device)
    input = t.rand((batch, d_model), device=device)
    expected = (input + t.ones(d_model, device=device)) * 2 - t.ones(
        d_model, device=device
    )
    actual = uncentered_linear_map_transformation(input)
    tt.assert_close(expected, actual)


def test_biased_uncentered_linear_map_transformation():
    # Test the biased linear map transformation by adding a tensor of ones to the
    # input, linear mapping this and taking away a tensor of ones and seeing if the
    # results match the module. As this has a bias, the linear map consists of a
    # doubling and adding of ones
    biased_uncentered_linear_map_transformation, _ = initialize_transform_and_optim(
        d_model, "biased_uncentered_linear_map"
    )
    # Setting the linear transformation to double the input and center to ones
    biased_uncentered_linear_map_transformation.linear_map.weight.data = (
        t.eye(d_model, device=device) * 2
    )
    biased_uncentered_linear_map_transformation.linear_map.bias.data = t.ones(
        d_model, device=device
    )
    biased_uncentered_linear_map_transformation.center.data = t.ones(
        d_model, device=device
    )
    input = t.rand((batch, d_model), device=device)
    expected = (
        (input + t.ones(d_model, device=device)) * 2 + t.ones(d_model, device=device)
    ) - t.ones(d_model, device=device)
    actual = biased_uncentered_linear_map_transformation(input)
    tt.assert_close(expected, actual)


def test_rotation_transformation():
    rotation_transformation, _ = initialize_transform_and_optim(d_model, "rotation")
    # Initialize rotation as identity for testing
    rotation_transformation.rotation.weight = t.eye(d_model, device=device)
    data = t.rand((batch, d_model), device=device)
    # Since rotation is identity, the expected output is the input itself
    expected = data.detach().clone()
    actual = rotation_transformation(data)

    tt.assert_close(actual, expected)

    # Test rotation transformation with a 45-degree rotation matrix
    angle = t.tensor(t.pi / 4, dtype=t.float, device=device)
    sin, cos = t.sin(angle), t.cos(angle)
    rotation_matrix = t.tensor([[cos, -sin], [sin, cos]], device=device)
    # Extend rotation_matrix to match d_model dimensions if necessary
    if d_model > 2:
        identity_extension = t.eye(d_model - 2, device=device)
        rotation_matrix = t.block_diag(rotation_matrix, identity_extension)
    rotation_transformation.rotation.weight = rotation_matrix
    data = t.tensor([[1, 0] + [0] * (d_model - 2)], dtype=t.float, device=device)
    expected = t.tensor(
        [[cos.item(), sin.item()] + [0] * (d_model - 2)], dtype=t.float, device=device
    )
    actual = rotation_transformation(data)
    tt.assert_close(actual, expected)

    # Test rotation transformation with a 90-degree rotation matrix
    rotation_matrix = t.tensor([[0, -1], [1, 0]], device=device)
    # Extend rotation_matrix_90_deg to match d_model dimensions if necessary
    if d_model > 2:
        identity_extension = t.eye(d_model - 2, device=device)
        rotation_matrix = t.block_diag(rotation_matrix, identity_extension)
    rotation_transformation.rotation.weight = rotation_matrix
    data = t.tensor([[1, 0] + [0] * (d_model - 2)], dtype=t.float, device=device)
    expected = t.tensor([[0, 1] + [0] * (d_model - 2)], dtype=t.float, device=device)
    actual = rotation_transformation(data)
    tt.assert_close(actual, expected)

    # Test that norm is preserved during rotation.
    transform, _ = initialize_transform_and_optim(d_model, "rotation")
    data = t.arange(d_model, dtype=t.float, device=device).unsqueeze(
        0
    )  # Add batch dimension
    expected = t.norm(data, p=2, dim=-1)
    actual = t.norm(transform(data), p=2, dim=-1)
    tt.assert_close(actual, expected)


def test_mean_translation_transformation():
    # Test mean translation transformation by adding a mean difference vector to input.
    mean_diff = t.rand(d_model, device=device)
    mean_translation_transform, _ = initialize_transform_and_optim(
        d_model,
        "mean_translation",
        mean_diff=mean_diff,
    )
    input = t.rand((batch, d_model), device=device)
    expected = input + mean_diff
    actual = mean_translation_transform(input)
    tt.assert_close(actual, expected)


def test_tokenize_texts():
    en_fr_pairs = [["hospital", "hôpital"], ["electronic", "électronique"]]
    filtered_word_pairs = filter_word_pairs(
        model,
        en_fr_pairs,
        discard_if_same=False,
        capture_diff_case=True,
        min_length=3,
        capture_space=True,
        capture_no_space=True,
    )
    actual_en_toks, actual_fr_toks, _, _ = tokenize_word_pairs(
        model, filtered_word_pairs
    )

    expected_en_toks = [" hospital", " Hospital", " electronic", " Electronic"]
    expected_fr_toks = [" hôpital", " hôpital", " électronique", " électronique"]

    assert model.to_string(actual_en_toks) == expected_en_toks
    assert model.to_string(actual_fr_toks) == expected_fr_toks


def test_train_transform():

    device = model.cfg.device
    d_model = model.cfg.d_model
    n_toks = model.cfg.d_vocab_out

    en_fr_pairs = [
        ["hospital", "hôpital"],
        ["electronic", "électronique"],
        ["trajectory", "trajectoire"],
        ["commissioner", "commissaire"],
    ]

    filtered_word_pairs = filter_word_pairs(
        model,
        en_fr_pairs,
        discard_if_same=False,
        capture_diff_case=True,
        min_length=3,
        capture_space=True,
        capture_no_space=True,
    )
    print(filtered_word_pairs)

    en_toks, fr_toks, _, _ = tokenize_word_pairs(model, filtered_word_pairs)

    print(en_toks.shape)

    en_embeds = model.embed.W_E[en_toks].detach().clone()
    fr_embeds = model.embed.W_E[fr_toks].detach().clone()

    train_dataset = TensorDataset(en_embeds[:2], fr_embeds[:2])
    test_dataset = TensorDataset(en_embeds[2:], fr_embeds[2:])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    transformation_names = [
        "identity",
        # "translation",
        # "linear_map",
        # "biased_linear_map",
        # "uncentered_linear_map",
        # "biased_uncentered_linear_map",
        "rotation",
        # "biased_rotation",
        # "uncentered_rotation",
    ]

    trained_transforms = {}
    for transformation_name in transformation_names:

        transform = None
        optim = None

        transform, optim = initialize_transform_and_optim(
            d_model,
            transformation=transformation_name,
        )

        loss_module = initialize_loss("cosine_similarity")

        if optim is not None:
            transform, loss_history = train_transform(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                transform=transform,
                optim=optim,
                loss_module=loss_module,
                n_epochs=5,
                plot_fig=False,
            )
            trained_transforms[transformation_name] = transform

        if transformation_name == "identity":
            with t.no_grad():
                expected = t.randn(batch, d_model, device=device)
                actual = transform(expected)
                tt.assert_close(actual, expected)

        if transformation_name == "rotation":
            with t.no_grad():
                actual = t.det(transform(t.eye(d_model, device=device)))
                expected = t.tensor(1.0, device=device)
                tt.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

test_train_transform()