# %%
import os

from auto_embeds.data import filter_word_pairs, tokenize_word_pairs
from auto_embeds.embed_utils import initialize_embed_and_unembed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pytest
import torch as t
import torch.testing as tt
from transformers import AutoTokenizer

from auto_embeds.data import get_cached_weights
from auto_embeds.embed_utils import (
    initialize_transform_and_optim,
)
from auto_embeds.utils.logging import logger

np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model_weights = get_cached_weights("bigscience/bloom-560m")
d_model = model_weights["W_E"].shape[1]
device = t.device("cuda" if t.cuda.is_available() else "cpu")
embed_module, unembed_module = initialize_embed_and_unembed(
    tokenizer=tokenizer,
    model_weights=model_weights,
    embed_ln_weights="model_weights",
    unembed_ln_weights="model_weights",
    device=device,
)
batch = 10

# %%
en_fr_pairs = [["hospital", "hôpital"], ["electronic", "électronique"]]
filtered_word_pairs = filter_word_pairs(
    tokenizer,
    en_fr_pairs,
    discard_if_same=False,
    capture_diff_case=True,
    min_length=3,
    space_configurations=[
        {"en": "space", "fr": "space"},
    ],
)
en_toks, fr_toks, _, _ = tokenize_word_pairs(
    tokenizer=tokenizer, word_pairs=filtered_word_pairs
)


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
    transform_module, optim = initialize_transform_and_optim(
        d_model, transformation, optim_kwargs={}
    )
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


def test_identity_transformation():
    identity_transform, _ = initialize_transform_and_optim(
        d_model, "identity", optim_kwargs={}
    )
    data = t.rand((d_model), device=device)
    actual = identity_transform(data)
    # The identity transformation should not change the input tensor
    tt.assert_close(data, actual)
    assert t.allclose(data, actual)


def test_translation_transformation():
    # Test translation transformation by adding a vector of ones to input tensor.
    translation_transform, _ = initialize_transform_and_optim(
        d_model, "translation", optim_kwargs={}
    )
    translation_transform.translation.data = t.ones(d_model, device=device)
    data = t.randn((batch, d_model), device=device)
    expected = data + t.ones_like(data)
    actual = translation_transform(data)
    assert t.allclose(expected, actual)
    tt.assert_close(actual, expected)


def test_linear_map_transformation():
    # Test the linear map transformation by doubling the input tensor values.
    linear_map_transformation, _ = initialize_transform_and_optim(
        d_model, "linear_map", optim_kwargs={}
    )
    linear_map_transformation.linear.weight.data = t.eye(d_model, device=device) * 2
    data = t.rand((batch, d_model), device=device)
    expected = data * 2
    actual = linear_map_transformation(data)
    tt.assert_close(actual, expected)


def test_uncentered_linear_map_transformation():
    # Test the uncentered linear map transformation by adding a tensor of ones to the
    # input, doubling it and taking away a tensor of ones and seeing if the results
    # match the module.
    uncentered_linear_map_transformation, _ = initialize_transform_and_optim(
        d_model, "uncentered_linear_map", optim_kwargs={}
    )
    # Setting the linear transformation to double the input and translation to ones
    uncentered_linear_map_transformation.linear_map.weight.data = (
        t.eye(d_model, device=device) * 2
    )
    uncentered_linear_map_transformation.center.data = t.ones(d_model, device=device)
    data = t.rand((batch, d_model), device=device)
    expected = (data + t.ones(d_model, device=device)) * 2 - t.ones(
        d_model, device=device
    )
    actual = uncentered_linear_map_transformation(data)
    tt.assert_close(expected, actual)


def test_biased_uncentered_linear_map_transformation():
    # Test the biased linear map transformation by adding a tensor of ones to the
    # input, linear mapping this and taking away a tensor of ones and seeing if the
    # results match the module. As this has a bias, the linear map consists of a
    # doubling and adding of ones
    biased_uncentered_linear_map_transformation, _ = initialize_transform_and_optim(
        d_model, "biased_uncentered_linear_map", optim_kwargs={}
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
    data = t.rand((batch, d_model), device=device)
    expected = (
        (data + t.ones(d_model, device=device)) * 2 + t.ones(d_model, device=device)
    ) - t.ones(d_model, device=device)
    actual = biased_uncentered_linear_map_transformation(data)
    tt.assert_close(expected, actual)


def test_rotation_transformation():
    # Initialize rotation as identity for testing
    rotation_transformation, _ = initialize_transform_and_optim(
        d_model, "rotation", optim_kwargs={}
    )
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
    transform, _ = initialize_transform_and_optim(d_model, "rotation", optim_kwargs={})
    data = t.arange(d_model, dtype=t.float, device=device).unsqueeze(0)
    expected = t.norm(data, p=2, dim=-1)
    actual = t.norm(transform(data), p=2, dim=-1)
    tt.assert_close(actual, expected)


def test_filter_word_pairs():
    input_word_pairs = [["hospital", "hôpital"], ["electronic", "électronique"]]
    expected_filtered_word_pairs = [
        [" hospital", " hôpital"],
        [" Hospital", " hôpital"],
        [" electronic", " électronique"],
        [" Electronic", " électronique"],
    ]
    filtered_word_pairs = filter_word_pairs(
        tokenizer,
        input_word_pairs,
        discard_if_same=False,
        capture_diff_case=True,
        min_length=3,
        space_configurations=[
            {"en": "space", "fr": "space"},
        ],
    )
    logger.info(filtered_word_pairs)
    assert filtered_word_pairs == expected_filtered_word_pairs


@pytest.mark.slow
def test_train_transform():

    en_fr_pairs = [
        ["hospital", "hôpital"],
        ["electronic", "électronique"],
        ["trajectory", "trajectoire"],
        ["commissioner", "commissaire"],
    ]

    filtered_word_pairs = filter_word_pairs(
        tokenizer,
        en_fr_pairs,
        discard_if_same=False,
        capture_diff_case=True,
        min_length=3,
        space_configurations=[
            {"en": "space", "fr": "space"},
        ],
    )

    en_toks, fr_toks, _, _ = tokenize_word_pairs(tokenizer, filtered_word_pairs)

    en_embeds = embed_module(en_toks)
    print(en_embeds.shape)
    fr_embeds = embed_module(fr_toks)

    train_dataset = TensorDataset(en_embeds[:2], fr_embeds[:2])
    test_dataset = TensorDataset(en_embeds[2:], fr_embeds[2:])
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)

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

        transform, optim = initialize_transform_and_optim(
            d_model,
            transformation=transformation_name,
            optim_kwargs={},
        )

        loss_module = initialize_loss("cosine_similarity")

        if optim is not None:
            transform, loss_history = train_transform(
                tokenizer=tokenizer,
                train_loader=train_loader,
                test_loader=test_loader,
                transform=transform,
                optim=optim,
                unembed_module=unembed_module,
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


@t.no_grad()
def test_initialize_embed_and_unembed_each_give_different_results():
    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    model_weights = get_cached_weights("bigscience/bloom-560m")
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    embed_module_model, unembed_module_model = initialize_embed_and_unembed(
        tokenizer=tokenizer,
        model_weights=model_weights,
        embed_ln_weights="model_weights",
        unembed_ln_weights="model_weights",
        device=device,
    )
    embed_module_default, unembed_module_default = initialize_embed_and_unembed(
        tokenizer=tokenizer,
        model_weights=model_weights,
        embed_ln_weights="default_weights",
        unembed_ln_weights="default_weights",
        device=device,
    )

    embed_result_model = embed_module_model(en_toks)
    embed_result_default = embed_module_default(en_toks)
    unembed_result_model = unembed_module_model(embed_result_model)
    unembed_result_default = unembed_module_default(embed_result_default)

    assert not t.equal(embed_result_model, embed_result_default)
    assert not t.equal(unembed_result_model, unembed_result_default)
