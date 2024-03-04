import numpy as np
import pytest
import torch as t
import torch.testing as tt
import transformer_lens as tl

from auto_steer.steering_utils import (
    initialize_transform_and_optim,
    tokenize_texts,
)

model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
d_model = model.cfg.d_model
device = model.cfg.device
batch = 10

@pytest.mark.parametrize(
    "transformation,expected_optim_type",
    [
        ("identity", None),
        ("translation", t.optim.Adam),
        ("linear_map", t.optim.Adam),
        ("rotation", t.optim.Adam),
        ("offset_linear_map", t.optim.Adam),
        ("offset_rotation", t.optim.Adam),
        ("uncentered_linear_map", t.optim.Adam),
        ("uncentered_rotation", t.optim.Adam),
    ],
)
def test_initialize_transform_and_optim_types_general(
    transformation, expected_optim_type
):
    transform_module, optim = initialize_transform_and_optim(
        d_model, transformation
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



def test_initialize_transform_and_optim_types_mean_diff():
    mean_diff = t.rand(d_model) - t.rand(d_model)
    transform_module, optim = initialize_transform_and_optim(
        d_model, "mean_translation", mean_diff=mean_diff
    )
    assert_msg = "Failed to initialize transform for mean_translation"
    assert transform_module is not None, assert_msg
    assert optim is None, "Expected no optimizer for mean_translation"

def test_identity_transformation():
    identity_transform, _ = initialize_transform_and_optim(
        d_model, "identity"
    )
    input = t.rand((d_model), device=device)
    actual = identity_transform(input)
    # The identity transformation should not change the input tensor
    tt.assert_close(input, actual)
    assert t.allclose(input, actual)


def test_translation_transformation():
    translation_transform, _ = initialize_transform_and_optim(
        d_model, "translation"
    )
    # Test translation transformation by adding a vector of ones to input tensor.
    translation_transform.translation.data = t.ones(
        d_model, device=device
    )
    input = t.randn((batch, d_model), device=device)
    expected = input + t.ones_like(input)
    actual = translation_transform(input)
    assert t.allclose(expected, actual)
    tt.assert_close(actual, expected)


def test_linear_map_transformation():
    linear_map_transformation, _ = initialize_transform_and_optim(
        d_model, "linear_map"
    )
    # Testing linear map transformation by doubling the input tensor values
    linear_map_transformation.weight.data = (
        t.eye(d_model, device=device) * 2
    )
    input = t.rand((batch, d_model), device=device)
    expected = input * 2
    actual = linear_map_transformation(input)
    tt.assert_close(actual, expected)


def test_rotation_transformation():
    rotation_transformation, _ = initialize_transform_and_optim(
        d_model, "rotation"
    )
    # Initialize rotation as identity for testing
    rotation_transformation.rotation.weight.data = t.eye(
        d_model, device=device
    )
    data = t.rand((batch, d_model), device=device)
    expected = data  # Since rotation is identity, expected output is the input itself
    actual = rotation_transformation(data)
    tt.assert_close(actual, expected)

    # Test rotation transformation with a 45-degree rotation matrix
    angle = t.tensor(t.pi / 4, dtype=t.float)
    sin, cos = t.sin(angle), t.cos(angle)
    rotation_matrix = t.tensor([[cos, -sin], [sin, cos]])
    # Extend rotation_matrix to match d_model dimensions if necessary
    if d_model > 2:
        identity_extension = t.eye(d_model - 2)
        rotation_matrix = t.block_diag(rotation_matrix, identity_extension)
    rotation_transformation.rotation.weight.data = rotation_matrix
    data = t.tensor([[1, 0] + [0] * (d_model - 2)], dtype=t.float)
    expected = t.tensor([[cos.item(), sin.item()] + [0] * (d_model - 2)], dtype=t.float)
    actual = rotation_transformation(data)
    tt.assert_close(actual, expected)

    # Test rotation transformation with a 90-degree rotation matrix
    rotation_matrix= t.tensor([[0, -1], [1, 0]])
    # Extend rotation_matrix_90_deg to match d_model dimensions if necessary
    if d_model > 2:
        identity_extension = t.eye(d_model - 2)
        rotation_matrix = t.block_diag(rotation_matrix, identity_extension)
    rotation_transformation.rotation.weight.data = rotation_matrix
    data = t.tensor([[1, 0] + [0] * (d_model - 2)], dtype=t.float)
    expected = t.tensor([[0, 1] + [0] * (d_model - 2)], dtype=t.float)
    actual = rotation_transformation(data)
    tt.assert_close(actual, expected)

    # Test that norm is preserved during rotation.
    transform, _ = initialize_transform_and_optim(
        d_model, "rotation"
    )
    data = t.arange(d_model, dtype=t.float).unsqueeze(0)  # Add batch dimension
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


def test_offset_rotation_transformation():
    # Assuming offset_rotation combines a rotation with an offset
    offset_rotation_transform, _ = initialize_transform_and_optim(
        d_model, "offset_rotation"
    )
    # Manually setting the rotation to identity and offset to ones for testing
    offset_rotation_transform.rotation.weight.data = t.eye(
        d_model, device=device
    )
    offset_rotation_transform.offset.data = t.ones(
        d_model, device=device
    )
    input = t.rand((batch, d_model), device=device)
    expected = input + t.ones(d_model, device=device)
    actual = offset_rotation_transform(input)
    tt.assert_close(expected, actual)

def test_uncentered_linear_map_transformation():
    # Assuming uncentered_linear_map applies a linear map followed by adding
    # a constant vector
    uncentered_linear_map_transformation, _ = initialize_transform_and_optim(
        d_model, "uncentered_linear_map"
    )
    # Setting the linear transformation to double the input and translation to ones
    uncentered_linear_map_transformation.linear_map.weight.data = (
        t.eye(d_model, device=device) * 2
    )
    uncentered_linear_map_transformation.translation.data = t.ones(
        d_model, device=device
    )
    input = t.rand((batch, d_model), device=device)
    expected = (input * 2) + t.ones(
        d_model, device=device
    )
    actual = uncentered_linear_map_transformation(input)
    tt.assert_close(expected, actual)


def test_uncentered_rotation_transformation():
    # Assuming uncentered_rotation combines a rotation with an offset, similar to
    # offset_rotation but with potential non-identity initial rotation
    uncentered_rotation_transformation, _ = initialize_transform_and_optim(
        d_model, "uncentered_rotation"
    )
    # Manually setting the rotation to identity and offset to ones for testing
    uncentered_rotation_transformation.rotation.weight.data = t.eye(
        d_model, device=device
    )
    uncentered_rotation_transformation.translation.data = t.ones(
        d_model, device=device
    )
    input = t.rand((batch, d_model), device=device)
    expected = input + t.ones(d_model, device=device)
    actual = uncentered_rotation_transformation(input)
    tt.assert_close(actual, expected)

def test_tokenize_texts():
    model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
    en_fr_pairs = [["hospital", "hôpital"], ["electronic", "électronique"]]
    expected_en_toks = [' hospital', ' Hospital', ' electronic', ' Electronic']
    expected_fr_toks = [' hôpital', ' hôpital', ' électronique', ' électronique']

    actual_en_toks, en_attn_mask, actual_fr_toks, fr_attn_mask = tokenize_texts(
        model,
        en_fr_pairs,
        padding_side="left",
        single_tokens_only=True,
        discard_if_same=False,
        min_length=3,
        capture_diff_case=True,
        capture_space=True,
        capture_no_space=True,
    )

    assert model.to_string(actual_en_toks) == expected_en_toks
    assert model.to_string(actual_fr_toks) == expected_fr_toks

# def test_create_data_loaders():
#     en_embeds = t.rand((batch0, d_model), device=device)
#     fr_embeds = t.rand((batch0, d_model), device=device)
#     train_loader, test_loader = create_data_loaders(
#         en_embeds, fr_embeds, batch_size=batch, train_ratio=0.8
#     )
#     assert isinstance(train_loader, DataLoader)
#     assert isinstance(test_loader, DataLoader)
#     for batch in train_loader:
#         assert len(batch) == 2  # Each batch should have en_embeds and fr_embeds
#         break  # Test only the first batch for brevity


# def test_mean_vec():
#     en_resids = t.rand((batch, d_model), device=device)
#     fr_resids = t.rand((batch, d_model), device=device)
#     mean_diff = mean_vec(en_resids, fr_resids)
#     assert mean_diff.shape == (d_model,)
