from typing import Tuple

import torch as t
import torch.nn as nn
from torch import Tensor

from auto_embeds.modules import ManualTransformModule


def calculate_rotation(train_src_embeds: Tensor, train_tgt_embeds: Tensor) -> Tensor:
    """Calculates rotation matrix for source to target language embeddings."""
    X = train_src_embeds.detach().clone().squeeze()
    Y = train_tgt_embeds.detach().clone().squeeze()
    C = t.matmul(X.T, Y)
    U, _, V = t.svd(C)
    W = t.matmul(U, V.t())
    return W


def calculate_translation(train_src_embeds: Tensor, train_tgt_embeds: Tensor) -> Tensor:
    """Calculates translation vector for source to target language embeddings."""
    X = train_src_embeds.detach().clone().squeeze()
    Y = train_tgt_embeds.detach().clone().squeeze()
    T = t.mean(Y - X, dim=0)
    return T


def calculate_rotation_translation(
    train_src_embeds: Tensor, train_tgt_embeds: Tensor
) -> Tuple[Tensor, Tensor]:
    """Calculates rotation then translation transformation matrix for embeddings."""
    X = train_src_embeds.detach().clone().squeeze()
    Y = train_tgt_embeds.detach().clone().squeeze()
    X_mean = t.mean(X, dim=0, keepdim=True)
    Y_mean = t.mean(Y, dim=0, keepdim=True)
    X_centered = X - X_mean
    Y_centered = Y - Y_mean
    C = t.matmul(X_centered.T, Y_centered)
    U, _, V = t.svd(C)
    W = t.matmul(U, V.t())
    X_rotated = t.matmul(X_centered, W)
    b = Y_mean - t.mean(X_rotated, dim=0, keepdim=True)
    return W, b


def initialize_manual_transform(
    transform_name, train_loader, apply_ln=False, d_model=None
):
    """Initializes a ManualTransformModule.

    Initializes a ManualTransformModule with transformations derived analytically from
    the training data. Also calculates expected metrics for the transformation.

    Args:
        transform_name: The name of the transformation to apply. Supported names
            include 'analytical_rotation', 'analytical_translation', etc.
        train_loader: DataLoader containing the training data used to
            calculate the transformation weights.
        apply_ln: If True, applies layer normalization after the transformations.
        d_model: Model embedding dimensionality. Required if apply_ln is True.

    Returns:
        tuple: A tuple containing the ManualTransformModule and a metrics dictionary.
    """
    transformations = []
    metrics = {}

    # Initialize placeholders for embeddings
    train_src_embeds, train_tgt_embeds = [], []

    # Extract embeddings from the train_loader
    for batch in train_loader:
        src_embeds, tgt_embeds = (
            batch  # Assuming each batch is a tuple (src_embeds, tgt_embeds)
        )
        train_src_embeds.append(src_embeds)
        train_tgt_embeds.append(tgt_embeds)

    # Convert lists to tensors
    train_src_embeds = t.cat(train_src_embeds, dim=0)
    train_tgt_embeds = t.cat(train_tgt_embeds, dim=0)

    if transform_name == "analytical_rotation":
        rotation_matrix = calculate_rotation(train_src_embeds, train_tgt_embeds)
        transformations.append(("multiply", rotation_matrix))
        metrics["expected_loss"] = (
            0.0  # TODO: Placeholder for expected metric calculation
        )
    elif transform_name == "analytical_translation":
        translation_vector = calculate_translation(train_src_embeds, train_tgt_embeds)
        transformations.append(("add", translation_vector))
        metrics["expected_translation_magnitude"] = (
            0.0  # Placeholder metric calculation
        )

    elif transform_name == "rotation_translation":
        rotation_matrix, translation_vector = calculate_rotation_translation(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("multiply", rotation_matrix))
        transformations.append(("add", translation_vector))
        metrics["expected_combined_magnitude"] = 0.0  # Placeholder metric calculation

    transform_module = ManualTransformModule(transformations)

    if apply_ln:
        if d_model is None:
            raise ValueError("d_model must be specified if apply_ln is True.")
        transform_module = nn.Sequential(transform_module, nn.LayerNorm(d_model))

    return transform_module, metrics
