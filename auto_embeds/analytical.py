from typing import Any, Dict, Tuple

import torch as t
from jaxtyping import Float
from roma.utils import rigid_vectors_registration
from torch import Tensor
from torch.utils.data import DataLoader

from auto_embeds.modules import ManualTransformModule


def calculate_translation(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "pos d_model"]:
    """Calculates translation vector for source to target language embeddings."""
    X = train_src_embeds.detach().clone().squeeze()
    Y = train_tgt_embeds.detach().clone().squeeze()
    T = t.mean(Y - X, dim=0)
    return T


def calculate_procrustes_roma(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Tuple[Float[Tensor, "d_model d_model"], Float[Tensor, ""]]:
    A = train_src_embeds.detach().clone().squeeze()
    B = train_tgt_embeds.detach().clone().squeeze()
    R, scale = rigid_vectors_registration(A, B, compute_scaling=True)
    return R, scale


def calculate_orthogonal_procrustes(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
    ensure_rotation: bool = False,
) -> Tuple[Float[Tensor, "d_model d_model"], Float[Tensor, ""]]:
    A = train_src_embeds.detach().clone().squeeze()
    B = train_tgt_embeds.detach().clone().squeeze()
    M = t.matmul(B.T, A)
    U, S, Vt = t.linalg.svd(M)
    if ensure_rotation:
        if t.det(t.matmul(U, Vt)) < 0.0:
            Vt[:, -1] *= -1.0
    R = t.matmul(U, Vt)
    scale = S.sum()
    return R, scale


def calculate_linear_map(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "d_model d_model"]:
    """Calculates the best linear map matrix for source to target language embeddings."""
    A = train_src_embeds.detach().clone().squeeze()
    B = train_tgt_embeds.detach().clone().squeeze()
    # A and B after squeezing is [batch d_model] and as we are following the convention
    # of having our transformation matrix be left-multiplied i.e.
    # XA = B
    # however, this is not actually possible as A is shape [batch d_model] and X needs
    # to be shape [d_model d_model]. therefore we need to take the transpose of A and
    # whose multiplication with X gives a result of shape [d_model batch]. we then need
    # to take the transpose of this to get the B that we want.
    # as such, the linear system we want to solve for is
    # XA^T = B^T
    # but as lstsq solves the linear system AX = B for X, we can take the transpose of
    # both sides to give:
    # AX^T = B
    # which we can feed into torch.linalg.lstsq and take the transpose of the solution
    # to get X.
    result = t.linalg.lstsq(A, B)
    X = result.solution.T
    return X


def initialize_manual_transform(
    transform_name: str, train_loader: DataLoader
) -> Tuple[ManualTransformModule, Dict[str, Any]]:
    """Initializes a ManualTransformModule.

    Initializes a ManualTransformModule with transformations derived analytically from
    the training data. Also calculates expected metrics for the transformation.

    Args:
        transform_name: The name of the transformation to apply. Supported names
            include 'analytical_rotation', 'analytical_translation', etc.
        train_loader: DataLoader containing the training data used to
            calculate the transformation weights.

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

    if transform_name == "roma_analytical":
        rotation_matrix, scale = calculate_procrustes_roma(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("multiply", rotation_matrix))

    elif transform_name == "roma_scale_analytical":
        rotation_matrix, scale = calculate_procrustes_roma(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("multiply", rotation_matrix))
        transformations.append(("scale", scale))

    elif transform_name == "analytical_rotation":
        rotation_matrix, scale = calculate_orthogonal_procrustes(
            train_src_embeds, train_tgt_embeds, ensure_rotation=True
        )
        transformations.append(("multiply", rotation_matrix))

    elif transform_name == "analytical_rotation_and_reflection":
        rotation_matrix, scale = calculate_orthogonal_procrustes(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("multiply", rotation_matrix))

    elif transform_name == "analytical_translation":
        translation_vector = calculate_translation(train_src_embeds, train_tgt_embeds)
        transformations.append(("add", translation_vector))
        metrics["expected_translation_magnitude"] = (
            0.0  # Placeholder metric calculation
        )

    elif transform_name == "analytical_linear_map":
        linear_map_matrix = calculate_linear_map(train_src_embeds, train_tgt_embeds)
        transformations.append(("multiply", linear_map_matrix))
        metrics["expected_linear_map_accuracy"] = (
            0.0  # Placeholder for expected metric calculation
        )

    else:
        raise ValueError(f"Unknown transformation name: {transform_name}")

    transform_module = ManualTransformModule(transformations)

    return transform_module, metrics
