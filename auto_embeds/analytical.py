from typing import Tuple, Union

import torch as t
from jaxtyping import Float
from roma.utils import rigid_vectors_registration
from torch import Tensor
from torch.utils.data import DataLoader

from auto_embeds.modules import ManualTransformModule
from auto_embeds.utils.logging import logger
from auto_embeds.utils.misc import default_device


def calc_translation(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "pos d_model"]:
    """Calculates translation vector for source to target language embeddings.

    Args:
        train_src_embeds: Source language embeddings of shape.
        train_tgt_embeds: Target language embeddings of shape.

    Returns:
        Translation vector of shape (pos, d_model).
    """
    X = train_src_embeds.detach().clone().squeeze()
    Y = train_tgt_embeds.detach().clone().squeeze()
    T = t.mean(Y - X, dim=0)
    return T


def calc_procrustes_roma(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Tuple[Float[Tensor, "d_model d_model"], Float[Tensor, ""]]:
    """Calculates the Procrustes rotation matrix and scale using ROMA.

    Args:
        train_src_embeds: Source language embeddings.
        train_tgt_embeds: Target language embeddings.

    Returns:
        A tuple containing:
            - Rotation matrix of shape (d_model, d_model).
            - Scale factor.
    """
    A = train_src_embeds.detach().clone().squeeze()
    B = train_tgt_embeds.detach().clone().squeeze()
    R, scale = rigid_vectors_registration(A, B, compute_scaling=True)
    return R, scale


def calc_orthogonal_procrustes(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
    ensure_rotation: bool = False,
) -> Tuple[Float[Tensor, "d_model d_model"], Float[Tensor, ""]]:
    """Calculates the orthogonal Procrustes rotation matrix and scale.

    Args:
        train_src_embeds: Source language embeddings.
        train_tgt_embeds: Target language embeddings.
        ensure_rotation: If True, ensures the resulting matrix is a proper rotation.

    Returns:
        A tuple containing:
            - Rotation matrix of shape (d_model, d_model).
            - Scale factor.
    """
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


def calc_linear_map(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "d_model d_model"]:
    """Calculates the best linear map matrix for source to target language embeddings.

    Args:
        train_src_embeds: Source language embeddings.
        train_tgt_embeds: Target language embeddings.

    Returns:
        A tuple containing:
            - Linear map matrix of shape (d_model, d_model).
            - Residual of the least squares solution.
    """
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
    transform_name: str,
    train_loader: DataLoader,
    device: Union[str, t.device] = default_device,
) -> ManualTransformModule:
    """Initializes a ManualTransformModule.

    Initializes a ManualTransformModule with transformations derived analytically from
    the training data.

    Args:
        transform_name: The name of the transformation to apply. Supported names
            include 'analytical_rotation', 'analytical_translation', etc.
        train_loader: DataLoader containing the training data used to
            calculate the transformation weights.

    Returns:
        A ManualTransformModule initialized with the specified transformation.
    """
    transformations = []

    # Initialize placeholders for embeddings
    train_src_embeds, train_tgt_embeds = [], []

    # Extract embeddings from the train_loader
    for src_embeds, tgt_embeds in train_loader:
        src_embeds = src_embeds.to(device)
        tgt_embeds = tgt_embeds.to(device)
        train_src_embeds.append(src_embeds)
        train_tgt_embeds.append(tgt_embeds)

    # Convert lists to tensors
    train_src_embeds = t.cat(train_src_embeds, dim=0)
    train_tgt_embeds = t.cat(train_tgt_embeds, dim=0)

    if transform_name == "analytical_translation":
        translation_vector = calc_translation(train_src_embeds, train_tgt_embeds)
        transformations.append(("add", translation_vector))

    elif transform_name in {"roma_analytical", "roma_scale_analytical"}:
        rotation_matrix, scale = calc_procrustes_roma(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("multiply", rotation_matrix))
        if transform_name == "roma_scale_analytical":
            transformations.append(("scale", scale))

    elif transform_name in {
        "analytical_rotation",
        "analytical_rotation_and_reflection",
    }:
        rotation_matrix, scale = calc_orthogonal_procrustes(
            train_src_embeds,
            train_tgt_embeds,
            ensure_rotation=(transform_name == "analytical_rotation"),
        )
        transformations.append(("multiply", rotation_matrix))

    elif transform_name == "analytical_linear_map":
        linear_map_matrix = calc_linear_map(train_src_embeds, train_tgt_embeds)
        transformations.append(("multiply", linear_map_matrix))

    else:
        logger.error(f"Unknown transformation name: {transform_name}")
        raise ValueError(f"Unknown transformation name: {transform_name}")

    transform_module = ManualTransformModule(transformations)

    return transform_module
