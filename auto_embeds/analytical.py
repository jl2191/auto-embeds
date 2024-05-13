from typing import Any, Dict, Tuple

import torch as t
from jaxtyping import Float
from roma.utils import rigid_vectors_registration
from torch import Tensor
from torch.utils.data import DataLoader

from auto_embeds.modules import ManualTransformModule
from auto_embeds.utils.logging import logger


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
    scale = [1.0] * A.shape[0]
    R = rigid_vectors_registration(A, B)
    # ic(R.shape)
    # ic(scale)
    return R, scale


def calculate_procrustes_torch(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Tuple[Float[Tensor, "d_model d_model"], Float[Tensor, ""]]:
    A = train_src_embeds.detach().clone().squeeze()
    B = train_tgt_embeds.detach().clone().squeeze()
    u, w, vt = t.linalg.svd(t.matmul(B.T, A).T)
    R = u @ vt
    scale = w.sum()
    return R, scale


def calculate_kabsch(P, Q):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
            translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"
    P = P.detach().clone().squeeze()
    Q = Q.detach().clone().squeeze()

    # Compute centroids
    centroid_P = t.mean(P, dim=0)
    centroid_Q = t.mean(Q, dim=0)

    # Optimal translation
    T = centroid_Q - centroid_P

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = t.matmul(p.transpose(0, 1), q)

    # SVD
    U, S, Vt = t.linalg.svd(H)

    # Validate right-handed coordinate system
    if t.det(t.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
        Vt[:, -1] *= -1.0

    # Optimal rotation
    R = t.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

    # RMSD
    rmsd = t.sqrt(t.sum(t.square(t.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

    return R, T, rmsd


def calculate_linear_map(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "d_model d_model"]:
    """Calculates the best linear map matrix for source to target language embeddings."""
    A = train_src_embeds.detach().clone().squeeze()
    B = train_tgt_embeds.detach().clone().squeeze()
    logger.debug(f"A.shape: {A.shape}")
    logger.debug(f"B.shape: {B.shape}")
    # to solve the linear system XA = B for X, as lstsq solves the linear system
    # AX = B we can solve A^T X^T = B^T and then transpose the result to get X.
    result = t.linalg.lstsq(A, B)
    X = result.solution
    logger.debug(f"X.shape: {X.shape}")
    return X


def calculate_rotation_translation(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Tuple[Float[Tensor, "pos d_model d_model"], Float[Tensor, "pos d_model"]]:
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
    elif transform_name == "torch_analytical":
        rotation_matrix, scale = calculate_procrustes_torch(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("multiply", rotation_matrix))
    elif transform_name == "roma_scale_analytical":
        rotation_matrix, scale = calculate_procrustes_roma(
            train_src_embeds, train_tgt_embeds
        )
        logger.debug(f"scale: {scale}")
        transformations.append(("multiply", rotation_matrix))
        transformations.append(("scale", scale))
    elif transform_name == "torch_scale_analytical":
        rotation_matrix, scale = calculate_procrustes_torch(
            train_src_embeds, train_tgt_embeds
        )
        logger.debug(f"scale: {scale}")
        transformations.append(("multiply", rotation_matrix))
        transformations.append(("scale", scale))
    elif transform_name == "kabsch_analytical":
        rotation_matrix, translation_vector, rmsd = calculate_kabsch(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("multiply", rotation_matrix))
        transformations.append(("add", translation_vector))
        metrics["expected_kabsch_rmsd"] = rmsd
    elif transform_name == "kabsch_analytical_new":
        rotation_matrix, translation_vector, rmsd = calculate_kabsch(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("add", translation_vector))
        transformations.append(("multiply", rotation_matrix))
        metrics["expected_kabsch_rmsd"] = rmsd
    elif transform_name == "kabsch_analytical_no_scale":
        rotation_matrix, translation_vector, rmsd = calculate_kabsch(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("multiply", rotation_matrix))
        metrics["expected_kabsch_rmsd"] = rmsd
    elif transform_name == "analytical_translation":
        translation_vector = calculate_translation(train_src_embeds, train_tgt_embeds)
        transformations.append(("add", translation_vector))
        metrics["expected_translation_magnitude"] = (
            0.0  # Placeholder metric calculation
        )
    elif transform_name == "rotation_then_translation":
        rotation_matrix, translation_vector = calculate_rotation_translation(
            train_src_embeds, train_tgt_embeds
        )
        transformations.append(("multiply", rotation_matrix))
        transformations.append(("add", translation_vector))
        metrics["expected_combined_magnitude"] = 0.0
    elif transform_name == "analytical_linear_map":
        linear_map_matrix = calculate_linear_map(train_src_embeds, train_tgt_embeds)
        transformations.append(("multiply", linear_map_matrix))
        metrics["expected_linear_map_accuracy"] = (
            0.0  # Placeholder for expected metric calculation
        )
    transform_module = ManualTransformModule(transformations)

    return transform_module, metrics
