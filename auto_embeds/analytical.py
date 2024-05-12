from typing import Tuple

import torch as t
from jaxtyping import Float
from scipy.linalg import orthogonal_procrustes
from torch import Tensor

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


def calculate_rotation_kabsch(
    P: Float[Tensor, "batch pos d_model"],
    Q: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "d_model d_model"]:
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
            translation vector, and the RMSD.
    """
    assert P.shape == Q.shape, "Matrix dimensions must match"
    logger.debug(f"P.shape: {P.shape}")
    logger.debug(f"Q.shape: {Q.shape}")

    # Compute centroids
    centroid_P = t.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = t.mean(Q, dim=1, keepdims=True)  # Bx1x3

    logger.debug(
        f"Allocated GPU memory after computing centroids: "
        f"{t.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    # Optimal translation
    T = centroid_Q - centroid_P  # Bx1x3
    T = T.squeeze(1)  # Bx3

    # Center the points
    p = P - centroid_P  # BxNx3
    q = Q - centroid_Q  # BxNx3

    # Compute the covariance matrix
    H = t.matmul(p.transpose(1, 2), q)  # Bx3x3
    logger.debug(f"H.shape: {H.shape}")

    logger.debug(
        f"Allocated GPU memory after computing covariance matrix: "
        f"{t.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    # SVD
    U, S, Vt = t.linalg.svd(H)  # Bx3x3
    logger.debug(f"U.shape: {U.shape}")
    logger.debug(f"S.shape: {S.shape}")
    logger.debug(f"Vt.shape: {Vt.shape}")

    # Validate right-handed coordinate system
    d = t.det(t.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))  # B
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0

    # Optimal rotation
    R = t.matmul(Vt.transpose(1, 2), U.transpose(1, 2))

    # RMSD
    rmsd = t.sqrt(
        t.sum(t.square(t.matmul(p, R.transpose(1, 2)) - q), dim=(1, 2)) / P.shape[1]
    )

    return R, T, rmsd


def calculate_rotation_scipy(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "d_model d_model"]:
    """Calculates the best linear map matrix for source to target language embeddings."""
    A = train_src_embeds.detach().clone().squeeze().cpu()
    B = train_tgt_embeds.detach().clone().squeeze().cpu()
    R, _ = orthogonal_procrustes(A, B)
    return R


def calculate_rotation_torch(
    train_src_embeds: Float[Tensor, "batch pos d_model"],
    train_tgt_embeds: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "d_model d_model"]:
    """Calculates the best linear map matrix for source to target language embeddings."""
    A = train_src_embeds.detach().clone().squeeze()
    B = train_tgt_embeds.detach().clone().squeeze()
    u, w, vt = t.linalg.svd(B.T.dot(A).T)
    R = u.dot(vt)
    return R


def calculate_rotation(train_src_embeds: Tensor, train_tgt_embeds: Tensor) -> Tensor:
    """Calculates rotation matrix for source to target language embeddings using RoMa."""
    A = train_src_embeds.detach().clone().squeeze()
    B = train_tgt_embeds.detach().clone().squeeze()
    # Using RoMa to compute the special Procrustes orthonormalization
    R = roma.special_procrustes(t.matmul(A.T, B))
    return R


def calculate_procrustes(
    train_src_embeds: Tensor, train_tgt_embeds: Tensor, force_rotation: bool = True
) -> Tensor:
    """Calculates the Procrustes transformation matrix using RoMa, optionally forcing a proper rotation."""
    A = train_src_embeds.detach().clone().squeeze()
    B = train_tgt_embeds.detach().clone().squeeze()
    C = t.matmul(A.T, B)  # Compute the cross-covariance matrix
    R, _ = roma.procrustes(C, force_rotation=force_rotation)
    return R


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


def initialize_manual_transform(transform_name, train_loader):
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

    if transform_name == "analytical_rotation":
        rotation_matrix = calculate_rotation(train_src_embeds, train_tgt_embeds)
        transformations.append(("multiply", rotation_matrix))
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
