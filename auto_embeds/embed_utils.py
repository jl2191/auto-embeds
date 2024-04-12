import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import plotly.express as px
import torch as t
import torch.nn as nn
import transformer_lens as tl
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from auto_embeds.metrics import mark_translation
from auto_embeds.modules import (
    BiasedRotationTransform,
    CosineSimilarityLoss,
    IdentityTransform,
    LinearTransform,
    MeanTranslationTransform,
    MSELoss,
    RotationTransform,
    TranslationTransform,
    UncenteredLinearMapTransform,
    UncenteredRotationTransform,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.misc import default_device


def initialize_loss(loss: str, loss_kwargs: Dict[str, Any] = {}) -> nn.Module:
    """Initializes a loss module based on the specified loss type and optional kwargs.

    Args:
        loss: A string specifying the type of loss to initialize. Supported types
            include 't_cosine_similarity', 't_l1_loss', 't_mse_loss', 'mse_loss',
            'cosine_similarity', 'l1_cosine_similarity', and 'l2_cosine_similarity'.
        loss_kwargs: A dictionary of keyword arguments for the loss module.

    Returns:
        An instance of a loss module.

    Raises:
        ValueError: If an unsupported loss type is specified.
    """
    if loss == "t_cosine_similarity":
        return nn.CosineEmbeddingLoss(**loss_kwargs)
    elif loss == "t_l1_loss":
        return nn.L1Loss(**loss_kwargs)
    elif loss == "t_mse_loss":
        return nn.MSELoss(**loss_kwargs)
    elif loss == "mse_loss":
        return MSELoss(**loss_kwargs)
    elif loss == "cosine_similarity":
        return CosineSimilarityLoss(**loss_kwargs)
    elif loss == "l1_cosine_similarity":
        return CosineSimilarityLoss(**loss_kwargs)
    elif loss == "l2_cosine_similarity":
        return CosineSimilarityLoss(**loss_kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss}")


def initialize_transform_and_optim(
    d_model: int,
    transformation: str,
    mean_diff: Optional[Tensor] = None,
    transform_kwargs: Dict[str, Any] = {},
    optim_kwargs: Dict[str, Any] = {},
    device: Union[str, t.device] = default_device,
) -> Tuple[nn.Module, Optional[Optimizer]]:
    """Initializes a transformation and its corresponding optimizer.

    Initializes a transformation and its optimizer based on the specified type,
    allowing for flexible configuration through keyword argument dictionaries for
    both components.

    Args:
        d_model: The dimensionality of the model embeddings.
        transformation: The type of transformation to initialize. Supported are
            include 'identity', 'translation', 'mean_translation', 'linear_map',
            'biased_linear_map', 'uncentered_linear_map',
            'biased_uncentered_linear_map', 'rotation', 'biased_rotation',
            'uncentered_rotation'.
        mean_diff: Optional tensor containing the mean difference for mean-centered
            steering transformation.
        transform_kwargs: Dict containing kwargs for transformation initialization.
        optim_kwargs: Dict containing kwargs for optimizer initialization.
        device: The device on which to allocate tensors. If None, defaults to
            default_device.

    Returns:
        A tuple containing the transformation module and its optimizer.
    """
    transform_kwargs["device"] = device

    if transformation == "identity":
        transform = IdentityTransform(d_model, **transform_kwargs)
        optim = None

    elif transformation == "translation":
        transform = TranslationTransform(d_model, **transform_kwargs)
        optim = t.optim.Adam([transform.translation], **optim_kwargs)

    elif transformation == "mean_translation":
        if mean_diff is None:
            raise ValueError(("The mean difference tensor must be provided "))
        transform = MeanTranslationTransform(mean_diff, **transform_kwargs)
        optim = None

    elif transformation == "linear_map":
        transform = LinearTransform(d_model, bias=False, **transform_kwargs)
        optim = t.optim.Adam(transform.parameters(), **optim_kwargs)

    elif transformation == "biased_linear_map":
        transform = LinearTransform(d_model, bias=True, **transform_kwargs)
        optim = t.optim.Adam(transform.parameters(), **optim_kwargs)

    elif transformation == "uncentered_linear_map":
        transform = UncenteredLinearMapTransform(d_model, **transform_kwargs)
        optim = t.optim.Adam(list(transform.parameters()), **optim_kwargs)

    elif transformation == "biased_uncentered_linear_map":
        transform = UncenteredLinearMapTransform(d_model, bias=True, **transform_kwargs)
        optim = t.optim.Adam(list(transform.parameters()), **optim_kwargs)

    elif transformation == "rotation":
        transform = RotationTransform(d_model, **transform_kwargs)
        optim = t.optim.Adam(list(transform.parameters()), **optim_kwargs)

    elif transformation == "biased_rotation":
        transform = BiasedRotationTransform(d_model, **transform_kwargs)
        optim = t.optim.Adam(list(transform.parameters()), **optim_kwargs)

    elif transformation == "uncentered_rotation":
        transform = UncenteredRotationTransform(d_model, **transform_kwargs)
        optim = t.optim.Adam(list(transform.parameters()), **optim_kwargs)
    else:
        raise Exception(f"the supplied transform '{transformation}' was unrecognized")
    return transform, optim


def train_transform(
    model: tl.HookedTransformer,
    train_loader: DataLoader[Tuple[Tensor, ...]],
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transform: nn.Module,
    optim: Optimizer,
    loss_module: nn.Module,
    n_epochs: int,
    plot_fig: bool = True,
    save_fig: bool = False,
    device: Union[str, t.device] = default_device,
    wandb: Optional[Any] = None,
    azure_translations_path: Optional[Union[str, Path]] = None,
    unembed_config: Optional[dict] = None,
) -> Tuple[nn.Module, Dict[str, List[Dict[str, Union[float, int]]]]]:
    """Trains the transformation, returning the learned transformation and loss history.

    Args:
        model: The transformer model used for training.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
        transform: The transformation module to be optimized.
        optim: The optimizer for the transformation.
        loss_module: The loss function used for training.
        n_epochs: The number of epochs to train for.
        plot_fig: If True, plots the training and test loss history.
        device: The device on which the model is allocated.
        wandb: If provided, log training metrics to Weights & Biases.
        azure_translations_path: Path to JSON file for mark_translation evaluation.

    Returns:
        The learned transformation after training, the train and test loss history.
    """
    train_history = {"train_loss": [], "test_loss": [], "mark_translation_score": []}
    transform.train()
    if wandb:
        wandb.watch(transform, log="all", log_freq=500)
    # if a azure_translations_path is provided we process the azure json file into a
    # more accessible format just once to speed up the marking, passing a
    # translations_dict directly into mark_translate()
    if azure_translations_path:
        with open(azure_translations_path, "r") as file:
            allowed_translations = json.load(file)
        translations_dict = {}
        for item in allowed_translations:
            source = item["normalizedSource"]
            translations = [
                trans["normalizedTarget"]
                for trans in item["translations"]
                if trans["normalizedTarget"] is not None
            ]
            translations_dict[source] = translations
    for epoch in (epoch_pbar := tqdm(range(n_epochs))):
        for batch_idx, (en_embed, fr_embed) in enumerate(train_loader):
            optim.zero_grad()
            pred = transform(en_embed)
            train_loss = loss_module(pred.squeeze(), fr_embed.squeeze())
            info_dict = {
                "train_loss": train_loss.item(),
                "batch": batch_idx,
                "epoch": epoch,
            }
            train_history["train_loss"].append(info_dict)
            train_loss.backward()
            optim.step()
            if wandb:
                wandb.log(info_dict)
            epoch_pbar.set_description(f"train loss: {train_loss.item():.3f}")
        # Calculate and log test loss at the end of each epoch
        with t.no_grad():
            total_test_loss = 0
            for test_en_embed, test_fr_embed in test_loader:
                test_pred = transform(test_en_embed)
                test_loss = loss_module(test_pred.squeeze(), test_fr_embed.squeeze())
                total_test_loss += test_loss.item()
            avg_test_loss = total_test_loss / len(test_loader)
            info_dict = {"test_loss": avg_test_loss, "epoch": epoch}
            train_history["test_loss"].append(info_dict)
            # Calculate and log mark_translation score if azure_translations_path
            if azure_translations_path:
                mark_translation_score = mark_translation(
                    model=model,
                    transformation=transform,
                    test_loader=test_loader,
                    translations_dict=translations_dict,
                    print_results=False,
                    unembed_config=unembed_config,
                )
                info_dict.update(
                    {
                        "mark_translation_score": mark_translation_score,
                    }
                )
                train_history["mark_translation_score"].append(info_dict)
            if wandb:
                wandb.log(info_dict)
    if plot_fig or save_fig:
        fig = px.line(title="Train and Test Loss with Mark Correct Score")
        fig.add_scatter(
            x=[epoch_info["epoch"] for epoch_info in train_history["train_loss"]],
            y=[epoch_info["train_loss"] for epoch_info in train_history["train_loss"]],
            name="Train Loss",
        )
        fig.add_scatter(
            x=[epoch_info["epoch"] for epoch_info in train_history["test_loss"]],
            y=[epoch_info["test_loss"] for epoch_info in train_history["test_loss"]],
            name="Test Loss",
        )
        # Plot mark_translation_score if available
        if azure_translations_path:
            fig.add_scatter(
                x=[
                    epoch_info["epoch"]
                    for epoch_info in train_history["mark_translation_score"]
                ],
                y=[
                    epoch_info["mark_translation_score"]
                    for epoch_info in train_history["mark_translation_score"]
                ],
                name="Mark Correct Score",
            )
        if plot_fig:
            fig.show()
        if save_fig:
            fig.write_image("plot.png")
    return transform, train_history


def generate_new_embeddings_from_noise(
    embedding_matrix: t.Tensor,
    num_copies: int = 1,
    dist_dict: dict = {"distribution": "uniform", "alpha": 0.01},
) -> t.Tensor:
    """Generate multiple new embeddings by adding noise to the original embeddings.

    This function generates multiple new embeddings by adding noise to the original
    embeddings multiple times based on the specified distribution.

    Args:
        embedding_matrix: The original embedding matrix.
        num_copies: Number of noisy copies to generate for each vector in the
            embedding matrix.
        dist_dict: A dictionary specifying the noise distribution and its parameters.
            Supported distributions are "uniform" (requires "alpha") and "gaussian"
            (requires "sd").

    Returns:
        New embeddings with added noise. The shape will be
        (num_copies, *embedding_matrix.shape).
    """
    distribution = dist_dict.get("distribution", "uniform")
    if distribution == "gaussian":
        sd = dist_dict.get("sd")
        if sd is None:
            raise ValueError(
                "Standard deviation 'sd' must be provided for gaussian distribution."
            )
        new_embeddings = []
        for _ in range(num_copies):
            noise = t.randn_like(embedding_matrix) * sd
            noisy_embeddings = embedding_matrix + noise
            new_embeddings.append(noisy_embeddings)
        new_embeddings = t.cat(new_embeddings, dim=0)
    elif distribution == "uniform":
        alpha = dist_dict.get("alpha")
        if alpha is None:
            raise ValueError(
                "Scaling factor 'alpha' must be provided for uniform distribution."
            )
        seq_len, d_model = (
            embedding_matrix.shape[-2],
            embedding_matrix.shape[-1],
        )  # Assuming [batch, seq_len, d_model] or [seq_len, d_model]
        scale_factor = alpha / (seq_len * d_model) ** 0.5
        new_embeddings = []
        for _ in range(num_copies):
            noise = (
                t.rand_like(embedding_matrix) * 2 - 1
            ) * scale_factor  # Uniform noise in [-1, 1]
            noisy_embeddings = embedding_matrix + noise
            new_embeddings.append(noisy_embeddings)
        new_embeddings = t.cat(new_embeddings, dim=0)
    else:
        raise ValueError(f"Distribution '{distribution}' is not recognized.")
    new_embeddings = t.cat([embedding_matrix, new_embeddings], dim=0)
    assert new_embeddings.shape[0] == embedding_matrix.shape[0] * (num_copies + 1)
    return new_embeddings
