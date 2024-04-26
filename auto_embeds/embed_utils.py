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
from transformers import PreTrainedTokenizerFast

from auto_embeds.metrics import mark_translation
from auto_embeds.modules import (
    BiasedRotationTransform,
    CosineSimilarityLoss,
    Embed,
    IdentityTransform,
    LinearTransform,
    MSELoss,
    RotationTransform,
    TranslationTransform,
    UncenteredLinearMapTransform,
    UncenteredRotationTransform,
    Unembed,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.misc import default_device


def initialize_loss(loss: str, loss_kwargs: Dict[str, Any] = {}) -> nn.Module:
    """Initializes a loss module.

    Initializes a loss module based on the specified loss type and optional kwargs.

    Args:
        loss: The type of loss to initialize. Supported types
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
    transform_kwargs: Dict[str, Any] = {},
    optim_kwargs: Dict[str, Any] = {},
    device: Union[str, t.device] = default_device,
) -> Tuple[nn.Module, Optional[Optimizer]]:
    """Initializes a transformation and its optimizer.

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


def initialize_embed_and_unembed(
    tokenizer: PreTrainedTokenizerFast,
    model_weights: Dict[str, Tensor],
    embed_weight: str = "model_weights",
    embed_ln: bool = True,
    embed_ln_weights: str = "model_weights",
    unembed_weight: str = "model_weights",
    unembed_ln: bool = True,
    unembed_ln_weights: str = "model_weights",
    device: Union[str, t.device] = default_device,
) -> Tuple[Embed, Unembed]:
    """
    Initializes instances of Embed and Unembed with optional layer normalization
    and the ability to use weights from a provided model or initialize randomly.

    Args:
        tokenizer: The tokenizer used for tokenization.
        model_weights: Dict with keys "W_E", "W_U", "embed.ln.w", "embed.ln.b",
            "ln_final.w", "ln_final.b".
        embed_weight: If "model_weights", uses weights from the model for EmbedModule. If
            "random_normal" or "random_uniform", initializes weights randomly with
            normal or uniform distribution respectively. Defaults to "model_weights".
        embed_ln: If True, applies layer normalization in EmbedModule.
        embed_ln_weights: If "model_weights", uses layer normalization weights from the model
            for EmbedModule. If "default_weights", does not apply weights. Defaults to "model_weights".
        unembed_weight: Similar to embed_weight but for UnembedModule.
        unembed_ln: If True, applies layer normalization in UnembedModule.
        unembed_ln_weights: Similar to embed_ln_weights but for UnembedModule.
        device: The device on which to allocate tensors. If None, defaults to
            default_device.

    Returns:
        A tuple containing instances of Embed and Unembed.
    """
    d_model = model_weights["W_E"].shape[1]
    d_vocab = tokenizer.vocab_size

    # Initialize EmbedModule
    embed_module = Embed(d_model, d_vocab, apply_ln=embed_ln, device=device)
    if embed_weight == "model_weights":
        embed_module.W_E.data = model_weights["W_E"].detach().clone()
    elif embed_weight == "random_normal":
        t.nn.init.normal_(embed_module.W_E.data, mean=0, std=1)
    elif embed_weight == "random_uniform":
        t.nn.init.uniform_(embed_module.W_E.data, -1, 1)
    elif embed_weight == "default_weights":
        pass
    else:
        raise ValueError(f"Unsupported embedding initialization method: {embed_weight}")

    if embed_ln and embed_ln_weights == "model_weights":
        embed_module.embed_ln.weight.data = model_weights["embed.ln.w"].detach().clone()
        embed_module.embed_ln.bias.data = model_weights["embed.ln.b"].detach().clone()
    elif embed_ln and embed_ln_weights == "random_normal":
        t.nn.init.normal_(embed_module.embed_ln.weight.data, mean=0, std=1)
        t.nn.init.normal_(embed_module.embed_ln.bias.data, mean=0, std=1)
    elif embed_ln and embed_ln_weights == "random_uniform":
        t.nn.init.uniform_(embed_module.embed_ln.weight.data, -1, 1)
        t.nn.init.uniform_(embed_module.embed_ln.bias.data, -1, 1)
    elif embed_ln and embed_ln_weights == "default_weights":
        pass
    elif not embed_ln:
        pass
    else:
        raise ValueError(
            f"Unsupported embedding layer normalization method: {embed_ln_weights}"
        )

    # Initialize Unembed
    unembed_module = Unembed(d_model, d_vocab, apply_ln=unembed_ln, device=device)
    if unembed_weight == "model_weights":
        unembed_module.W_U.data = model_weights["W_U"].detach().clone()
        unembed_module.b_U.data = model_weights["b_U"].detach().clone()
    elif unembed_weight == "random_normal":
        t.nn.init.normal_(unembed_module.W_U.data, mean=0, std=1)
        t.nn.init.normal_(unembed_module.b_U.data, mean=0, std=1)
    elif unembed_weight == "random_uniform":
        t.nn.init.uniform_(unembed_module.W_U.data, -1, 1)
        t.nn.init.uniform_(unembed_module.b_U.data, -1, 1)
    else:
        raise ValueError(
            f"Unsupported unembedding initialization method: {unembed_weight}"
        )

    if unembed_ln and unembed_ln_weights == "model_weights":
        unembed_module.ln_final.weight.data = (
            model_weights["ln_final.w"].detach().clone()
        )
        unembed_module.ln_final.bias.data = model_weights["ln_final.b"].detach().clone()
    elif unembed_ln and unembed_ln_weights == "random_normal":
        t.nn.init.normal_(unembed_module.ln_final.weight.data, mean=0, std=1)
        t.nn.init.normal_(unembed_module.ln_final.bias.data, mean=0, std=1)
    elif unembed_ln and unembed_ln_weights == "random_uniform":
        t.nn.init.uniform_(unembed_module.ln_final.weight.data, -1, 1)
        t.nn.init.uniform_(unembed_module.ln_final.bias.data, -1, 1)
    elif unembed_ln and unembed_ln_weights == "default_weights":
        pass
    elif not unembed_ln:
        pass
    else:
        raise ValueError(
            f"Unsupported unembedding layer normalization method: "
            f"{unembed_ln_weights}"
        )

    return embed_module, unembed_module


def train_transform(
    tokenizer: PreTrainedTokenizerFast,
    train_loader: DataLoader[Tuple[Tensor, ...]],
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transform: nn.Module,
    optim: Optimizer,
    loss_module: nn.Module,
    unembed_module: nn.Module,
    n_epochs: int,
    plot_fig: bool = True,
    save_fig: bool = False,
    device: Union[str, t.device] = default_device,
    wandb: Optional[Any] = None,
    azure_translations_path: Optional[Union[str, Path]] = None,
) -> Tuple[nn.Module, Dict[str, List[Dict[str, Union[float, int]]]]]:
    """Trains the transformation.

    Trains the transformation, returning the learned transformation and loss history.

    Args:
        tokenizer: The tokenizer used for tokenization.
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
        with open(azure_translations_path, "r", encoding="utf-8") as file:
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
                wandb.log(info_dict, commit=False)
            epoch_pbar.set_description(f"train loss: {train_loss.item():.3f}")
        # Calculate and log test loss at the end of each epoch divisible by 10
        if epoch % 10 == 0:
            with t.no_grad():
                total_test_loss = 0
                for test_en_embed, test_fr_embed in test_loader:
                    test_pred = transform(test_en_embed)
                    test_loss = loss_module(
                        test_pred.squeeze(), test_fr_embed.squeeze()
                    )
                    total_test_loss += test_loss.item()
                avg_test_loss = total_test_loss / len(test_loader)
                info_dict = {"test_loss": avg_test_loss, "epoch": epoch}
                train_history["test_loss"].append(info_dict)
                # Calculate and log mark_translation score if azure_translations_path
                if azure_translations_path:
                    mark_translation_score = mark_translation(
                        tokenizer=tokenizer,
                        transformation=transform,
                        unembed_module=unembed_module,
                        test_loader=test_loader,
                        translations_dict=translations_dict,
                        print_results=False,
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
    """Generates new embeddings.

    Generates multiple new embeddings by adding noise to the original
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
