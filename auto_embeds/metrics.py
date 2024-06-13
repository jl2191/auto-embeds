import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch as t
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from auto_embeds.data import (
    get_most_similar_embeddings,
    print_most_similar_embeddings_dict,
)
from auto_embeds.modules import CosineSimilarityLoss
from auto_embeds.utils.misc import (
    default_device,
)


def calc_cos_sim_acc(
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transform: nn.Module,
    device: Optional[Union[str, t.device]] = default_device,
    print_result: bool = False,
) -> float:
    """Calculate the mean cosine similarity between predicted and actual embeddings.

    Args:
        test_loader: DataLoader for the testing dataset.
        transform: The transformation module to be evaluated.
        device: The device to perform calculations on.
        print_result: If True, prints the mean cosine similarity.

    Returns:
        The mean cosine similarity accuracy.
    """
    cosine_sims = []
    for batch_idx, (en_embed, fr_embed) in enumerate(test_loader):
        en_embed = en_embed.to(device)
        fr_embed = fr_embed.to(device)
        with t.no_grad():
            pred = transform(en_embed)
        cosine_sim = -nn.functional.cosine_similarity(pred, fr_embed, -1)
        cosine_sims.append(cosine_sim)
    mean_cosine_sim = t.cat(cosine_sims).mean().item()
    if print_result:
        print(f"Cosine Similarity = {mean_cosine_sim}")
    return mean_cosine_sim


def mean_vec(train_en_resids: t.Tensor, train_fr_resids: t.Tensor) -> t.Tensor:
    """Calculates the mean vector difference between English and French residuals.

    Args:
        train_en_resids: The tensor containing English residuals.
        train_fr_resids: The tensor containing French residuals.

    Returns:
        The mean vector difference between English and French residuals.
    """
    return train_en_resids.mean(dim=0) - train_fr_resids.mean(dim=0)


@t.no_grad()
def calc_acc_detailed(
    tokenizer: PreTrainedTokenizerBase,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transformation: nn.Module,
    unembed_module: nn.Module,
    exact_match: bool,
    device: Optional[Union[str, t.device]] = default_device,
    print_results: bool = False,
    print_top_preds: bool = True,
    print_acc: bool = True,
) -> float:
    """Evaluates the accuracy of the learned transformation by comparing the predicted
    embeddings to the actual French embeddings.

    It supports requiring exact matches or allowing for case-insensitive comparisons.

    Args:
        tokenizer: A PreTrainedTokenizerBase instance used for tokenizing texts.
        test_loader: DataLoader for test dataset.
        transformation: Transformation module to be evaluated.
        exact_match: If True, requires exact matches between predicted and actual
            embeddings. If False, matches are correct if identical ignoring case
            differences.
        device: The device on which to allocate tensors. If None, defaults to
            default_device.
        print_results: If True, prints translation attempts/results. Defaults to False.
        print_top_preds: If True and print_results=True, prints top predictions.
            Defaults to True.
        print_acc: If True, prints the correct percentage. Defaults to True.

    Returns:
        The accuracy of the learned transformation as a float.
    """
    correct_count = 0
    total_count = 0
    for en_embeds, fr_embeds in test_loader:
        en_logits = unembed_module(en_embeds)
        en_strs: List[str] = tokenizer.batch_decode(en_logits.argmax(dim=-1))
        fr_logits = unembed_module(fr_embeds)
        fr_strs: List[str] = tokenizer.batch_decode(fr_logits.argmax(dim=-1))
        pred = transformation(en_embeds)
        pred_logits = unembed_module(pred)
        pred_top_strs = tokenizer.batch_decode(pred_logits.argmax(dim=-1))
        pred_top_strs = [
            item if isinstance(item, str) else item[0] for item in pred_top_strs
        ]
        assert all(isinstance(item, str) for item in pred_top_strs)
        most_similar_embeds = get_most_similar_embeddings(
            tokenizer,
            out=pred_logits,
            top_k=4,
        )
        for i, pred_top_str in enumerate(pred_top_strs):
            fr_str = fr_strs[i]
            en_str = en_strs[i]
            correct = (
                (fr_str == pred_top_str)
                if exact_match
                else (fr_str.strip().lower() == pred_top_str.strip().lower())
            )
            correct_count += correct
            if print_results:
                result_emoji = "✅" if correct else "❌"
                print(
                    f'English: "{en_str}"\n'
                    f'French: "{fr_str}"\n'
                    f'Predicted: "{pred_top_str}" {result_emoji}'
                )
                if print_top_preds:
                    print("Top Predictions:")
                    current_most_similar_embeds = {0: most_similar_embeds[i]}
                    print_most_similar_embeddings_dict(current_most_similar_embeds)
                print()
        total_count += len(en_embeds)

    accuracy = correct_count / total_count
    if print_acc:
        print(f"Correct Percentage: {accuracy * 100:.2f}%")
    return accuracy


@t.no_grad()
def calc_acc_fast(
    tokenizer: PreTrainedTokenizerBase,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transformation: nn.Module,
    unembed_module: nn.Module,
    exact_match: bool,
    device: Optional[Union[str, t.device]] = default_device,
    print_results: bool = False,
    print_top_preds: bool = True,
    print_acc: bool = True,
) -> float:
    """Evaluates the accuracy of the learned transformation by comparing the predicted
    embeddings to the actual French embeddings.

    It supports requiring exact matches or allowing for case-insensitive comparisons.

    Args:
        tokenizer: A PreTrainedTokenizerBase instance used for tokenizing texts.
        test_loader: DataLoader for test dataset.
        transformation: Transformation module to be evaluated.
        exact_match: If True, requires exact matches between predicted and actual
            embeddings. If False, matches are correct if identical ignoring case
            differences.
        device: The device on which to allocate tensors. If None, defaults to
            default_device.
        print_results: If True, prints translation attempts/results. Defaults to False.
        print_top_preds: If True and print_results=True, prints top predictions.
            Defaults to True.
        print_acc: If True, prints the correct percentage. Defaults to True.

    Returns:
        The accuracy of the learned transformation as a float.
    """
    correct_count = 0
    total_count = 0
    for en_embeds, fr_embeds in test_loader:
        en_embeds = en_embeds.to(device)
        fr_embeds = fr_embeds.to(device)
        fr_logits = unembed_module(fr_embeds)
        fr_strs: List[str] = tokenizer.batch_decode(fr_logits.argmax(dim=-1))
        pred = transformation(en_embeds)
        pred_logits = unembed_module(pred)
        pred_top_strs = tokenizer.batch_decode(pred_logits.argmax(dim=-1))
        pred_top_strs = [
            item if isinstance(item, str) else item[0] for item in pred_top_strs
        ]
        assert all(isinstance(item, str) for item in pred_top_strs)
        correct_count += sum(
            (
                fr_str == pred_top_str
                if exact_match
                else fr_str.strip().lower() == pred_top_str.strip().lower()
            )
            for fr_str, pred_top_str in zip(fr_strs, pred_top_strs)
        )
        total_count += len(en_embeds)

    accuracy = correct_count / total_count
    if print_acc:
        print(f"Correct Percentage: {accuracy * 100:.2f}%")
    return accuracy


@t.no_grad()
def mark_translation(
    tokenizer: PreTrainedTokenizerBase,
    transformation: nn.Module,
    unembed_module: nn.Module,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    azure_translations_path: Optional[Union[str, Path]] = None,
    print_results: bool = False,
    print_top_preds: bool = False,
    translations_dict: Optional[Dict[str, List[str]]] = None,
    device: Optional[Union[str, t.device]] = default_device,
) -> float:
    """Marks translations as correct according to a dictionary of translations.

    Can either take in a dictionary of translations or a path to a JSON file containing
    translations from Azure. At least one of `azure_translations_path` or
    `translations_dict` must be provided.

    Args:
        tokenizer: A PreTrainedTokenizerBase instance used for tokenizing texts.
        transformation: The transformation module to evaluate.
        unembed_module: The module used for unembedding.
        test_loader: DataLoader for the test dataset.
        azure_translations_path: Optional; Path to the JSON file containing acceptable
            translations from Azure Translator. Defaults to None.
        print_results: Whether to print marking results.
        print_top_preds: If True and print_results=True, prints top predictions.
            Defaults to True.
        translations_dict: Optional; A dictionary of translations. Defaults to None.
        device: The device on which to allocate tensors. If None, defaults to
            default_device.
    Returns:
        The accuracy of the translations as a float.

    Raises:
        ValueError: If neither `azure_translations_path` nor `translations_dict`
            is provided.
    """
    if azure_translations_path is None and translations_dict is None:
        raise ValueError(
            "Either 'azure_translations_path' or 'translations_dict' must be given."
        )
    # load acceptable translations from JSON file if given
    if azure_translations_path:
        with open(azure_translations_path, "r", encoding="utf-8") as file:
            azure_translations = json.load(file)

    # convert list of acceptable translations to a dict is this is not already provided.
    # directly providing a dict is faster.
    if translations_dict is None:
        translations_dict = {}
        for item in azure_translations:
            source = item["normalizedSource"]
            translations = [
                trans["normalizedTarget"]
                for trans in item["translations"]
                if trans["normalizedTarget"] is not None
            ]
            translations_dict[source] = translations

    correct_count = 0
    total_marked = 0

    for en_embeds, fr_embeds in test_loader:
        en_embeds = en_embeds.to(device)
        fr_embeds = fr_embeds.to(device)
        en_logits = unembed_module(en_embeds)
        en_strs: List[str] = tokenizer.batch_decode(en_logits.squeeze().argmax(dim=-1))
        fr_logits = unembed_module(fr_embeds)
        fr_strs: List[str] = tokenizer.batch_decode(fr_logits.squeeze().argmax(dim=-1))
        pred = transformation(en_embeds)
        pred_logits = unembed_module(pred)
        pred_top_strs = tokenizer.batch_decode(pred_logits.squeeze().argmax(dim=-1))
        pred_top_strs = [
            item if isinstance(item, str) else item[0] for item in pred_top_strs
        ]
        assert all(isinstance(item, str) for item in pred_top_strs)
        # if statement for performance
        if print_top_preds:
            most_similar_embeds = get_most_similar_embeddings(
                tokenizer,
                out=pred_logits,
                top_k=4,
            )
        for i, pred_top_str in enumerate(pred_top_strs):
            correct = None
            en_str = en_strs[i]
            fr_str = fr_strs[i]
            word_found = en_str.strip().lower() in translations_dict
            if word_found:
                all_allowed_translations = translations_dict[en_str.strip().lower()]
                correct = (
                    pred_top_str.strip() in all_allowed_translations
                    or pred_top_str.strip() + "s" in all_allowed_translations
                    or pred_top_str.strip()[:-1] in all_allowed_translations
                )
                correct_count += correct
                total_marked += 1
            if print_results:
                result_emoji = "✅" if correct else "❌"
                print(
                    f'English: "{en_str}"\n'
                    f'Target: "{fr_str}"\n'
                    f'Predicted: "{pred_top_str}" {result_emoji}'
                )
                if word_found:
                    print(
                        f"Check: Found "
                        f"{list(translations_dict[en_str.strip().lower()])}"
                    )
                else:
                    print("Check: Not Found")
                if print_top_preds:
                    print("Top Predictions:")
                    current_most_similar_embeds = {0: most_similar_embeds[i]}
                    print_most_similar_embeddings_dict(current_most_similar_embeds)
                print()
    accuracy = correct_count / total_marked
    return accuracy


def calc_canonical_angles(A: t.Tensor, B: t.Tensor) -> t.Tensor:
    """Calculates the canonical angles between two matrices.

    This is to measure their rotational
    similarity. This function supports tensors of shape [n, m] and [n, 1, m], where 1
    usually represents a positional dimension.

    Args:
        A: Tensor of shape (n, m) or (n, 1, m). If a positional dimension of 1 is
           provided, it is removed. n: rows, m: columns.
        B: Tensor of shape (n, p) or (n, 1, p). If a positional dimension of 1 is
           provided, it is removed. n: rows, p: columns.

    Returns:
        t.Tensor: Contains the cosines of the canonical angles, derived from the
                  singular values of the SVD of QT_A * QB.
    """

    # Adjust A and B to 2D if they are 3D by squeezing the positional dimension
    if A.ndim == 3:
        A = A.squeeze(1)
    if B.ndim == 3:
        B = B.squeeze(1)

    # Ensure A and B are 2-dimensional
    assert A.ndim == 2, "Tensor A must be 2-dimensional after adjustment"
    assert B.ndim == 2, "Tensor B must be 2-dimensional after adjustment"

    # Ensure A and B have the same number of rows
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            "Matrices A and B must have the same number of rows after adjustment."
        )

    # QR decomposition of A and B
    QA, RA = t.linalg.qr(A)
    QB, RB = t.linalg.qr(B)

    # Compute SVD of the product of QA.T and QB
    U, Sigma, Vt = t.linalg.svd(QA.T @ QB)

    return Sigma


@t.no_grad()
def calc_pred_same_as_input(
    tokenizer: PreTrainedTokenizerBase,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transformation: nn.Module,
    unembed_module: nn.Module,
    device: Optional[Union[str, t.device]] = default_device,
) -> float:
    same_count = 0
    total_count = 0
    for en_embeds, _ in test_loader:
        en_embeds = en_embeds.to(device)
        en_logits = unembed_module(en_embeds)
        en_strs: List[str] = tokenizer.batch_decode(en_logits.argmax(dim=-1))
        pred = transformation(en_embeds)
        pred_logits = unembed_module(pred)
        pred_top_strs = tokenizer.batch_decode(pred_logits.argmax(dim=-1))
        pred_top_strs = [
            item if isinstance(item, str) else item[0] for item in pred_top_strs
        ]
        assert all(isinstance(item, str) for item in pred_top_strs)
        same_count += sum(
            en_str.strip().lower() == pred_top_str.strip().lower()
            for en_str, pred_top_str in zip(en_strs, pred_top_strs)
        )
        total_count += len(en_embeds)
    proportion_same = same_count / total_count
    return proportion_same


def initialize_loss(loss: str, loss_kwargs: Dict[str, Any] = {}) -> nn.Module:
    """Initializes a loss module.

    Initializes a loss module based on the specified loss type and optional kwargs.

    Args:
        loss: The type of loss to initialize. Supported types
            include 't_cos_sim', 't_l1_loss', 'mse_loss',
            'cos_sim', 'l1_cos_sim', and 'l2_cos_sim'.
        loss_kwargs: A dictionary of keyword arguments for the loss module.

    Returns:
        An instance of a loss module.

    Raises:
        ValueError: If an unsupported loss type is specified.
    """
    if loss == "cosine_embedding_loss":
        return nn.CosineEmbeddingLoss(**loss_kwargs)
    elif loss == "l1_loss":
        return nn.L1Loss(**loss_kwargs)
    elif loss == "mse_loss":
        return nn.MSELoss(**loss_kwargs)
    elif loss == "cos_sim":
        return CosineSimilarityLoss(**loss_kwargs)
    elif loss == "l1_cos_sim":
        return CosineSimilarityLoss(**loss_kwargs)
    elif loss == "l2_cos_sim":
        return CosineSimilarityLoss(**loss_kwargs)
    else:
        raise ValueError(f"Unsupported loss type: {loss}")


@t.no_grad()
def calc_loss(
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transform: nn.Module,
    loss_module: nn.Module,
    device: Optional[Union[str, t.device]] = default_device,
    
) -> float:
    """Calculate the average test loss over all batches in the test loader.

    Args:
        test_loader: DataLoader for the test dataset.
        transform: The transformation module to be evaluated.
        loss_module: The loss function used for evaluation.
        device: The device on which to allocate tensors. If None, defaults to
            default_device.

    Returns:
        The average test loss as a float.
    """
    total_test_loss = 0.0
    for test_en_embed, test_fr_embed in test_loader:
        test_en_embed = test_en_embed.to(device)
        test_fr_embed = test_fr_embed.to(device)
        test_pred = transform(test_en_embed)
        test_loss = loss_module(test_pred.squeeze(), test_fr_embed.squeeze())
        total_test_loss += test_loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    return avg_test_loss


@t.no_grad()
def calc_metrics(
    loader: DataLoader[Tuple[Tensor, ...]],
    transform: nn.Module,
    tokenizer: Any,
    unembed_module: nn.Module,
    azure_translations_path: Optional[Path],
) -> Dict[str, float]:
    """Calculate various metrics for a given data loader.

    Args:
        loader: DataLoader for the dataset (train or test).
        transform: The transformation module to be evaluated.
        tokenizer: The tokenizer used for tokenization.
        unembed_module: The module used for unembedding operations.
        azure_translations_path: Path to Azure translations, if available.

    Returns:
        A dictionary containing calculated metrics.
    """
    metrics = {}
    metrics["accuracy"] = calc_acc_fast(
        tokenizer=tokenizer,
        test_loader=loader,
        transformation=transform,
        unembed_module=unembed_module,
        exact_match=False,
        print_results=False,
        print_top_preds=False,
        print_acc=False,
    )
    metrics["cos_sim_loss"] = calc_loss(
        test_loader=loader,
        transform=transform,
        loss_module=initialize_loss("cos_sim"),
    )
    metrics["mse_loss"] = calc_loss(
        test_loader=loader,
        transform=transform,
        loss_module=initialize_loss("mse_loss"),
    )
    metrics["pred_same_as_input"] = calc_pred_same_as_input(
        tokenizer=tokenizer,
        test_loader=loader,
        transformation=transform,
        unembed_module=unembed_module,
    )
    if azure_translations_path:
        metrics["mark_translation_acc"] = mark_translation(
            tokenizer=tokenizer,
            transformation=transform,
            unembed_module=unembed_module,
            test_loader=loader,
            azure_translations_path=azure_translations_path,
            print_results=False,
        )
    else:
        metrics["mark_translation_acc"] = None
    return metrics
