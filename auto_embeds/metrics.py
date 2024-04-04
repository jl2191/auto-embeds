import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import einops
import torch as t
import torch.nn as nn
import transformer_lens as tl
from torch import Tensor
from torch.utils.data import DataLoader

from auto_embeds.utils.misc import (
    default_device,
    get_most_similar_embeddings,
    print_most_similar_embeddings_dict,
)


def word_distance_metric(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Computes the negative cosine similarity between two tensors.

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        The negative cosine similarity between the input tensors.
    """
    return -nn.functional.cosine_similarity(a, b, -1)


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
        cosine_sim = word_distance_metric(pred, fr_embed)
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


def evaluate_accuracy(
    model: tl.HookedTransformer,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transformation: nn.Module,
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
        model: Transformer model for evaluation.
        test_loader: DataLoader for test dataset.
        transformation: Transformation module to be evaluated.
        exact_match: If True, requires exact matches between predicted and actual
            embeddings. If False, matches are correct if identical ignoring case
            differences.
        device: Model's device. Defaults to None.
        print_results: If True, prints translation attempts/results. Defaults to False.
        print_top_preds: If True and print_results=True, prints top predictions.
            Defaults to True.
        print_acc: If True, prints the correct percentage. Defaults to True.

    Returns:
        The accuracy of the learned transformation as a float.
    """
    with t.no_grad():
        correct_count = 0
        total_count = 0
        for batch in test_loader:
            en_embeds, fr_embeds = batch
            en_logits = model.unembed(en_embeds)
            en_strs: List[str] = model.to_str_tokens(en_logits.argmax(dim=-1))  # type: ignore
            fr_logits = model.unembed(fr_embeds)
            fr_strs: List[str] = model.to_str_tokens(fr_logits.argmax(dim=-1))  # type: ignore
            with t.no_grad():
                pred = transformation(en_embeds)
            pred_logits = model.unembed(pred)
            pred_top_strs = model.to_str_tokens(pred_logits.argmax(dim=-1))
            pred_top_strs = [
                item if isinstance(item, str) else item[0] for item in pred_top_strs
            ]
            assert all(isinstance(item, str) for item in pred_top_strs)
            most_similar_embeds = get_most_similar_embeddings(
                model,
                out=pred,
                top_k=4,
                apply_embed=True,
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
        if print_acc:  # New condition to print accuracy
            print(f"Correct Percentage: {accuracy * 100:.2f}%")
    return accuracy


def mark_translation(
    model: tl.HookedTransformer,
    transformation: nn.Module,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    azure_translations_path: Optional[Union[str, Path]] = None,
    print_results: bool = False,
    print_top_preds: bool = True,
    translations_dict: Optional[Dict[str, List[str]]] = None,
) -> float:
    """Marks translations as correct.

    Can either take in a dictionary of translations or a path to a JSON file containing
    translations from Azure. At least one of `azure_translations_path` or
    `translations_dict` must be provided.

    Args:
        model: The model whose tokenizer we are using.
        transformation: The transformation module to evaluate.
        test_loader: DataLoader for the test dataset.
        azure_translations_path: Optional; Path to the JSON file containing acceptable
            translations from Azure Translator. Defaults to None.
        print_results: Whether to print marking results.
        print_top_preds: If True and print_results=True, prints top predictions.
            Defaults to True.
        translations_dict: Optional; A dictionary of translations. Defaults to None.
    Returns:
        The accuracy of the translations as a float.

    Raises:
        ValueError: If neither `azure_translations_path` nor `translations_dict`
            is provided.
    """
    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    if azure_translations_path is None and translations_dict is None:
        raise ValueError(
            "Either 'azure_translations_path' or 'translations_dict' must be given."
        )
    # load acceptable translations from JSON file if given
    if azure_translations_path:
        with open(azure_translations_path, "r") as file:
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

    with t.no_grad():
        for batch in test_loader:
            en_embeds, fr_embeds = batch
            en_logits = model.unembed(en_embeds)
            en_strs: List[str] = model.tokenizer.batch_decode(
                en_logits.squeeze().argmax(dim=-1)
            )
            fr_logits = model.unembed(fr_embeds)
            fr_strs: List[str] = model.tokenizer.batch_decode(
                fr_logits.squeeze().argmax(dim=-1)
            )
            with t.no_grad():
                pred = transformation(en_embeds)
            pred_logits = model.unembed(pred)
            pred_top_strs = model.tokenizer.batch_decode(
                pred_logits.squeeze().argmax(dim=-1)
            )
            pred_top_strs = [
                item if isinstance(item, str) else item[0] for item in pred_top_strs
            ]
            assert all(isinstance(item, str) for item in pred_top_strs)
            # if statement for performance
            if print_top_preds:
                most_similar_embeds = get_most_similar_embeddings(
                    model,
                    out=pred,
                    top_k=4,
                    apply_embed=True,
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
    """
    Calculates the canonical angles between two matrices to measure their rotational
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
