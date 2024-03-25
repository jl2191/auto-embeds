import difflib
import json
from collections import defaultdict
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import einops
import plotly.express as px
import torch as t
import torch.nn as nn
import transformer_lens as tl
from Levenshtein import distance as levenshtein_distance
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from auto_embeds.data import get_dataset_path
from auto_embeds.modules import (
    BiasedRotationTransform,
    CosineSimilarityLoss,
    MeanTranslationTransform,
    MSELoss,
    RotationTransform,
    TranslationTransform,
    UncenteredLinearMapTransform,
    UncenteredRotationTransform,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.misc import (
    get_default_device,
    get_most_similar_embeddings,
    print_most_similar_embeddings_dict,
    remove_hooks,
)

default_device = get_default_device()


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
    device: Optional[Union[str, t.device]] = default_device,
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
            model.cfg.device.

    Returns:
        A tuple containing the transformation module and its optimizer.
    """
    transform_kwargs["device"] = device

    if transformation == "identity":
        transform = nn.Identity(**transform_kwargs)
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
        transform = nn.Linear(d_model, d_model, bias=False, **transform_kwargs)
        optim = t.optim.Adam(transform.parameters(), **optim_kwargs)

    elif transformation == "biased_linear_map":
        transform = nn.Linear(d_model, d_model, bias=True, **transform_kwargs)
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


def word_distance_metric(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Computes the negative cosine similarity between two tensors.

    Args:
        a: The first tensor.
        b: The second tensor.

    Returns:
        The negative cosine similarity between the input tensors.
    """
    return -nn.functional.cosine_similarity(a, b, -1)


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
    device: Optional[Union[str, t.device]] = None,
    wandb: Optional[Any] = None,
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

    Returns:
        The learned transformation after training, the train and test loss history.
    """
    if device is None:
        device = model.cfg.device
    loss_history = {"train_loss": [], "test_loss": []}
    transform.train()
    if wandb:
        wandb.watch(transform, log="all", log_freq=500)
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
            loss_history["train_loss"].append(info_dict)
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
            loss_history["test_loss"].append(info_dict)
            if wandb:
                wandb.log(info_dict)
    if plot_fig or save_fig:
        fig = px.line(title="Train and Test Loss")
        fig.add_scatter(
            x=[epoch_info["epoch"] for epoch_info in loss_history["train_loss"]],
            y=[epoch_info["train_loss"] for epoch_info in loss_history["train_loss"]],
            name="Train Loss",
        )
        fig.add_scatter(
            x=[epoch_info["epoch"] for epoch_info in loss_history["test_loss"]],
            y=[epoch_info["test_loss"] for epoch_info in loss_history["test_loss"]],
            name="Test Loss",
        )
        if plot_fig:
            fig.show()
        if save_fig:
            fig.write_image("plot.png")
    return transform, loss_history


def evaluate_accuracy(
    model: tl.HookedTransformer,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transformation: nn.Module,
    exact_match: bool,
    device: Optional[Union[str, t.device]] = default_device,
    print_results: bool = False,
    print_top_preds: bool = True,
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
    2       print_top_preds: If True and print_results=True, prints top predictions.
                Defaults to True.

        Returns:
            The accuracy of the learned transformation as a float.
    """
    with t.no_grad():
        correct_count = 0
        total_count = 0
        for batch in test_loader:
            en_embeds, fr_embeds = batch
            en_logits = einops.einsum(
                en_embeds,
                model.embed.W_E,
                "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
            )
            en_strs: List[str] = model.to_str_tokens(en_logits.argmax(dim=-1))  # type: ignore
            fr_logits = einops.einsum(
                fr_embeds,
                model.embed.W_E,
                "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
            )
            fr_strs: List[str] = model.to_str_tokens(fr_logits.argmax(dim=-1))  # type: ignore
            with t.no_grad():
                pred = transformation(en_embeds)
            pred_logits = einops.einsum(
                pred,
                model.embed.W_E,
                "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
            )
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
    return accuracy


def calc_cos_sim_acc(
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transform: nn.Module,
    device: Optional[Union[str, t.device]] = default_device,
) -> float:
    """Calculate the cosine similarity accuracy between predicted and actual embeddings.

    Args:
        test_loader: DataLoader for the testing dataset.
        transform: The transformation module to be evaluated.
        device: The device to perform calculations on.

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
    return t.cat(cosine_sims).mean().item()


def filter_word_pairs(
    model: tl.HookedTransformer,
    word_pairs: List[List[str]],
    max_token_id: Optional[int] = None,
    discard_if_same: bool = False,
    min_length: int = 1,
    capture_diff_case: bool = False,
    capture_space: bool = True,
    capture_no_space: bool = False,
    print_pairs: bool = False,
    print_number: bool = False,
    verbose_count: bool = False,
    most_common_english: bool = False,
    most_common_french: bool = False,
    acceptable_english_overlap: Optional[float] = 1.0,
    acceptable_french_overlap: Optional[float] = 1.0,
) -> List[List[str]]:
    """Filters and tokenizes Source-Target word pairs.

    This function filters the input word pairs, retaining only those that result in
    single-token outputs upon tokenization.

    Args:
        model: The model equipped with a tokenizer for processing the texts.
        word_pairs: A list containing pairs of words to be tokenized.
        max_token_id: Filters out words with a tokenized ID above this threshold.
        discard_if_same: Exclude word pairs that are identical.
        min_length: Sets the minimum text length eligible for tokenization.
        capture_diff_case: Includes text variations with different capitalizations.
        capture_space: Prepends a space to the text before tokenization.
        capture_no_space: Tokenizes the text without adding a leading space.
        print_pairs: Enables printing of each word pair processed.
        print_number: Outputs the total count of word pairs processed.
        verbose_count: Outputs the count of word pairs at each filtering step.
        most_common_english: When true, prefers the translation pair with the lowest
            aggregate token ID in cases of multiple translations for English.
        most_common_french: When true, prefers the translation pair with the lowest
            aggregate token ID in cases of multiple translations for French.
        acceptable_overlap: The maximum acceptable similarity between word pairs before
            they are merged by taking the one with the lowest aggregate token ID. Can
            be used with most_common_english and most_common_french for performance
            reasons as the metric used, the Levenshtein distance can be quite slow to
            compute so it is good to filter out words that are identical first.

    Returns:
        A list of filtered word pairs that tokenize into single tokens.
    """
    if max_token_id is None:
        max_token_id = model.cfg.d_vocab
    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    print(f"Initial length: {len(word_pairs)}")

    if discard_if_same:
        word_pairs = [pair for pair in word_pairs if pair[0].lower() != pair[1].lower()]
        if verbose_count:
            print(f"After discard_if_same: {len(word_pairs)}")

    word_pairs = [
        pair
        for pair in word_pairs
        if len(pair[0]) >= min_length and len(pair[1]) >= min_length
    ]

    if capture_diff_case:
        diff_case_pairs = []
        for pair in word_pairs:
            diff_case_pairs.append([pair[0], pair[1]])
            diff_case_pairs.append([pair[0].capitalize(), pair[1]])
            diff_case_pairs.append([pair[0], pair[1].capitalize()])
            diff_case_pairs.append([pair[0].capitalize(), pair[1].capitalize()])
        word_pairs = diff_case_pairs
        if verbose_count:
            print(f"After capture_diff_case: {len(word_pairs)}")

    pairs_to_filter = []

    if capture_no_space:
        pairs_to_filter.extend(word_pairs)
        if verbose_count:
            print(f"After capture/no_space: {len(pairs_to_filter)}")

    if capture_space:
        word_pairs_w_space = [[f" {pair[0]}", f" {pair[1]}"] for pair in word_pairs]
        pairs_to_filter.extend(word_pairs_w_space)
        if verbose_count:
            print(f"After capture_space: {len(pairs_to_filter)}")

    english_words, french_words = [
        list(words) for words in zip(*pairs_to_filter, strict=True)
    ]
    en_tokens = model.tokenizer(english_words, add_special_tokens=False).data[
        "input_ids"
    ]
    fr_tokens = model.tokenizer(french_words, add_special_tokens=False).data[
        "input_ids"
    ]

    # overwrites pairs_to_filter so we have the word tokens as well
    pairs_to_filter_with_tokens = [
        [en_tokens, fr_tokens, word_pair]
        for en_tokens, fr_tokens, word_pair in zip(
            en_tokens, fr_tokens, pairs_to_filter
        )
    ]

    pairs_to_filter = pairs_to_filter_with_tokens

    pairs_to_filter = [
        pair for pair in pairs_to_filter if all(len(token) == 1 for token in pair[:2])
    ]
    if verbose_count:
        print(f"After filtering for single tokens only: {len(pairs_to_filter)}")

    pairs_to_filter = [
        pair for pair in pairs_to_filter if max(pair[0][0], pair[1][0]) < max_token_id
    ]

    if verbose_count:
        print(f"After max_token_id: {len(pairs_to_filter)}")

    # add on token sums without removing existing information
    pairs_to_filter = [
        [sum(word_pair[0] + word_pair[1]), *word_pair] for word_pair in pairs_to_filter
    ]

    if most_common_english:
        most_common = {}
        for (
            token_sum_current,
            en_token,
            fr_token,
            (en_word, fr_word),
        ) in pairs_to_filter:
            if en_word in most_common:
                token_sum_existing, _, _, _ = most_common[en_word]
                if token_sum_current < token_sum_existing:
                    most_common[en_word] = (
                        token_sum_current,
                        fr_word,
                        en_token,
                        fr_token,
                    )
            else:
                most_common[en_word] = (
                    token_sum_current,
                    fr_word,
                    en_token,
                    fr_token,
                )
        pairs_to_filter = [
            [token_sum, en_token, fr_token, [en_word, fr_word]]
            for en_word, (
                token_sum,
                fr_word,
                en_token,
                fr_token,
            ) in most_common.items()
        ]
        if verbose_count:
            print(f"After most_common_english: {len(pairs_to_filter)}")

    if most_common_french:
        most_common = {}
        for (
            token_sum_current,
            en_token,
            fr_token,
            (en_word, fr_word),
        ) in pairs_to_filter:
            if fr_word in most_common:
                token_sum_existing, _, _, _ = most_common[fr_word]
                if token_sum_current < token_sum_existing:
                    most_common[fr_word] = (
                        token_sum_current,
                        en_word,
                        en_token,
                        fr_token,
                    )
            else:
                most_common[fr_word] = (
                    token_sum_current,
                    en_word,
                    en_token,
                    fr_token,
                )
        pairs_to_filter = [
            [token_sum, en_token, fr_token, [en_word, fr_word]]
            for en_word, (
                token_sum,
                fr_word,
                en_token,
                fr_token,
            ) in most_common.items()
        ]
        if verbose_count:
            print(f"After most_common_french: {len(pairs_to_filter)}")

    if acceptable_english_overlap != 1.0:
        filtered_pairs = []
        while pairs_to_filter:
            current_pair = pairs_to_filter.pop(0)
            current_id_sum, current_en_token, current_fr_token, current_words = (
                current_pair
            )
            current_en_word, _ = current_words
            most_similar_pair = current_pair
            most_similar_id_sum = current_id_sum

            for other_pair in pairs_to_filter[:]:
                other_id_sum, _, _, other_words = other_pair
                other_en_word, _ = other_words
                en_similarity_ratio = 1 - levenshtein_distance(
                    current_en_word, other_en_word
                ) / max(len(current_en_word), len(other_en_word))
                if (
                    en_similarity_ratio >= acceptable_english_overlap
                    and other_id_sum < most_similar_id_sum
                ):
                    most_similar_pair = other_pair
                    most_similar_id_sum = other_id_sum
                    pairs_to_filter.remove(other_pair)
            filtered_pairs.append(most_similar_pair)
        if verbose_count:
            print(f"After acceptable_english_overlap: {len(filtered_pairs)}")

        pairs_to_filter = filtered_pairs

    if acceptable_french_overlap != 1.0:
        filtered_pairs = []
        while pairs_to_filter:
            current_pair = pairs_to_filter.pop(0)
            current_id_sum, current_en_token, current_fr_token, current_words = (
                current_pair
            )
            _, current_fr_word = current_words
            most_similar_pair = current_pair
            most_similar_id_sum = current_id_sum

            for other_pair in pairs_to_filter[:]:
                other_id_sum, _, _, other_words = other_pair
                _, other_fr_word = other_words
                fr_similarity_ratio = 1 - levenshtein_distance(
                    current_fr_word, other_fr_word
                ) / max(len(current_fr_word), len(other_fr_word))
                if (
                    fr_similarity_ratio >= acceptable_french_overlap
                    and other_id_sum < most_similar_id_sum
                ):
                    most_similar_pair = other_pair
                    most_similar_id_sum = other_id_sum
                    pairs_to_filter.remove(other_pair)
            filtered_pairs.append(most_similar_pair)
        if verbose_count:
            print(f"After acceptable_french_overlap: {len(filtered_pairs)}")

        pairs_to_filter = filtered_pairs

    # extracting just the word pairs out again (discarding id_sum and token_ids)
    filtered_pairs = [
        word_pair for [id_sum, en_tokens, fr_tokens, word_pair] in pairs_to_filter
    ]
    word_pairs = filtered_pairs

    if print_pairs:
        for pair in word_pairs:
            print(f"English: {pair[0]}, French: {pair[1]}")
    if print_number:
        print(f"Total word pairs: {len(word_pairs)}")

    return word_pairs


def tokenize_word_pairs(
    model: tl.HookedTransformer, word_pairs: List[List[str]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Converts a list of word pairs into tensors suitable for model input.

    Args:
        model: A HookedTransformer model instance with a tokenizer.
        word_pairs: A list of word pairs, where each pair is a list of two strings.

    Returns:
        A tuple containing four tensors:
        - en_tokens: Tokenized English words.
        - fr_tokens: Tokenized French words.
        - en_mask: Attention mask for English tokens.
        - fr_mask: Attention mask for French tokens.
    """
    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    english_words, french_words = zip(*word_pairs)
    combined_texts = list(english_words) + list(french_words)
    # print(combined_texts)

    tokenized = model.tokenizer(
        combined_texts, padding="longest", return_tensors="pt", add_special_tokens=False
    )
    num_pairs = tokenized.input_ids.shape[0]
    assert num_pairs % 2 == 0
    word_each = num_pairs // 2
    tokens = tokenized.data["input_ids"]
    attn_masks = tokenized.data["attention_mask"]
    en_tokens, fr_tokens = tokens[:word_each], tokens[word_each:]
    en_mask, fr_mask = attn_masks[:word_each], attn_masks[word_each:]

    return en_tokens, fr_tokens, en_mask, fr_mask


def run_and_gather_acts(
    model: tl.HookedTransformer,
    dataloader: DataLoader[Tuple[Tensor, ...]],
    layers: List[int],
) -> Tuple[Dict[int, List[Tensor]], Dict[int, List[Tensor]]]:
    """Gathers layer-specific embeddings for English and French text batches.

    This function processes batches of English and French text embeddings, extracting
    activations from the specified layers of the transformer model.

    Args:
        model: The transformer model used for gathering activations.
        dataloader: The dataloader with batches of English and French text embeddings.
        layers: List of integers specifying the layers for gathering embeddings.

    Returns:
        Two dicts containing lists of embeddings for English and French texts,
        separated by layer.
    """
    en_embeds, fr_embeds = defaultdict(list), defaultdict(list)
    for en_batch, fr_batch, en_mask, fr_mask in tqdm(dataloader):
        with t.inference_mode():
            _, en_cache = model.run_with_cache(en_batch, prepend_bos=True)
            for layer in layers:
                en_resids = en_cache[f"blocks.{layer}.hook_resid_pre"]
                filtered_en_resids = en_resids[en_mask == 1]
                en_embeds[layer].append(filtered_en_resids.detach().clone().cpu())
            del en_cache

            _, fr_cache = model.run_with_cache(fr_batch, prepend_bos=True)
            for layer in layers:
                fr_resids = fr_cache[f"blocks.{layer}.hook_resid_pre"]
                filtered_fr_resids = fr_resids[fr_mask == 1]
                fr_embeds[layer].append(filtered_fr_resids.detach().clone().cpu())
            del fr_cache
    en_embeds = dict(en_embeds)
    fr_embeds = dict(fr_embeds)
    return en_embeds, fr_embeds


def save_acts(
    cache_folder: Union[str, Path],
    filename_base: str,
    en_acts: Dict[int, List[t.Tensor]],
    fr_acts: Dict[int, List[t.Tensor]],
):
    """Saves model activations, separated by layer, to the specified cache folder.

    Args:
        cache_folder: The folder path where the activations will be saved.
        filename_base: The base name for the saved files.
        en_acts: A dict containing lists of English embeddings, separated by layer.
        fr_acts: A dict containing lists of French embeddings, separated by layer.
    """
    en_layers = [layer for layer in en_acts]
    fr_layers = [layer for layer in fr_acts]
    t.save(en_acts, f"{cache_folder}/{filename_base}-en-layers-{en_layers}.pt")
    t.save(fr_acts, f"{cache_folder}/{filename_base}-fr-layers-{fr_layers}.pt")


def mean_vec(train_en_resids: t.Tensor, train_fr_resids: t.Tensor) -> t.Tensor:
    """Calculates the mean vector difference between English and French residuals.

    Args:
        train_en_resids: The tensor containing English residuals.
        train_fr_resids: The tensor containing French residuals.

    Returns:
        The mean vector difference between English and French residuals.
    """
    return train_en_resids.mean(dim=0) - train_fr_resids.mean(dim=0)


def perform_steering_tests(
    model: nn.Module,
    en_strs: List[str],
    fr_strs: List[str],
    layer_idx: int,
    gen_length: int,
    transformation: nn.Module,
    positions_to_steer: str,
    num_tests: int = 3,
) -> None:
    """Performs steering tests on given English and French strings.

    This function tests the effect of steering on model-generated continuations of
    English strings towards their French translations. It compares the model's
    continuations before and after applying the steering transformation.

    Args:
        model: The model to perform steering tests on.
        en_strs: A list of English strings to generate continuations for.
        fr_strs: A list of corresponding French strings.
        layer_idx: The index of the layer at which to apply the steering transformation.
        gen_length: The number of tokens to generate for each continuation.
        transformation: The transformation module to apply for steering.
        positions_to_steer: Specifies which positions in the sequence to apply the
            transformation to. Can be 'all' or 'final'.
        num_tests: The number of tests to perform. Defaults to 3.

    """
    for idx, (test_en_str, test_fr_str) in enumerate(zip(en_strs, fr_strs)):
        if idx >= num_tests:
            break
        print("\n----------------------------------------------")
        print("original:", test_en_str)
        initial_len = len(test_en_str)
        for _ in range(gen_length):
            top_tok = model(test_en_str, prepend_bos=True)[:, -1].argmax(dim=-1)
            top_tok_str = model.to_string(top_tok)
            test_en_str += top_tok_str
        print("model continuation:", test_en_str[initial_len:])
        with remove_hooks() as handles, t.inference_mode():
            handle = model.blocks[layer_idx].hook_resid_pre.register_forward_hook(
                partial(
                    steering_hook,
                    transformation=transformation,
                    positions_to_steer=positions_to_steer,
                )
            )
            handles.add(handle)
            for _ in range(gen_length):
                top_tok = model(test_en_str, prepend_bos=True)[:, -1].argmax(dim=-1)
                top_tok_str = model.to_string(top_tok)
                test_en_str += top_tok_str
            print("steered model continuation:", test_en_str[initial_len:])


def steering_hook(
    module: nn.Module,
    input: Tuple[t.Tensor],
    output: t.Tensor,
    transformation: nn.Module,
    positions_to_steer: str,
) -> t.Tensor:
    # input is a tuple containing 1 tensor of shape[batch, pos, d_model]
    # prefix_toks, final_tok = input[0][:, :-1], input[0][:, -1]
    # prefix_toks is of shape [1, 100, 1024]
    # final_tok is of shape [1, 1024]
    # rotated_final_tok = transformation(final_tok)
    # out = t.cat([prefix_toks, rotated_final_tok.unsqueeze(1)], dim=1)
    # out is of shape []
    if positions_to_steer == "all":
        out = transformation(input[0])
    elif positions_to_steer == "final":
        prefix_toks, final_tok = input[0][:, :-1], input[0][:, -1]
        # prefix_toks is of shape [1, 100, 1024]
        # final_tok is of shape [1, 1024]
        transformed_final_tok = transformation(final_tok)
        out = t.cat([prefix_toks, transformed_final_tok.unsqueeze(1)], dim=1)
    else:
        raise ValueError(f"Unsupported positions_to_steer value: {positions_to_steer}")
    return out


def load_test_strings(file_path: Union[str, Path], skip_lines: int) -> List[str]:
    """Loads test strings from a file, skipping the first `skip_lines` lines.

    Args:
        file_path: The path to the file from which to load test strings.
        skip_lines: The number of lines to skip before loading test strings.

    Returns:
        A list of test strings loaded from the file.
    """
    test_strs = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= skip_lines:
                next_line = next(f, "").strip()
                test_strs.append(line.strip() + " " + next_line)
    return test_strs


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


def mark_correct(
    model: tl.HookedTransformer,
    transformation: nn.Module,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    acceptable_translations_path: Union[str, Path],
    print_results: bool = False,
    print_top_preds: bool = True,
) -> float:
    """Marks translations as correct from an Azure JSON file.

    Args:
        model: The model whose tokenizer we are using.
        transformation: The transformation module to evaluate.
        test_loader: DataLoader for the test dataset.
        acceptable_translations_path: Path to the JSON file containing acceptable
            translations from Azure Translator.
        print_results: Whether to print marking results.
        print_top_preds: If True and print_results=True, prints top predictions.
            Defaults to True.
    Returns:
        The accuracy of the translations as a float.
    """
    # Load acceptable translations from JSON file
    with open(acceptable_translations_path, "r") as file:
        acceptable_translations = json.load(file)

    # Convert list of acceptable translations to a more accessible format
    translations_dict = {}
    for item in acceptable_translations:
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
            en_logits = einops.einsum(
                en_embeds,
                model.embed.W_E,
                "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
            )
            en_strs: List[str] = model.to_str_tokens(en_logits.argmax(dim=-1))  # type: ignore
            fr_logits = einops.einsum(
                fr_embeds,
                model.embed.W_E,
                "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
            )
            fr_strs: List[str] = model.to_str_tokens(fr_logits.argmax(dim=-1))  # type: ignore
            with t.no_grad():
                pred = transformation(en_embeds)
            pred_logits = einops.einsum(
                pred,
                model.embed.W_E,
                "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
            )
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
                correct = None
                en_str = en_strs[i]
                fr_str = fr_strs[i]
                word_found = en_str.strip() in translations_dict
                if word_found:
                    all_acceptable_translations = translations_dict[en_str.strip()]
                    correct = (
                        pred_top_str.strip() in all_acceptable_translations
                        or pred_top_str.strip() + "s" in all_acceptable_translations
                        or pred_top_str.strip()[:-1] in all_acceptable_translations
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
                            f"Check: Found {[target for target in translations_dict[en_str.strip()]]}"
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
