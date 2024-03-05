from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import plotly.express as px
import torch as t
import torch.nn as nn
import transformer_lens as tl
from fancy_einsum import einsum
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from auto_steer.modules import (
    MeanTranslationTransform,
    OffsetRotationTransform,
    RotationTransform,
    TranslationTransform,
    UncenteredLinearMapTransform,
    UncenteredRotationTransform,
)
from auto_steer.utils.custom_tqdm import tqdm
from auto_steer.utils.misc import (
    get_default_device,
    get_most_similar_embeddings,
    print_most_similar_embeddings_dict,
    remove_hooks,
)

default_device = get_default_device()


def initialize_transform_and_optim(
    d_model: int,
    transformation: str,
    mean_diff: Optional[Tensor] = None,
    transform_kwargs: Dict[str, Any] = {},
    optim_kwargs: Dict[str, Any] = {},
    device: Optional[Union[str, t.device]] = default_device,
) -> Tuple[nn.Module, Optional[Optimizer]]:
    """
    Initializes a transformation and its corresponding optimizer based on the specified
    transformation type, with additional flexibility provided by explicit dictionaries
    for both the transformation and optimizer configurations.

    Args:
        d_model: The dimensionality of the model embeddings.
        transformation: The type of transformation to initialize.
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
            raise ValueError(
                (
                    "The mean difference tensor must be provided "
                    "for mean-centered steering transformation."
                )
            )
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

    elif transformation == "offset_rotation":
        transform = OffsetRotationTransform(d_model, **transform_kwargs)
        optim = t.optim.Adam(list(transform.parameters()), **optim_kwargs)

    elif transformation == "uncentered_rotation":
        transform = UncenteredRotationTransform(d_model, **transform_kwargs)
        optim = t.optim.Adam(list(transform.parameters()), **optim_kwargs)
    else:
        raise Exception("the supplied transform was unrecognized")
    return transform, optim


def word_distance_metric(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """
    Computes the negative cosine similarity between two tensors.

    Args:
        a (t.Tensor): The first tensor.
        b (t.Tensor): The second tensor.

    Returns:
        t.Tensor: The negative cosine similarity between the input tensors.
    """
    return -nn.functional.cosine_similarity(a, b, -1)


def train_transform(
    model: tl.HookedTransformer,
    train_loader: DataLoader[Tuple[Tensor, ...]],
    transform: nn.Module,
    optim: Optimizer,
    n_epochs: int,
    device: Optional[Union[str, t.device]] = None,
    wandb: Optional[Any] = None,
) -> Tuple[nn.Module, List[float]]:
    """
    Trains the transformation, returning the learned transformation and loss history.

    Args:
        model: The transformer model used for training.
        device: The device on which the model is allocated.
        train_loader: DataLoader for the training dataset.
        transform: The transformation module to be optimized.
        optim: The optimizer for the transformation.
        n_epochs: The number of epochs to train for.
        wandb: If provided, log training metrics to Weights & Biases.

    Returns:
        The learned transformation after training and the loss history.
    """
    if device is None:
        device = model.cfg.device
    loss_history = []
    transform.to(device)
    for epoch in (epoch_pbar := tqdm(range(n_epochs))):
        for batch_idx, (en_embed, fr_embed) in enumerate(train_loader):
            en_embed, fr_embed = en_embed.to(device), fr_embed.to(device)
            optim.zero_grad()
            pred = transform(en_embed)
            loss = word_distance_metric(pred, fr_embed).mean()
            loss_history.append(loss.item())
            loss.backward()
            optim.step()
            if wandb:
                wandb.log({"epoch": epoch, "loss": loss.item(), "batch_idx": batch_idx})
            epoch_pbar.set_description(f"Loss: {loss.item():.3f}")
    px.line(y=loss_history, title="Loss History").show()
    return transform, loss_history


def evaluate_accuracy(
    model: tl.HookedTransformer,
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transformation: nn.Module,
    exact_match: bool,
    device: Optional[Union[str, t.device]] = None,
    print_results: bool = False,
) -> float:
    """
    Evaluates the accuracy of the learned transformation by comparing the predicted
    embeddings to the actual French embeddings. It supports requiring exact matches
    or allowing for case-insensitive comparisons.

    Args:
        model: Transformer model for evaluation.
        test_loader: DataLoader for test dataset.
        transformation: Transformation module to be evaluated.
        exact_match: If True, requires exact matches between predicted and actual
        embeddings. If False, matches are correct if identical ignoring case
        differences.
        device: Model's device. Defaults to None.
        print_results: If True, prints translation attempts/results. Defaults to False.

    Returns:
        The accuracy of the learned transformation as a float.
    """
    if device is None:
        device = model.cfg.device
    correct_count = 0
    total_count = 0
    for batch in test_loader:
        en_embeds, fr_embeds = batch
        en_embeds = en_embeds.to(device)
        fr_embeds = fr_embeds.to(device)

        en_logits = einsum(
            en_embeds,
            model.embed.W_E,
            "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
        )
        en_strs: List[str] = model.to_str_tokens(en_logits.argmax(dim=-1))  # type: ignore
        fr_logits = einsum(
            fr_embeds,
            model.embed.W_E,
            "batch pos d_model, d_vocab d_model -> batch pos d_vocab",
        )
        fr_strs: List[str] = model.to_str_tokens(fr_logits.argmax(dim=-1))  # type: ignore

        pred = transformation(en_embeds)
        pred_logits = einsum(
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

        # print(most_similar_embeds)
        # print_most_similar_embeddings_dict(most_similar_embeds)

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
                    f"English: {en_str}\n"
                    f"French: {fr_str}\n"
                    f"Predicted: {pred_top_str} {result_emoji}"
                )
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
    device: Optional[str] = None,
) -> float:
    """
    Calculates the cosine similarity accuracy between predicted and actual embeddings.

    Args:
        test_loader: DataLoader for the testing dataset.
        transform: The transformation module to be evaluated.
        device: The device to perform calculations on.

    Returns:
        The mean cosine similarity accuracy.
    """
    if device is not None:
        transform.to(device)
    cosine_sims = []
    for batch_idx, (en_embed, fr_embed) in enumerate(test_loader):
        en_embed = en_embed.to(device)
        fr_embed = fr_embed.to(device)
        pred = transform(en_embed)
        cosine_sim = word_distance_metric(pred, fr_embed)
        cosine_sims.append(cosine_sim)
    return t.cat(cosine_sims).mean().item()


# %% ----------------------- functions --------------------------


def read_file_lines(file_path: Union[str, Path], lines_count: int = 5000) -> List[str]:
    """
    Reads the specified number of lines from a file, excluding the first line.
    Each line read is concatenated with the next line, separated by a space.

    Args:
        file_path: The path to the file to be read.
        lines_count: The number of lines to read from the file.

    Returns:
        A list of concatenated lines read from the file.
    """
    with open(file_path, "r") as file:
        return [
            file.readline().strip() + " " + file.readline().strip()
            for _ in range(lines_count + 1)
        ][1:]


def tokenize_texts(
    model: tl.HookedTransformer,
    word_pairs: List[List[str]],
    padding_side: str = "right",
    pad_to_same_length: bool = True,
    padding_strategy: str = "longest",
    single_tokens_only: bool = False,
    discard_if_same: bool = False,
    min_length: int = 1,
    capture_diff_case: bool = False,
    capture_space: bool = True,
    capture_no_space: bool = True,
) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """
    Tokenizes texts and returns tensors for English and French tokens and masks.

    Args:
        model: The model with a tokenizer to process the texts.
        word_pairs: A list of word pairs as lists to be tokenized.
        padding_side: The side ('right' or 'left') to apply padding.
        pad_to_same_length: Whether to pad texts to the same length.
        padding_strategy: The strategy ('longest', 'max_length', etc.) for padding.
        single_tokens_only: Whether to filter for single-token texts only.
        discard_if_same: Whether to discard text pairs that are identical.
        min_length: The minimum length of text to be considered for tokenization.
        capture_diff_case: Whether to include different casing variations of texts.
        capture_space: Whether to include a space in front of the text.
        capture_no_space: Whether to include the text without a leading space.

    Returns:
        A tuple of tensors for tokenized English texts, their attention masks,
        tokenized French texts, and their attention masks.
    """
    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    model.tokenizer.padding_side = padding_side

    if discard_if_same:
        word_pairs = [pair for pair in word_pairs if pair[0] != pair[1]]

    word_pairs = [
        pair
        for pair in word_pairs
        if len(pair[0]) >= min_length and len(pair[1]) >= min_length
    ]

    if capture_diff_case:
        diff_case_texts = []
        for pair in word_pairs:
            diff_case_texts.append([pair[0], pair[1]])
            diff_case_texts.append([pair[0].capitalize(), pair[1]])
            diff_case_texts.append([pair[0], pair[1].capitalize()])
            diff_case_texts.append([pair[0].capitalize(), pair[1].capitalize()])
        word_pairs = diff_case_texts

    if single_tokens_only:
        pairs_to_filter = []
        filtered_pairs = []
        if capture_no_space:
            pairs_to_filter.extend(word_pairs)
        if capture_space:
            word_pairs_w_space = [[f" {pair[0]}", f" {pair[1]}"] for pair in word_pairs]
            pairs_to_filter.extend(word_pairs_w_space)
        english_words, french_words = [list(words) for words in zip(*pairs_to_filter)]
        en_word_tokens = model.tokenizer(english_words).data["input_ids"]
        fr_word_tokens = model.tokenizer(french_words).data["input_ids"]
        for en_word_tokens, fr_word_tokens, word_pair in zip(
            en_word_tokens, fr_word_tokens, pairs_to_filter
        ):
            if len(en_word_tokens) == 1 and len(fr_word_tokens) == 1:
                filtered_pairs.append(word_pair)
        word_pairs = filtered_pairs

    english_words, french_words = zip(*word_pairs)
    combined_texts = list(english_words) + list(french_words)
    # print(combined_texts[:30])
    # print(len(combined_texts))

    tokenized = model.tokenizer(combined_texts, padding="longest", return_tensors="pt")
    num_pairs = tokenized.input_ids.shape[0]
    assert num_pairs % 2 == 0
    word_each = num_pairs // 2
    tokens = tokenized.data["input_ids"]
    attn_masks = tokenized.data["attention_mask"]
    en_word_tokens, fr_word_tokens = tokens[:word_each], tokens[word_each:]
    en_attn_masks, fr_attn_masks = attn_masks[:word_each], attn_masks[word_each:]

    return en_word_tokens, en_attn_masks, fr_word_tokens, fr_attn_masks


def run_and_gather_acts(
    model: tl.HookedTransformer,
    dataloader: DataLoader[Tuple[Tensor, ...]],
    layers: List[int],
) -> Tuple[Dict[int, List[Tensor]], Dict[int, List[Tensor]]]:
    """
    Runs the model on batches of English and French text embeddings from the dataloader
    and gathers embeddings from specified layers.

    Args:
        model: The transformer model used for gathering activations.
        dataloader: The dataloader with batches of English and French text embeddings.
        layers: List of integers specifying the layers for gathering embeddings.

    Returns:
        Two dicts containing lists of embeddings for English and French texts,
        separated by layer.
    """
    en_embeds, fr_embeds = defaultdict(list), defaultdict(list)
    for en_batch, fr_batch, en_attn_mask, fr_attn_mask in tqdm(dataloader):
        with t.inference_mode():
            _, en_cache = model.run_with_cache(en_batch, prepend_bos=True)
            _, fr_cache = model.run_with_cache(fr_batch, prepend_bos=True)
            for layer in layers:
                en_resids = en_cache[f"blocks.{layer}.hook_resid_pre"]
                en_resids_flat = en_resids.flatten(start_dim=0, end_dim=1)
                en_mask_flat = en_attn_mask.flatten(start_dim=0, end_dim=1)
                filtered_en_resids = en_resids_flat[en_mask_flat == 1]
                en_embeds[layer].append(filtered_en_resids.detach().clone().cpu())

                fr_resids = fr_cache[f"blocks.{layer}.hook_resid_pre"]
                fr_resids_flat = fr_resids.flatten(start_dim=0, end_dim=1)
                fr_mask_flat = fr_attn_mask.flatten(start_dim=0, end_dim=1)
                filtered_fr_resids = fr_resids_flat[fr_mask_flat == 1]
                fr_embeds[layer].append(filtered_fr_resids.detach().clone().cpu())
    en_embeds = dict(en_embeds)
    fr_embeds = dict(fr_embeds)
    return en_embeds, fr_embeds


def save_acts(
    cache_folder: Union[str, Path],
    filename_base: str,
    en_acts: Dict[int, List[t.Tensor]],
    fr_acts: Dict[int, List[t.Tensor]],
):
    """
    Saves model activations, separated by layer, to the specified cache folder.

    Args:
        cache_folder: The folder path where the activations will be saved.
        filename_base : The base name for the saved files.
        en_acts: A dict containing lists of English embeddings, separated by layer.
        fr_acts: A dict containing lists of French embeddings, separated by layer.

    """
    en_layers = [layer for layer in en_acts]
    fr_layers = [layer for layer in fr_acts]
    t.save(en_acts, f"{cache_folder}/{filename_base}-en-layers-{en_layers}.pt")
    t.save(fr_acts, f"{cache_folder}/{filename_base}-fr-layers-{fr_layers}.pt")


# -------------- functions 3 - train fr en embed rotation ------------------
def mean_vec(train_en_resids: t.Tensor, train_fr_resids: t.Tensor) -> t.Tensor:
    """
    Calculates the mean vector difference between English and French residuals.

    Args:
        train_en_resids (t.Tensor): The tensor containing English residuals.
        train_fr_resids (t.Tensor): The tensor containing French residuals.

    Returns:
        t.Tensor: The mean vector difference between English and French residuals.
    """
    return train_en_resids.mean(dim=0) - train_fr_resids.mean(dim=0)


def perform_translation_tests(
    model: nn.Module,
    en_strs: List[str],
    fr_strs: List[str],
    layer_idx: int,
    gen_length: int,
    transformation: nn.Module,
) -> None:
    """
    Performs translation tests on a model by generating translations for English
    and French strings. For each pair of strings in the provided lists, it prints the
    original string, generates a translation by iteratively appending the most likely
    next token, and prints the generated translation. For French strings, it modifies
    the model's behavior using a `steering_hook` during translation.

    Args:
        model (nn.Module): The transformer model used for translation tests.
        en_strs (List[str]): The list containing English strings.
        fr_strs (List[str]): The list containing French strings.
        layer_idx (int): The index of the layer to apply the steering hook.
        gen_length (int): The number of tokens to generate for the translation.
    """

    # Perform translation tests
    for idx, (test_en_str, test_fr_str) in enumerate(zip(en_strs, fr_strs)):
        print("\n----------------------------------------------")

        generate_translation(model, test_en_str, gen_length)
        generate_translation_with_hook(
            model, test_fr_str, gen_length, layer_idx, transformation
        )

        if idx > 5:
            break


def load_test_strings(file_path: Union[str, Path], skip_lines: int) -> List[str]:
    """
    Loads test strings from a file, skipping the first `skip_lines` lines.

    Args:
        file_path (str): The path to the file from which to load test strings.
        skip_lines (int): The number of lines to skip before loading test strings.

    Returns:
        List[str]: A list of test strings loaded from the file.
    """
    test_strs = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= skip_lines:
                next_line = next(f, "").strip()
                test_strs.append(line.strip() + " " + next_line)
    return test_strs


def generate_translation(model: nn.Module, test_str: str, gen_length: int) -> str:
    """
    Generates a translation for a given string using the model.

    Args:
        model (nn.Module): The transformer model used for generating translations.
        test_str (str): The string to translate.
        gen_length (int): The number of tokens to generate for the translation.

    Returns:
        str: The generated translation.
    """
    print("test_en_str:", test_str)
    original_len = len(test_str)
    for _ in range(gen_length):
        top_tok = model(test_str, prepend_bos=True)[:, -1].argmax(dim=-1)
        top_tok_str = model.to_string(top_tok)
        test_str += top_tok_str
    print("result fr str:", test_str[original_len:])
    return test_str


def generate_translation_with_hook(
    model: nn.Module,
    test_str: str,
    gen_length: int,
    layer_idx: int,
    transformation: nn.Module,
) -> str:
    """
    Generates a translation for a given string using the model with a steering hook.

    Args:
        model (nn.Module): The transformer model used for generating translations.
        test_str (str): The string to translate.
        gen_length (int): The number of tokens to generate for the translation.
        layer_idx (int): The index of the layer to apply the steering hook.
        transformation (nn.Module): The transformation to apply using the
        steering hook.

    Returns:
        str: The generated translation with the steering hook applied.
    """
    print("test_fr_str:", test_str)
    original_len = len(test_str)
    with remove_hooks() as handles, t.inference_mode():
        handle = model.blocks[layer_idx].hook_resid_pre.register_forward_hook(
            lambda module, input, output: steering_hook(
                module, input, output, transformation
            )
        )
        handles.add(handle)
        for _ in range(gen_length):
            top_tok = model(test_str, prepend_bos=True)[:, -1].argmax(dim=-1)
            top_tok_str = model.to_string(top_tok)
            test_str += top_tok_str
    print("result fr str", test_str[original_len:])
    return test_str


def steering_hook(
    module: nn.Module,
    input: Tuple[t.Tensor],
    output: t.Tensor,
    transformation: nn.Module,
) -> t.Tensor:
    """
    Modifies a module's output during translation by applying a transformation.

    Intended for use as a forward hook on a transformer model layer, this function
    steers the model's behavior, such as aligning embeddings across languages.

    Args:
        module (nn.Module): The module where the hook is registered.
        input (Tuple[t.Tensor]): Input tensors to the module, with the first tensor
                                 usually being the input embeddings.
        output (t.Tensor): The original output tensor of the module.
        transformation (nn.Module): The transformation to apply to the
                                                output tensor, which could be a
                                                learned matrix or another module.

    Returns:
        t.Tensor: The output tensor after applying the transformation, replacing the
                  original output in the model's forward pass.
    """
    prefix_toks, final_tok = input[0][:, :-1], input[0][:, -1]
    rotated_final_tok = transformation(final_tok)
    out = t.cat([prefix_toks, rotated_final_tok.unsqueeze(1)], dim=1)
    return out
