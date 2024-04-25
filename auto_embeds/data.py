# Mapping of dataset names to their file locations
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias, Union, overload

import numpy as np
import torch as t
import transformer_lens as tl
from einops import repeat
from fancy_einsum import einsum
from Levenshtein import distance as levenshtein_distance
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from word2word import Word2word

from auto_embeds.utils.cache import auto_embeds_cache
from auto_embeds.utils.misc import (
    default_device,
    repo_path_to_abs_path,
)


@dataclass(frozen=True)
class WordData:
    """Encapsulates common data for words, including tokens and embeddings."""

    words: List[str]
    toks: t.Tensor
    embeds: t.Tensor


@dataclass(frozen=True)
class ExtendedWordData(WordData):
    """Extends WordData with cosine similarities and Euclidean distances."""

    cos_sims: t.Tensor
    euc_dists: t.Tensor


@dataclass(frozen=True)
class WordCategory:
    """Contains instances of WordData for selected and other words."""

    selected: WordData
    other: ExtendedWordData


@dataclass(frozen=True)
class VerifyWordPairAnalysis:
    """Contains two instances of WordCategory for src and tgt."""

    src: WordCategory
    tgt: WordCategory


DATASETS = {
    # cc-cedict dataset
    "cc_cedict_zh_en_raw": repo_path_to_abs_path("datasets/cc-cedict/cedict_ts.u8"),
    "cc_cedict_zh_en_extracted": repo_path_to_abs_path(
        "datasets/cc-cedict/cc-cedict-zh-en-parsed.json"
    ),
    "cc_cedict_zh_en_parsed": repo_path_to_abs_path(
        "datasets/cc-cedict/cc-cedict-zh-en-parsed.json"
    ),
    "cc_cedict_zh_en_filtered": repo_path_to_abs_path(
        "datasets/cc-cedict/cc-cedict-zh-en-filtered.json"
    ),
    "cc_cedict_zh_en_azure_validation": repo_path_to_abs_path(
        "datasets/cc-cedict/cc-cedict-zh-en-azure-validation.json"
    ),
    # facebook muse en-fr dataset
    "muse_en_fr_raw": repo_path_to_abs_path("datasets/muse/en-fr/1_raw/en-fr.txt"),
    "muse_en_fr_extracted": repo_path_to_abs_path(
        "datasets/muse/en-fr/2_extracted/en-fr.json"
    ),
    "muse_en_fr_filtered": repo_path_to_abs_path(
        "datasets/muse/en-fr/3_filtered/en-fr.json"
    ),
    "muse_en_fr_azure_validation": repo_path_to_abs_path(
        "datasets/muse/en-fr/4_azure_validation/en-fr.json"
    ),
    # facebook muse zh-en dataset
    "muse_zh_en_raw_train": repo_path_to_abs_path(
        "datasets/muse/zh-en/1_raw/muse-zh-en-train.txt"
    ),
    "muse_zh_en_raw_test": repo_path_to_abs_path(
        "datasets/muse/zh-en/1_raw/muse-zh-en-test.txt"
    ),
    "muse_zh_en_extracted_train": repo_path_to_abs_path(
        "datasets/muse/zh-en/2_extracted/muse-zh-en-train.json"
    ),
    "muse_zh_en_extracted_test": repo_path_to_abs_path(
        "datasets/muse/zh-en/2_extracted/muse-zh-en-test.json"
    ),
    "muse_zh_en_filtered_train": repo_path_to_abs_path(
        "datasets/muse/zh-en/3_filtered/muse-zh-en-train.json"
    ),
    "muse_zh_en_filtered_test": repo_path_to_abs_path(
        "datasets/muse/zh-en/3_filtered/muse-zh-en-test.json"
    ),
    "muse_zh_en_azure_validation": repo_path_to_abs_path(
        "datasets/muse/zh-en/4_azure_validation/muse-zh-en-azure-val.json"
    ),
    # azure dataset
    "azure_translator_bloom_zh_en_zh_only": repo_path_to_abs_path(
        "datasets/azure_translator/bloom-zh-en-zh-only.json"
    ),
    "azure_translator_bloom_zh_en_all_translations": repo_path_to_abs_path(
        "datasets/azure_translator/bloom-zh-en-all-translations.json"
    ),
    # wikdict dataset
    "wikdict_en_fr_extracted": repo_path_to_abs_path(
        "datasets/wikdict/2_extracted/eng-fra.json"
    ),
    "wikdict_en_fr_filtered": repo_path_to_abs_path(
        "datasets/wikdict/3_filtered/eng-fra.json"
    ),
    "wikdict_en_fr_azure_validation": repo_path_to_abs_path(
        "datasets/wikdict/4_azure_validation/eng-fra.json"
    ),
    # random datasets to act as controls
    "random_word_pairs": repo_path_to_abs_path(
        "datasets/random/random_word_pairs.json"
    ),
    "singular_plural_pairs": repo_path_to_abs_path(
        "datasets/singular-plural/singular_plural_pairs.json"
    ),
}


def get_dataset_path(name: str) -> Path:
    """
    Fetches the absolute path for a given dataset identifier.

    Args:
        name (str): The identifier of the dataset to retrieve the path for.

    Raises:
        KeyError: If the specified dataset identifier does not exist in the DATASETS
            dictionary.

    Returns:
        Path: The absolute path of the dataset corresponding to the given identifier.

    Example:
        >>> from auto_embeds.data import get_dataset_path
        >>> dataset_path = get_dataset_path("muse_en_fr_raw")
        >>> print(dataset_path)
    """
    if name not in DATASETS:
        raise KeyError(f"Dataset name '{name}' not found.")
    return DATASETS[name]


def generate_tokens(
    model: tl.HookedTransformer,
    n_toks: int,
    device: Union[str, t.device] = default_device,
) -> Tuple[Tensor, Tensor]:
    """
    Generate token pairs from a model for a given number of tokens.

    This function generates English and French token pairs using a HookedTransformer
    model. It filters out tokens based on specific criteria and attempts to translate
    English tokens to French using the Word2word library. The function returns tensors
    of English and French tokens that meet the criteria.

    Args:
        model: The transformer model used for token generation.
        n_toks: The number of tokens to generate.
        device: The device on which to allocate tensors. Defaults to model's device.

    Returns:
        A tuple of tensors containing English and French tokens.
    """
    en2fr = Word2word("en", "fr")
    en_toks, fr_toks = [], []
    for tok in range(n_toks):
        en_tok_str = model.to_string([tok])
        if len(en_tok_str) < 7 or en_tok_str[0] != " ":
            continue
        try:
            fr_tok_str = " " + en2fr(en_tok_str[1:], n_best=1)[0]
        except Exception as e:
            print(f"Translation failed for {en_tok_str}: {e}")
            continue
        if en_tok_str.lower() == fr_tok_str.lower():  # type: ignore
            continue
        try:
            fr_tok = model.to_single_token(fr_tok_str)
        except Exception as e:
            print(f"Token conversion failed for {fr_tok_str}: {e}")
            continue
        en_toks.append(tok)
        fr_toks.append(fr_tok)
    return t.tensor(en_toks, device=device), t.tensor(fr_toks, device=device)


@auto_embeds_cache
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
    acceptable_english_overlap: float = 1.0,
    acceptable_french_overlap: float = 1.0,
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
            for fr_word, (
                token_sum,
                en_word,
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
            (
                current_id_sum,
                current_en_token,
                current_fr_token,
                current_words,
            ) = current_pair
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
            (
                current_id_sum,
                current_en_token,
                current_fr_token,
                current_words,
            ) = current_pair
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
    model: tl.HookedTransformer,
    word_pairs: List[List[str]],
    device: Union[str, t.device] = default_device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Converts a list of word pairs into tensors suitable for model input.

    Args:
        model: A HookedTransformer model instance with a tokenizer.
        word_pairs: A list of word pairs, where each pair is a list of two strings.
        device: The device to perform calculations on. Defaults to None.

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
    ).to(
        device
    )  # type: ignore
    num_pairs = tokenized.input_ids.shape[0]
    assert num_pairs % 2 == 0
    word_each = num_pairs // 2
    tokens = tokenized.data["input_ids"]
    attn_masks = tokenized.data["attention_mask"]
    en_tokens, fr_tokens = tokens[:word_each], tokens[word_each:]
    en_mask, fr_mask = attn_masks[:word_each], attn_masks[word_each:]

    return en_tokens, fr_tokens, en_mask, fr_mask


TwoDataLoaders: TypeAlias = Tuple[DataLoader, DataLoader]
TwoTensors: TypeAlias = Tuple[Tensor, Tensor]
FourTensors: TypeAlias = Tuple[Tensor, Tensor, Tensor, Tensor]
TwoDatasets: TypeAlias = Tuple[TensorDataset, TensorDataset]


@overload
def prepare_data(
    model: tl.HookedTransformer,
    embed_module: t.nn.Module,
    dataset_name: str,
    return_type: Literal["tensor"],
    split_ratio: None = None,
    seed: int | None = None,
    filter_options: dict[str, bool | int | str] | None = None,
    batch_size: int = 128,
    shuffle_word_pairs: bool = False,
    shuffle_dataloader: bool = True,
) -> TwoTensors: ...


@overload
def prepare_data(
    model: tl.HookedTransformer,
    embed_module: t.nn.Module,
    dataset_name: str,
    return_type: Literal["tensor"],
    split_ratio: float,
    seed: int | None = None,
    filter_options: dict[str, bool | int | str] | None = None,
    batch_size: int = 128,
    shuffle_word_pairs: bool = False,
    shuffle_dataloader: bool = True,
) -> FourTensors: ...


@overload
def prepare_data(
    model: tl.HookedTransformer,
    embed_module: t.nn.Module,
    dataset_name: str,
    return_type: Literal["dataset"],
    split_ratio: None = None,
    seed: int | None = None,
    filter_options: dict[str, bool | int | str] | None = None,
    batch_size: int = 128,
    shuffle_word_pairs: bool = False,
    shuffle_dataloader: bool = True,
) -> TensorDataset: ...


@overload
def prepare_data(
    model: tl.HookedTransformer,
    embed_module: t.nn.Module,
    dataset_name: str,
    return_type: Literal["dataset"],
    split_ratio: float,
    seed: int | None = None,
    filter_options: dict[str, bool | int | str] | None = None,
    batch_size: int = 128,
    shuffle_word_pairs: bool = False,
    shuffle_dataloader: bool = True,
) -> TwoDatasets: ...


@overload
def prepare_data(
    model: tl.HookedTransformer,
    embed_module: t.nn.Module,
    dataset_name: str,
    return_type: Literal["dataloader"],
    split_ratio: None = None,
    seed: int | None = None,
    filter_options: dict[str, bool | int | str] | None = None,
    batch_size: int = 128,
    shuffle_word_pairs: bool = False,
    shuffle_dataloader: bool = True,
) -> DataLoader: ...


@overload
def prepare_data(
    model: tl.HookedTransformer,
    embed_module: t.nn.Module,
    dataset_name: str,
    return_type: Literal["dataloader"],
    split_ratio: float,
    seed: int | None = None,
    filter_options: dict[str, bool | int | str] | None = None,
    batch_size: int = 128,
    shuffle_word_pairs: bool = False,
    shuffle_dataloader: bool = True,
) -> TwoDataLoaders: ...


def prepare_data(
    model: tl.HookedTransformer,
    embed_module: t.nn.Module,
    dataset_name: str,
    return_type: str = "dataset",
    split_ratio: float | None = None,
    seed: int | None = None,
    filter_options: dict[str, bool | int | str] | None = None,
    batch_size: int = 128,
    shuffle_word_pairs: bool = False,
    shuffle_dataloader: bool = True,
) -> (
    TensorDataset | TwoDatasets | TwoTensors | FourTensors | DataLoader | TwoDataLoaders
):
    """Prepares and processes data from a specified dataset for training and testing.

    This function is overloaded to return different types based on the `return_type`
        and `dataloader` parameters.

    Args:
        model: The model used for tokenization.
        dataset_name: The name of the dataset to load.
        split_ratio: The ratio to split the dataset into training and testing sets. If
            None, no splitting is performed.
        seed: The seed for random word_pair and dataloader shuffling operations.
            Setting a seed should make the results deterministic. Defaults to None.
        filter_options: Additional options for filtering word pairs. Defaults to
            None.
        return_type: The type of return object. Can be 'dataset', 'tensor', or
            'dataloader'.
            Defaults to 'dataset'.
        dataloader: Whether to return a DataLoader instead of TensorDataset.
            Defaults to False.
        batch_size: The batch size for the DataLoader. Ignored if dataloader is False.
            Defaults to 128.
        shuffle_word_pairs: Whether to shuffle the word pairs before processing.
            Defaults to False.
        shuffle_dataloader: Whether to shuffle the DataLoader. Defaults to True.
        embed_module: The module used for embedding operations.

    Returns:
        Depending on return_type, dataloader, and whether a split is performed, either
        a single TensorDataset instance, a tuple of TensorDataset instances, raw
        Tensors, a single DataLoader, or a tuple of DataLoaders.
    """
    if dataset_name is None:
        raise ValueError("dataset_name must be specified.")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        t.manual_seed(seed)
        t.cuda.manual_seed(seed)

    file_path = get_dataset_path(dataset_name)
    with open(file_path, "r") as file:
        word_pairs = json.load(file)

    default_filter_options = {
        "discard_if_same": True,
        "min_length": 2,
        "capture_no_space": True,
        "print_number": True,
    }
    if filter_options is not None:
        default_filter_options.update(filter_options)

    all_word_pairs = filter_word_pairs(model, word_pairs, **default_filter_options)

    if shuffle_word_pairs:
        random.shuffle(all_word_pairs)

    if split_ratio is not None:
        split_index = int(len(all_word_pairs) * split_ratio)

        train_pairs = all_word_pairs[:split_index]
        test_pairs = all_word_pairs[split_index:]

        train_en_toks, train_fr_toks, _, _ = tokenize_word_pairs(model, train_pairs)
        test_en_toks, test_fr_toks, _, _ = tokenize_word_pairs(model, test_pairs)

        train_en_embeds, train_fr_embeds = map(
            embed_module, (train_en_toks, train_fr_toks)
        )
        test_en_embeds, test_fr_embeds = map(embed_module, (test_en_toks, test_fr_toks))

        train_dataset = TensorDataset(train_en_embeds, train_fr_embeds)
        test_dataset = TensorDataset(test_en_embeds, test_fr_embeds)

        if return_type == "dataloader":
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=shuffle_dataloader
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=shuffle_dataloader
            )
            return train_loader, test_loader
        elif return_type == "dataset":
            return train_dataset, test_dataset
        else:  # return_type == "tensor"
            return (train_en_embeds, train_fr_embeds, test_en_embeds, test_fr_embeds)
    else:
        en_toks, fr_toks, _, _ = tokenize_word_pairs(model, all_word_pairs)
        en_embeds, fr_embeds = map(embed_module, (en_toks, fr_toks))

        dataset = TensorDataset(en_embeds, fr_embeds)

        if return_type == "dataloader":
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle_dataloader
            )
            return loader
        elif return_type == "dataset":
            return dataset
        else:  # return_type == "tensor"
            return (en_embeds, fr_embeds)


def print_most_similar_embeddings_dict(
    most_similar_embeds_dict: Dict[int, Any],
) -> None:
    for i in range(len(most_similar_embeds_dict)):
        if "answer_rank" in most_similar_embeds_dict[i]:
            for answer_rank in most_similar_embeds_dict[i]["answer_rank"]:
                print(answer_rank)
                print(
                    f'\n"{answer_rank["token"]}" token rank:',
                    f'{answer_rank["rank"]: <8}',
                    f'\nLogit: {answer_rank["logit"]:5.2f}',
                    f'Prob: {answer_rank["prob"]:6.2%}',
                )
        for top_token in most_similar_embeds_dict[i]["top_tokens"]:
            print(
                f"Top {top_token['rank']}th token. Logit: {top_token['logit']:5.2f}",
                f"Prob: {top_token['prob']:6.2%}",
                f'Token: "{top_token["token"]}"',
            )


def get_most_similar_embeddings(
    model: tl.HookedTransformer,
    out: t.Tensor,
    answer: Optional[List[str]] = None,
    top_k: int = 10,
    print_results: bool = False,
) -> Dict[int, Any]:

    results = {}
    # Adjust tensor dimensions if needed
    out = out.unsqueeze(0).unsqueeze(0) if out.ndim == 1 else out
    # Reshape the output tensor
    logits = out.squeeze(1)
    # Convert logits to probabilities
    probs = logits.softmax(dim=-1)

    sorted_token_probs, sorted_token_values = probs.sort(descending=True)

    if answer is not None:
        answer_token = model.to_tokens(answer, prepend_bos=False)
        answer_str_token = model.to_str_tokens(answer, prepend_bos=False)
        correct_rank = repeat(
            t.arange(sorted_token_values.shape[-1]),
            "d_vocab -> batch d_vocab",
            batch=sorted_token_values.shape[0],
        )[(sorted_token_values == answer_token).cpu()]

    results = {}
    # This loop compiles a results dictionary per batch, including rankings of correct
    # answers (if any) and the top-k predicted tokens.
    for batch_idx in range(sorted_token_values.shape[0]):
        # Initialize a dictionary to hold results for the current batch.
        word_results = {}
        # If an answer is provided, calculate its rank and related information.
        if answer is not None:
            # Collect rankings for each answer token.
            answer_ranks = [
                {
                    "token": token,
                    "rank": correct_rank[idx].item(),  # type: ignore
                    "logit": logits[idx, answer_token[idx]].item(),  # type: ignore
                    "prob": probs[idx, answer_token[idx]].item(),  # type: ignore
                }
                for idx, token in enumerate(answer_str_token)  # type: ignore
            ]
            # Store the collected answer ranks in the results dictionary.
            word_results["answer_rank"] = answer_ranks
        # Identify and store the top-k tokens based on their probabilities.
        top_tokens = [
            {
                "rank": i,
                "logit": logits[batch_idx, sorted_token_values[batch_idx, i]].item(),
                "prob": sorted_token_probs[batch_idx, i].item(),
                "token": model.tokenizer.decode(sorted_token_values[batch_idx, i]),
            }
            for i in range(top_k)
        ]
        word_results["top_tokens"] = top_tokens
        # Assign results for the current batch to the main results dictionary.
        results[batch_idx] = word_results
    # Optionally print the results for each batch.
    if print_results:
        for key, batch_results in results.items():
            print_most_similar_embeddings_dict(batch_results)
            print()
    return results
