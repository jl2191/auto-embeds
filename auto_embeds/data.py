# Mapping of dataset names to their file locations
from pathlib import Path
from typing import Optional, Tuple, Union

import torch as t
import transformer_lens as tl
from torch import Tensor
from word2word import Word2word

from auto_embeds.utils.misc import repo_path_to_abs_path

DATASETS = {
    # cc-cedict dataset
    "cc_cedict_zh_en_raw": repo_path_to_abs_path("datasets/cc-cedict/cedict_ts.u8"),
    "cc_cedict_zh_en_parsed": repo_path_to_abs_path(
        "datasets/cc-cedict/cc-cedict-zh-en-parsed.json"
    ),
    "cc_cedict_zh_en_filtered": repo_path_to_abs_path(
        "datasets/cc-cedict/cc-cedict-zh-en-filtered.json"
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
    "muse_zh_en_azure_validation_train": repo_path_to_abs_path(
        "datasets/muse/zh-en/4_azure_validation/muse-zh-en-train.json"
    ),
    "muse_zh_en_azure_validation_test": repo_path_to_abs_path(
        "datasets/muse/zh-en/4_azure_validation/muse-zh-en-test.json"
    ),
    # azure dataset
    "azure_translator_bloom_zh_en_zh_only": repo_path_to_abs_path(
        "datasets/azure_translator/bloom-zh-en-zh-only.json"
    ),
    "azure_translator_bloom_zh_en_all_translations": repo_path_to_abs_path(
        "datasets/azure_translator/bloom-zh-en-all-translations.json"
    ),
    # wikdict dataset
    "wikdict_en_fr_filtered": repo_path_to_abs_path(
        "datasets/wikdict/3_filtered/eng-fra.json"
    ),
}


def get_dataset_path(name: str) -> Path:
    """Retrieves the file path for a specified dataset using its logical identifier.

    This function serves as a centralized method for accessing the paths to various
    datasets and dictionary files within the project. It utilizes the `DATASETS`
    dictionary, which maps each dataset to a unique identifier, allowing for
    modification and expansion of dataset locations.

    Parameters:
    - name (str): The logical identifier of the dataset whose path is to be retrieved.

    Raises:
    - KeyError: If the dataset name is not found in the DATASETS dictionary.

    Returns:
    - Path: The absolute file path of the specified dataset.

    Example:
        from auto_embeds.data import get_dataset_path
        input_file_path = get_dataset_path("muse_en_fr_raw")
    """
    if name not in DATASETS:
        raise KeyError(f"Dataset name '{name}' not found.")
    return DATASETS[name]


def generate_embeddings(
    model: tl.HookedTransformer,
    en_toks: Tensor,
    fr_toks: Tensor,
    device: Optional[Union[str, t.device]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Generates embeddings for English and French tokens.

    Args:
        model: The transformer model used for generating embeddings.
        en_toks: Tensor of English token indices.
        fr_toks: Tensor of French token indices.
        device: The device on which tensors will be allocated. Defaults to None.

    Returns:
        Tuple: A tuple of tensors containing embeddings for English and French tokens.
    """
    if device is None:
        device = model.device
    en_embeds = model.embed.W_E[en_toks].detach().clone().to(device)
    fr_embeds = model.embed.W_E[fr_toks].detach().clone().to(device)
    return en_embeds, fr_embeds


def generate_tokens(
    model: tl.HookedTransformer,
    n_toks: int,
    device: Optional[Union[str, t.device]] = None,
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
    if device is None:
        device = model.cfg.device
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
