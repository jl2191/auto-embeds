from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch as t
import transformer_lens as tl
from jaxtyping import Float
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
from word2word import Word2word


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


def create_data_loaders(
    en_embeds: Float[Tensor, "pos d_model"],
    fr_embeds: Float[Tensor, "pos d_model"],
    batch_size: int,
    train_ratio: float,
    match_dims: bool = False,
) -> Tuple[DataLoader[Tuple[Tensor, Tensor]], DataLoader[Tuple[Tensor, Tensor]]]:
    """
    Creates data loaders for training and testing from embeddings.
    Returns a tuple containing a DataLoader for training and one for testing.

    Args:
        en_embeds: Tensor of English embeddings.
        fr_embeds: Tensor of French embeddings.
        batch_size: Size of each batch.
        train_ratio: Ratio for training dataset, a float greater than 0 but less than 1
        exclusive.
        match_dims: If True, matches dimensions of English and French embeddings.

    Returns:
        Tuple containing a DataLoader for training, and one for testing.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be greater than 0 and less than 1.")

    # Match dimensions if required
    if match_dims:
        min_len = min(len(en_embeds), len(fr_embeds))
        en_embeds, fr_embeds = en_embeds[:min_len], fr_embeds[:min_len]

    dataset = TensorDataset(en_embeds, fr_embeds)

    # Split dataset into training and testing
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    print(f"Train size: {train_size}, Test size: {test_size}")
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    train_loader, test_loader = cast(
        Tuple[DataLoader[Tuple[Tensor, Tensor]], DataLoader[Tuple[Tensor, Tensor]]],
        (train_loader, test_loader),
    )
    return train_loader, test_loader

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


def generate_google_words(
    model: tl.HookedTransformer,
    n_toks: int,
    en_file: List[str],
    device: Optional[Union[str, t.device]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Takes in a list of English strings then returns an English and French list.
    The French list contains French translations of English words for which a
    translation can be found.

    Args:
        model: The transformer model used for token processing.
        n_toks: The number of tokens to process.
        en_file: List of English strings to be translated.
        device: The device on which tensors will be allocated.

    Returns:
        A tuple of lists containing English strings and their French translations.
    """
    if device is None:
        device = model.cfg.device
    en2fr = Word2word("en", "fr")
    en_toks_list, fr_toks_list = [], []
    en_strs_list, fr_strs_list = [], []
    for i in range(n_toks):
        try:
            en_str = en_file[i]
            en_toks = model.to_tokens(en_str)
            fr_str = en2fr(en_str, n_best=1)[0]
            fr_toks = model.to_tokens(fr_str)
        except Exception:
            continue
        print(en_str)
        print(fr_str)
        en_toks_list.append(en_toks)
        fr_toks_list.append(fr_toks)
        en_strs_list.append(en_str)
        fr_strs_list.append(fr_str)
    return en_strs_list, fr_strs_list

