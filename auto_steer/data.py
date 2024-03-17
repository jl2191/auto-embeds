from typing import Optional, Tuple, Union

import torch as t
import transformer_lens as tl
from torch import Tensor
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
