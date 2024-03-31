from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch as t
import torch.nn as nn
import transformer_lens as tl
from torch import Tensor
from torch.utils.data import DataLoader

from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.misc import remove_hooks


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
