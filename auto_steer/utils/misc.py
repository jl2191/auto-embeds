import pickle
from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import torch as t
from einops import einsum, repeat
from torch.utils.hooks import RemovableHandle


def repo_path_to_abs_path(path: str) -> Path:
    repo_abs_path = Path(__file__).parent.parent.parent.absolute()
    return repo_abs_path / path


def save_cache(data_dict: Dict[Any, Any], folder_name: str, base_filename: str):
    folder = repo_path_to_abs_path(folder_name)
    folder.mkdir(parents=True, exist_ok=True)
    dt_string = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    file_path = folder / f"{base_filename}-{dt_string}.pkl"
    print(f"Saving cache to {file_path}")
    with open(file_path, "wb") as f:
        pickle.dump(data_dict, f)


def load_cache(folder_name: str, filename: str) -> Dict[Any, Any]:
    folder = repo_path_to_abs_path(folder_name)
    with open(folder / filename, "rb") as f:
        return pickle.load(f)


@contextmanager
def remove_hooks() -> Iterator[Set[RemovableHandle]]:
    handles: Set[RemovableHandle] = set()
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()


def module_by_name(model: Any, module_name: str) -> t.nn.Module:
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    return reduce(getattr, init_mod + module_name.split("."))  # type: ignore


def set_module_by_name(model: Any, module_name: str, new_module: t.nn.Module) -> None:
    parent = model
    init_mod = [model.wrapped_model] if hasattr(model, "wrapped_model") else [model]
    if "." in module_name:
        parent = reduce(getattr, init_mod + module_name.split(".")[:-1])  # type: ignore
    setattr(parent, module_name.split(".")[-1], new_module)


def percent_gpu_mem_used(total_gpu_mib: int = 49000) -> str:
    return (
        "Memory used {:.1f}".format(
            ((t.cuda.memory_allocated() / (2**20)) / total_gpu_mib) * 100
        )
        + "%"
    )


def run_prompt(
    model: t.nn.Module,
    prompt: str,
    answer: Optional[List[str]] = None,
    top_k: int = 10,
    prepend_bos: bool = False,
):
    print(" ")
    print("Testing prompt", model.to_str_tokens(prompt))
    toks = model.to_tokens(prompt, prepend_bos=prepend_bos)
    logits = model(toks)
    get_most_similar_embeddings(model, logits[0, -1], answer, top_k=top_k)


def get_most_similar_embeddings(
    model: t.nn.Module,
    out: t.Tensor,
    answer: Optional[List[str]] = None,
    top_k: int = 10,
    apply_ln_final: bool = False,
    apply_unembed: bool = False,
    apply_embed: bool = False,
    print_results: bool = False,
) -> Dict[int, Any]:
    assert not (apply_embed and apply_unembed), "Can't apply both embed and unembed"
    results = {}
    show_answer_rank = answer is not None
    answer = [" cheese"] * 129 if answer is None else answer
    out = out.unsqueeze(0).unsqueeze(0) if out.ndim == 1 else out
    out = model.ln_final(out) if apply_ln_final else out
    if apply_embed:
        unembeded = einsum(
            out, model.embed.W_E, "batch pos d_model, vocab d_model -> batch pos vocab"
        )
    elif apply_unembed:
        unembeded = model.unembed(out)
    else:
        unembeded = out
    answer_token = model.to_tokens(answer, prepend_bos=False)
    answer_str_token = model.to_str_tokens(answer, prepend_bos=False)
    logits = unembeded.squeeze()  # type: ignore
    probs = logits.softmax(dim=-1)

    sorted_token_probs, sorted_token_values = probs.sort(descending=True)

    # Janky way to get the index of the token in the sorted list
    if answer is not None:
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
        if show_answer_rank:
            # Collect rankings for each answer token.
            answer_ranks = [
                {
                    "token": token,
                    "rank": correct_rank[idx].item(),
                    "logit": logits[idx, answer_token[idx]].item(),
                    "prob": probs[idx, answer_token[idx]].item(),
                }
                for idx, token in enumerate(answer_str_token)
            ]
            # Store the collected answer ranks in the results dictionary.
            word_results["answer_rank"] = answer_ranks
        # Identify and store the top-k tokens based on their probabilities.
        top_tokens = [
            {
                "rank": i,
                "logit": logits[batch_idx, sorted_token_values[batch_idx, i]].item(),
                "prob": sorted_token_probs[batch_idx, i].item(),
                "token": model.to_string(sorted_token_values[batch_idx, i]),
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


def print_most_similar_embeddings_dict(
    most_similar_embeds_dict: Dict[int, Any]
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
