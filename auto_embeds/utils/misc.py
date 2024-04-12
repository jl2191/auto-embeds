import pickle
from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import torch as t
from einops import einsum
from torch.utils.hooks import RemovableHandle


def get_default_device():
    if t.cuda.is_available():
        return t.device("cuda")
    elif t.backends.mps.is_available():
        return t.device("mps")
    else:
        return t.device("cpu")


default_device = get_default_device()


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
    # get_most_similar_embeddings(model, logits[0, -1], answer, top_k=top_k)


def calculate_gradient_color(
    value: float, min_value: float, max_value: float, reverse: bool = False
) -> str:
    """
    Calculates the gradient color based on the value's magnitude within a range for rich
    printing library.

    This function normalizes the input value to a range between 0 and 1 based on the
    minimum and maximum values provided. It then calculates the red and green color
    components to represent the value on a gradient from red (low) to green (high).
    Optionally, the gradient can be reversed.

    Args:
        value: The value for which to calculate the gradient color.
        min_value: The minimum value in the range.
        max_value: The maximum value in the range.
        reverse: If True, reverses the gradient direction (default is False).

    Returns:
        A string representing the hex color code for the gradient.
    """
    # Normalize value to a range between 0 and 1
    normalized = (value - min_value) / (max_value - min_value)
    if reverse:
        normalized = 1 - normalized
    # Calculate red and green components, blue is kept to 0 for simplicity
    red = int(255 * (1 - normalized))
    green = int(255 * normalized)
    # Format as hex color
    return f"#{red:02x}{green:02x}00"
