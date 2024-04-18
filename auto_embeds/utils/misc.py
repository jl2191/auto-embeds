import itertools
import logging
import pickle
from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
import wandb
from einops import einsum
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm


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


def get_experiment_worker_config(
    experiment_config, split_parameter, n_splits, worker_id, print_config=True
) -> dict:
    """
    Generate a subset of the experiment configuration based on the specified split
    parameter, number of splits, and worker ID. This function divides the
    configuration into n_splits parts and returns the part corresponding to the
    worker_id as a dictionary.

    Args:
        experiment_config (dict): The full experiment configuration dictionary.
        split_parameter (str): The key in the configuration to split by.
        n_splits (int): The total number of splits to divide the configuration into.
        worker_id (int): The ID of the worker, determining which part of the split
                         to return, or 0 to indicate a test run of all experiments.
        print_config (bool): If True, prints the resulting configuration subset along
                             with additional information about how the division was
                             made.

    Returns:
        dict: A subset of the configuration based on the provided arguments.
    """
    # Add "test" or "actual" tag to wandb configuration based on worker_id
    tag = "test" if worker_id == 0 else "actual"
    experiment_config["wandb"]["tags"].append(tag)

    if worker_id == 0:
        # If worker_id is 0, return the entire configuration for testing
        if print_config:
            print("Running all experiments in test mode with 'test' tag.")
        return experiment_config

    # Validate the split parameter exists in the configuration
    if split_parameter not in experiment_config:
        raise ValueError(f"Split parameter '{split_parameter}' not found in config.")

    # Validate worker_id and n_splits
    if worker_id < 1 or worker_id > n_splits:
        raise ValueError(
            f"Worker ID {worker_id} is out of range for n_splits {n_splits}."
        )

    # Calculate the size of each split
    total_items = len(experiment_config[split_parameter])
    split_size = total_items // n_splits
    remainder = total_items % n_splits

    # Adjust start and end indices for workers (1 and above)
    start_index = (worker_id - 1) * split_size
    if worker_id <= remainder:
        start_index += worker_id - 1
    else:
        start_index += remainder

    end_index = start_index + split_size
    if worker_id <= remainder:
        end_index += 1

    # Update only the split_parameter part of the configuration for the given worker_id
    experiment_config[split_parameter] = experiment_config[split_parameter][
        start_index:end_index
    ]

    if print_config:
        print(
            f"Configuration for '{split_parameter}' is divided into {n_splits} parts."
        )
        print(f"Each part has up to {split_size} items.")
        print(
            f"Worker ID {worker_id} will process items from index {start_index} "
            f"to {end_index - 1}."
        )
        print(
            f"Config subset particular to worker_id {worker_id}: "
            f"{experiment_config[split_parameter]}"
        )

    return experiment_config


def is_notebook():
    """Check if the script is running in a Jupyter notebook/IPython session."""
    try:
        from IPython import get_ipython  # type: ignore

        if "IPKernelApp" in get_ipython().config:  # type: ignore
            return True
    except ImportError:
        return False
    except AttributeError:
        return False
    return False


def dynamic_text_wrap(text, plot_width_px, font_size=12, font_width_approx=7):
    """
    Dynamically wraps text for Plotly annotations based on the estimated plot width.

    Args:
    - text: The original text to wrap.
    - plot_width_px: The estimated width of the plot in pixels.
    - font_size: The font size used in the plot annotations.
    - font_width_approx: Approx width of each char in pixels, can vary by font.

    Returns:
    - str: The wrapped text with <br> tags inserted at appropriate intervals.
    """
    # Estimate the number of characters per line based on plot width and font size
    chars_per_line = max(1, plot_width_px // (font_size * font_width_approx / 12))

    # Split the text into words
    words = text.split()

    # Initialize variables for the wrapped text and the current line length
    wrapped_text = ""
    current_line_length = 0

    for word in words:
        # Check if adding the next word exceeds the line length
        if current_line_length + len(word) > chars_per_line:
            # If so, add a line break before the word
            wrapped_text += "<br>"
            current_line_length = 0
        elif wrapped_text:  # If not the first word, add a space before the word
            wrapped_text += " "
            current_line_length += 1

        # Add the word to the wrapped text and update the current line length
        wrapped_text += word
        current_line_length += len(word)

    return wrapped_text


# %%
def fetch_wandb_runs_as_lists(
    project_name: str, tags: list, samples: int = 10000
) -> tuple:
    """
    Fetches runs data from a specified project filtered by tags and compiles lists of
    run names, summaries, configurations, and histories.

    Args:
        project_name: The name of the project in wandb to fetch runs from.
        tags: A list of tags to filter the runs by.
        samples: The number of samples to fetch for each run history. Defaults to 10000.

    Returns:
        A tuple of lists: (name_list, summary_list, config_list, history_list).
        Each list contains elements corresponding to each run that matches the filters.
    """
    api = wandb.Api()
    filters = {"$and": [{"tags": tag} for tag in tags]}
    print(filters)
    runs = api.runs(project_name, filters=filters)

    name_list, summary_list, config_list, history_list = [], [], [], []

    for run_value in tqdm(runs, desc="Processing runs"):
        # .name is the human-readable name of the run.
        name_list.append(run_value.name)

        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run_value.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run_value.config.items() if not k.startswith("_")}
        )

        # added me-self
        history_list.append(run_value.history(samples=samples, pandas=False))
        # history_list.append(run_value.scan_history())

    return name_list, summary_list, config_list, history_list


def fetch_wandb_runs_as_df(
    project_name: str, tags: list, custom_labels: dict = None
) -> pd.DataFrame:
    """
    Fetches runs data from WandB, filters out runs with empty 'history' or 'summary',
    and creates a DataFrame. These happen usually because they are still in progress.

    Args:
        project_name: The name of the WandB project.
        tags: A list of tags to filter the runs by.
        custom_labels: Optional; Dict for custom column entry labels.

    Returns:
        A DataFrame with columns for 'name', 'summary', 'config', and 'history'.
    """
    name_list, summary_list, config_list, history_list = fetch_wandb_runs_as_lists(
        project_name=project_name,
        tags=tags,
    )

    df = pd.DataFrame(
        {
            "name": name_list,
            "summary": summary_list,
            "config": config_list,
            "history": history_list,
        }
    )

    filtered_df = df[
        df["history"].map(lambda x: len(x) > 0)
        | df["summary"].map(lambda x: len(x) > 0)
    ]
    num_filtered_out = len(df) - len(filtered_df)
    if num_filtered_out > 0:
        print(
            f"Warning: {num_filtered_out} run(s) with empty 'history' or 'summary' "
            "were removed. This is likely because they are still in progress."
        )

    df = filtered_df

    df = (
        # stealing columns but processed
        df.assign(
            run_name=lambda df: df["name"],
            dataset=lambda df: df["config"].apply(lambda x: x["dataset"]["name"]),
            seed=lambda df: df["config"].apply(lambda x: x["seed"]),
            transformation=lambda df: df["config"].apply(lambda x: x["transformation"]),
            embed_apply_ln=lambda df: df["config"].apply(lambda x: x["embed_apply_ln"]),
            transform_apply_ln=lambda df: df["config"].apply(
                lambda x: x["transform_apply_ln"]
            ),
            unembed_apply_ln=lambda df: df["config"].apply(
                lambda x: x["unembed_apply_ln"]
            ),
        )
        # process train_loss and test_loss
        .assign(
            train_loss=lambda df: df["history"].apply(
                lambda history: [
                    step["train_loss"]
                    for step in history
                ]
            )
        )
        .assign(
            test_loss=lambda df: df["history"].apply(
                lambda history: [
                    step["test_loss"]
                    for step in history
                ]
            )
        )
        # process mark translation scores
        .assign(
            mark_translation_scores=lambda df: df["history"].apply(
                lambda history: [
                    step["mark_translation_score"]
                    for step in history
                    if "mark_translation_score" in step
                ]
            )
        ).assign(
            avg_mark_translation_scores=lambda df: df["mark_translation_scores"].apply(
                lambda scores: (
                    np.nan
                    if not scores
                    else np.mean([score for score in scores if score is not None])
                )
            ),
            max_mark_translation_scores=lambda df: df["mark_translation_scores"].apply(
                lambda scores: (
                    np.nan
                    if not scores
                    else max([score for score in scores if score is not None])
                )
            ),
        )
        # turn these booleans into strings
        .astype(
            {
                "embed_apply_ln": str,
                "transform_apply_ln": str,
                "unembed_apply_ln": str,
            }
        )
    )
    labels_to_use = custom_labels if custom_labels is not None else {}
    df = df.replace(labels_to_use)
    return df


def create_parallel_categories_plot(
    df,
    dimensions,
    color,
    title,
    annotation_text,
    groupby_conditions=None,
    query=None,
    labels=None,
):
    """
    Creates and displays a parallel categories plot based on the provided parameters,
    with options to filter the DataFrame using a query string and to group the DataFrame
    by specified conditions. Rows where the color column is NA are filtered out.

    Args:
        df (pd.DataFrame): The DataFrame to plot.
        dimensions (list): The dimensions to use in the plot.
        color (str): The column name to color the lines by.
        title (str): The title of the plot.
        annotation_text (str): Text for the annotation to add to the plot.
        groupby_conditions (list, optional): Conditions to group the DataFrame by.
        query (str, optional): A query string to filter the DataFrame before plotting.
        labels (dict, optional): A mapping of column names to display labels.
            Defaults to a predefined dictionary.
    """

    # Apply query if provided
    if query:
        df = df.query(query)

    # Filter out rows where the color column is NA and log the action
    filtered_df = df.dropna(subset=[color])
    num_filtered = len(df) - len(filtered_df)
    if num_filtered > 0:
        logging.info(f"Filtered out {num_filtered} rows with NA in '{color}' column.")

    df = filtered_df

    # Use the DataFrame directly for plotting, applying groupby conditions if provided
    if groupby_conditions:
        df = df.groupby(groupby_conditions)[color].mean().reset_index()

    fig = (
        px.parallel_categories(
            df,
            dimensions=dimensions,
            color=color,
            labels=labels,
            title=title,
        )
        .update_traces(arrangement="freeform")
        .add_annotation(
            text=dynamic_text_wrap(annotation_text, 600),
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.25,
            font=dict(size=13),
        )
        .update_layout(autosize=True)
    )

    fig.show(config={"responsive": True})
