import json
import os
from typing import List, Tuple

import neptune
import pandas as pd
from tqdm.auto import tqdm

from auto_embeds.utils.cache import auto_embeds_cache
from auto_embeds.utils.logging import logger


def download_artifact(run, run_id, artifact_name):
    destination = f"neptune_artifact_downloads/{run_id}-{artifact_name}"
    run[f"results/json/{artifact_name}"].download(
        destination=destination,
        progress_bar=False,
    )
    with open(destination, "r") as file:
        artifact = file.read()
    return artifact


@auto_embeds_cache
def fetch_neptune_runs_df(
    project_name: str,
    tags: list,
    samples: int = 10000,
    get_artifacts: bool = False,
) -> pd.DataFrame:
    """
    Fetches runs data from a specified project filtered by tags and compiles a DataFrame
    with run names, summaries, configurations, and histories. By default uses disk
    caching to speed up repeated calls with the same arguments. This function now
    utilizes the Neptune API to directly fetch and filter runs based on system IDs,
    configurations, and results, and can optionally fetch artifacts related to the
    'cos_sims_trend_plot'.

    Args:
        project_name: The name of the project in neptune to fetch runs from.
        tags: A list of tags to filter the runs by.
        samples: The number of samples to fetch for each run history. Defaults to 10000.
        get_artifacts: Whether to download artifacts for the runs. If true, the artifact
            is saved to the summary dictionary. Defaults to False.

    Returns:
        A pandas DataFrame with columns: 'name', 'summary', 'config', 'history'.
        Each row corresponds to each run that matches the filters.
    """

    project = neptune.init_project(
        project=project_name,
        mode="read-only",
    )

    df = project.fetch_runs_table(tag=tags).to_pandas()
    df = df.filter(regex="|".join(["sys/id", "config", "results"]))

    artifact_types = ["cos_sims_trend_plot", "test_cos_sim_diff", "verify_results"]
    if get_artifacts:
        artifacts_data = {f"results/{artifact}": [] for artifact in artifact_types}

        pbar = tqdm(df["sys/id"].to_list(), desc="Downloading artifacts", unit="run")
        for run_id in pbar:
            run = neptune.init_run(
                project=project_name,
                with_id=run_id,
                mode="read-only",
            )
            os.makedirs("neptune_artifact_downloads", exist_ok=True)
            for artifact in artifact_types:
                artifacts_data[f"results/{artifact}"].append(
                    download_artifact(run, run_id, artifact)
                )
            pbar.set_description(f"Downloading artifacts for run ID: {run_id}")
        df = df.assign(**artifacts_data)
    else:
        artifacts_data = {
            f"results/{artifact}": [None] * len(df) for artifact in artifact_types
        }
        df = df.assign(**artifacts_data)

    return df


def process_neptune_runs_df(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Prepares a DataFrame from fetch_neptune_runs_df for analysis and visualization.

    This function modifies the DataFrame by renaming configuration columns to remove
    'config/' and 'results/' prefixes, reordering columns based on a predefined desired
    order, and logging unique configuration values. Additionally, it returns lists of
    all configuration names, names of configurations that change between runs, and
    names of results columns.

    Args:
        df: A DataFrame containing Neptune runs data from fetch_neptune_runs_df.

    Returns:
        A tuple containing:
        - A DataFrame with columns reordered and renamed for easier analysis.
        - A list of all config names.
        - A list of config names that change between runs.
        - A list of result column names.
    """
    desired_configs_order = [
        "config/description",
        "config/model_name",
        "config/processing",
        "config/dataset/name",
        "config/transformation",
        "config/train_batch_size",
        "config/test_batch_size",
        "config/top_k",
        "config/top_k_selection_method",
        "config/seed",
        "config/loss_function",
        "config/embed_weight",
        "config/embed_ln",
        "config/embed_ln_weights",
        "config/unembed_weight",
        "config/unembed_ln",
        "config/unembed_ln_weights",
        "config/n_epoch",
        "config/weight_decay",
        "config/lr",
    ]
    # filter desired order to include only columns that exist in the dataframe
    desired_configs_order = [col for col in desired_configs_order if col in df.columns]
    # append remaining columns that are not in the desired order
    ordered_columns = desired_configs_order + [
        col for col in df.columns if col not in desired_configs_order
    ]
    df = df.reindex(columns=ordered_columns)

    # deserialize and flatten the following columns
    columns_to_deserialize = [
        "results/test_cos_sim_diff",
        "results/verify_results",
    ]
    missing_artifacts = df[columns_to_deserialize].isnull().any(axis=1).sum()
    total_runs = len(df)
    logger.info(f"Runs with missing artifacts: {missing_artifacts}/{total_runs}")

    for col in columns_to_deserialize:
        if col in df.columns:
            df = df.join(
                pd.json_normalize(df[col].dropna().apply(json.loads)).add_prefix(
                    f"{col}."
                )
            ).drop(columns=[col])

    # extract config and result columns before renaming for further use
    config_columns = [col for col in df.columns if col.startswith("config/")]
    result_columns = [col for col in df.columns if col.startswith("results/")]

    # modify configuration columns to remove 'config/' and 'results/' prefixes for
    # convenience and backwards compatibility. also alias 'sys/id' to 'run_id'
    # and 'dataset/name' to 'dataset' for similar reasons.
    columns_to_shorten = config_columns + result_columns
    rename_mapping = {col: col.split("/", 1)[1] for col in columns_to_shorten}
    df = df.rename(columns=rename_mapping)

    # remove prefixes from config and result columns as we did so for the df
    config_columns = [col.split("/", 1)[1] for col in config_columns]
    result_columns = [col.split("/", 1)[1] for col in result_columns]

    df = df.assign(run_id=df["sys/id"])

    # rename dataset/name to dataset in both the df and config_columns list
    df = df.rename(columns={"dataset/name": "dataset"})
    config_columns = [col.replace("dataset/name", "dataset") for col in config_columns]

    # collect unique run configs to identify parameters that were changed or swept
    # through during the collection of runs
    unique_configs = {col: set() for col in config_columns}
    for col in config_columns:
        for value in df[col].unique():
            if isinstance(value, dict):
                value = str(value)
            unique_configs[col].add(value)

    # filter out config parameters that do not change between runs
    changed_configs = {
        param: values for param, values in unique_configs.items() if len(values) > 1
    }

    # log the unique config values
    for param, values in changed_configs.items():
        if any(isinstance(value, str) and "\n" in value for value in values):
            values = "\n".join(values)
            logger.info(f"unique config values | {param}: {values}")
        else:
            logger.info(f"unique config values | {param}: {values}")

    # log the unique config values that change between runs
    for param, values in changed_configs.items():
        if len(values) > 1:
            if any(isinstance(value, str) and "\n" in value for value in values):
                values = "\n".join(values)
                logger.info(
                    f"unique config values that change between runs | {param}: {values}"
                )
            else:
                logger.info(
                    f"unique config values that change between runs | {param}: {values}"
                )

    return (
        df,
        config_columns,
        list(changed_configs.keys()),
        result_columns,
    )
