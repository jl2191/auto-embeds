import os

import neptune
import pandas as pd
from loguru import logger


# @auto_embeds_cache
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
    print(df)

    # Use .filter() for more readable column filtering
    df = df.filter(regex="|".join(["sys/id", "config", "results"]))

    if get_artifacts:
        cos_sims_trend_plots = []
        test_cos_sim_diffs = []
        verify_results = []

        def download_and_append(run, filename, result_list):
            run[f"results/json/{filename}"].download(destination="./neptune_downloads/")
            with open(
                os.path.join(os.getcwd(), "neptune_downloads", f"{filename}.txt")
            ) as file:
                result_list.append(file.read())

        for run_id in df["sys/id"].to_list():
            run = neptune.init_run(
                project=project_name,
                with_id=run_id,
                mode="read-only",
            )
            download_and_append(run, "cos_sims_trend_plot", cos_sims_trend_plots)
            download_and_append(run, "test_cos_sim_diff", test_cos_sim_diffs)
            download_and_append(run, "verify_results", verify_results)

        # assign these still in their json form. will need to process them later.
        new_columns = {
            "results/cos_sims_trend_plot": cos_sims_trend_plots,
            "results/test_cos_sim_diff": test_cos_sim_diffs,
            "results/verify_results": verify_results,
        }
        df = df.assign(**new_columns)

    return df


def process_neptune_runs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame of neptune runs returned by fetch_neptune_runs.

    Args:
        df: A DataFrame containing neptune runs data.

    Returns:
        A DataFrame with columns for 'name', 'summary', 'config', and 'history'.
    """

    # log the unique run configs so that we know what parameters we changed / swept
    # through during the collection of runs
    unique_configs = list_changed_configs(df)

    # log the unique config values
    for param, values in unique_configs.items():
        if any(isinstance(value, str) and "\n" in value for value in values):
            values = "\n".join(values)
            logger.info(f"unique config values | {param}: {values}")
        else:
            logger.info(f"unique config values | {param}: {values}")

    # log the unique config values that change between runs
    for param, values in unique_configs.items():
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

    # rename columns to remove 'config/' and 'results/' prefixes and create a copy of
    # 'dataset/name' as 'dataset' for convenience and backwards compatibility
    rename_mapping = {col: col.split("/", 1)[1] for col in df.columns if "/" in col}
    df = df.rename(columns=rename_mapping)
    df = df.assign(dataset=df["dataset/name"])

    return df


def list_changed_configs(df):
    # collect unique run configs to identify parameters that were changed or swept
    # through during the collection of runs
    config_columns = [col for col in df.columns if col.startswith("config/")]
    unique_configs = {col.split("/", 1)[1]: set() for col in config_columns}
    for col in config_columns:
        for value in df[col].unique():
            if isinstance(value, dict):
                value = str(value)
            unique_configs[col.split("/", 1)[1]].add(value)

    # create a dictionary to store unique config values that change between runs
    changed_configs = {}
    for param, values in unique_configs.items():
        if len(values) > 1:
            changed_configs[param] = values

    return changed_configs
