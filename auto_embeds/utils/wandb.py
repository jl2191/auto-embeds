import json

import numpy as np
import pandas as pd
import wandb
from loguru import logger
from tqdm import tqdm

from auto_embeds.utils.cache import auto_embeds_cache


@auto_embeds_cache
def fetch_wandb_runs(
    project_name: str,
    tags: list,
    samples: int = 10000,
    get_artifacts: bool = False,
) -> pd.DataFrame:
    """
    Fetches runs data from a specified project filtered by tags and compiles a DataFrame
    with run names, summaries, configurations, and histories. By default uses disk
    caching to speed up repeated calls with the same arguments.

    Args:
        project_name: The name of the project in wandb to fetch runs from.
        tags: A list of tags to filter the runs by.
        samples: The number of samples to fetch for each run history. Defaults to 10000.
        get_artifacts: Whether to download artifacts for the runs. If true, the artifact
            is saved to the summary dictionary. Defaults to False.

    Returns:
        A pandas DataFrame with columns: 'name', 'summary', 'config', 'history'.
        Each row corresponds to each run that matches the filters.
    """

    api = wandb.Api()
    filters = {"$and": [{"tags": tag} for tag in tags]}
    print(filters)
    runs = api.runs(project_name, filters=filters)

    run_name_list, summary_list, config_list, history_list = [], [], [], []

    for run_value in tqdm(runs, desc="Processing runs"):
        # .name is the human-readable name of the run.
        run_name_list.append(run_value.name)

        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary = run_value.summary._json_dict
        if get_artifacts:
            # for artifact in run_value.logged_artifacts():
            for file in run_value.files():
                if "cos_sims_trend_plot" in file.name:
                    plot_json = file.download(exist_ok=True).read()
                    summary["cos_sims_trend_plot"] = plot_json

        summary_list.append(summary)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k, v in run_value.config.items() if not k.startswith("_")}
        )

        # added me-self
        history_list.append(run_value.history(samples=samples, pandas=False))
        # history_list.append(run_value.scan_history())

    df = pd.DataFrame(
        {
            "run_name": run_name_list,
            "summary": summary_list,
            "config": config_list,
            "history": history_list,
        }
    )

    return df


def process_wandb_runs_df(df: pd.DataFrame, has_plot: bool = True) -> pd.DataFrame:
    """
    Processes a DataFrame of WandB runs returned by fetch_wandb_runs.

    Args:
        wandb_run_df: A DataFrame containing WandB runs data.
        has_plot: A boolean to control whether to filter runs based on the presence
            of 'cos_sims_trend_plot' in the summary.

    Returns:
        A DataFrame with columns for 'name', 'summary', 'config', and 'history'.
    """

    # log the unique run configs so that we know what parameters we changed / swept
    # through during the collection of runs
    configs = df["config"].to_list()
    unique_configs = {param: set() for config in configs for param in config.keys()}
    for config in configs:
        for param, value in config.items():
            if isinstance(value, dict):
                value = json.dumps(value, indent=4)
            unique_configs[param].add(value)

    # Log the unique config values
    for param, values in unique_configs.items():
        if any(isinstance(value, str) and "\n" in value for value in values):
            values = "\n".join(values)
            logger.info(f"unique config values | {param}: {values}")
        else:
            logger.info(f"unique config values | {param}: {values}")

    # Log the unique config values that change between runs
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

    filtered_df = df.query("history.str.len() > 0 or summary.str.len() > 0")

    num_filtered_out = len(df) - len(filtered_df)
    if num_filtered_out > 0:
        print(
            f"Warning: {num_filtered_out} run(s) with empty 'history' or 'summary' "
            "were removed. This is likely because they are still in progress."
        )

    df = filtered_df

    # Additional filtering for 'cos_sims_trend_plot'
    if has_plot:

        def has_plot_func(x):
            return "cos_sims_trend_plot" in x and len(x["cos_sims_trend_plot"]) > 0

        filtered_df = df.query("summary.map(@has_plot_func)")
        num_filtered_out = len(df) - len(filtered_df)
        if num_filtered_out > 0:
            print(
                f"Warning: {num_filtered_out} run(s) filtered out due to missing "
                "'cos_sims_trend_plot' in summary. These runs may still be processing."
            )

    df = filtered_df
    # ensure the DataFrame contains the expected columns
    expected_columns = ["run_name", "summary", "config", "history"]
    if not all(column in df.columns for column in expected_columns):
        raise ValueError(f"Input DataFrame must contain columns: {expected_columns}")

    # define a helper function to attempt processing and fail elegantly
    def attempt_to_process(df, process_func, fallback_value=None):
        try:
            return process_func(df)
        except Exception:
            return fallback_value

    def extract_from_history(data, attribute):
        result = []
        for step in data:
            if attribute in step:
                result.append(step[attribute])
        return result

    df = (
        # processing these fine gentlemen
        df.assign(
            dataset=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["dataset"]["name"])
            ),
            seed=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["seed"])
            ),
            loss_function=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["loss_function"])
            ),
            transformation=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["transformation"])
            ),
            embed_weight=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["embed_weight"])
            ),
            embed_ln=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["embed_ln"])
            ),
            embed_ln_weights=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["embed_ln_weights"])
            ),
            unembed_weight=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["unembed_weight"])
            ),
            unembed_ln=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["unembed_ln"])
            ),
            unembed_ln_weights=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["unembed_ln_weights"])
            ),
        )
        # process train_loss and test_loss
        .assign(
            train_loss=lambda df: df["history"].apply(
                lambda x: extract_from_history(x, "train_loss")
            ),
            test_loss=lambda df: df["history"].apply(
                lambda x: extract_from_history(x, "test_loss")
            ),
            epoch=lambda df: df["history"].apply(
                lambda x: extract_from_history(x, "epoch")
            ),
        )
        .assign(
            mark_translation_scores=lambda df: attempt_to_process(
                df,
                lambda x: x["history"].apply(
                    lambda history: [
                        step["mark_translation_score"]
                        for step in history
                        if "mark_translation_score" in step
                    ]
                ),
            ),
        )
        .assign(
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
        # %% summary stats
        .assign(
            cos_sims_trend_plot=lambda df: attempt_to_process(
                df, lambda x: x["summary"].apply(lambda x: x["cos_sims_trend_plot"])
            ),
            mark_translation_acc=lambda df: attempt_to_process(
                df, lambda x: x["summary"].apply(lambda x: x["mark_translation_acc"])
            ),
            test_accuracy=lambda df: attempt_to_process(
                df, lambda x: x["summary"].apply(lambda x: x["test_accuracy"])
            ),
            cosine_similarity_test_loss=lambda df: attempt_to_process(
                df,
                lambda x: x["summary"].apply(
                    lambda x: x["cosine_similarity_test_loss"]
                ),
            ),
            mse_test_loss=lambda df: attempt_to_process(
                df, lambda x: x["summary"].apply(lambda x: x["mse_test_loss"])
            ),
        )
        # turn these booleans into strings
        .astype(
            {
                "embed_ln": str,
                "unembed_ln": str,
            }
        )
    )
    return df


def list_changed_configs(df):
    # Collect unique run configs to identify parameters that were changed or swept
    # through during the collection of runs
    configs = df["config"].to_list()
    unique_configs = {param: set() for config in configs for param in config.keys()}
    for config in configs:
        for param, value in config.items():
            if isinstance(value, dict):
                value = str(value)
            unique_configs[param].add(value)

    # Create a dictionary to store unique config values that change between runs
    changed_configs = {}
    for param, values in unique_configs.items():
        if len(values) > 1:
            changed_configs[param] = values

    return changed_configs
