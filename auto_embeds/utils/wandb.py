import numpy as np
import pandas as pd
import wandb
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

    name_list, summary_list, config_list, history_list = [], [], [], []

    for run_value in tqdm(runs, desc="Processing runs"):
        # .name is the human-readable name of the run.
        name_list.append(run_value.name)

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
            "name": name_list,
            "summary": summary_list,
            "config": config_list,
            "history": history_list,
        }
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
    def has_plot(x):
        return "cos_sims_trend_plot" in x and len(x["cos_sims_trend_plot"]) > 0

    filtered_df = df.query("summary.map(@has_plot)")
    num_filtered_out = len(df) - len(filtered_df)
    if num_filtered_out > 0:
        print(
            f"Warning: {num_filtered_out} run(s) filtered out due to missing "
            "'cos_sims_trend_plot' in summary. These runs may still be processing."
        )

    df = filtered_df

    return df


def process_wandb_runs_df(df: pd.DataFrame, custom_labels: dict = None) -> pd.DataFrame:
    """
    Processes a DataFrame of WandB runs returned by fetch_wandb_runs.

    Args:
        wandb_run_df: A DataFrame containing WandB runs data.
        custom_labels: Optional; Dict for custom column entry labels.

    Returns:
        A DataFrame with columns for 'name', 'summary', 'config', and 'history'.
    """
    # ensure the DataFrame contains the expected columns
    expected_columns = ["name", "summary", "config", "history"]
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
            run_name=lambda df: df["name"],
            dataset=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["dataset"]["name"])
            ),
            seed=lambda df: attempt_to_process(
                df, lambda x: x["config"].apply(lambda x: x["seed"])
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
            cos_sims_trend_plot=lambda df: attempt_to_process(
                df, lambda x: x["summary"].apply(lambda x: x["cos_sims_trend_plot"])
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
        # turn these booleans into strings
        .astype(
            {
                "embed_ln": str,
                "unembed_ln": str,
            }
        )
    )
    labels_to_use = custom_labels if custom_labels is not None else {}
    df = df.replace(labels_to_use)
    return df
