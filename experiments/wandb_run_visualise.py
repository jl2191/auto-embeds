# %%
import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from auto_embeds.utils.misc import (
    create_parallel_categories_plot,
    fetch_wandb_runs,
)

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass

# %%
# feel free to ignore, reduplicating this code here from misc.py because some wandb
# config names were changed around. this is the appropriate one for this file.


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
    name_list, summary_list, config_list, history_list = fetch_wandb_runs(
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
            dataset=lambda df: df["config"].apply(lambda x: x["name"]),
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
        # take care of mark translation scores
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


# %%

# fetching data and creating DataFrame
original_df = fetch_wandb_runs_as_df(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-04-15 effect of layernorm 7"],
    custom_labels={
        "transformation": {"rotation": "Rotation", "linear_map": "Linear Map"},
        "dataset": {
            "wikdict_en_fr_extracted": "wikdict_en_fr",
            "cc_cedict_zh_en_extracted": "cc_cedict_zh_en",
        },
    },
)

# %%
# max mark translation score by layernorm
create_parallel_categories_plot(
    df=original_df,
    dimensions=[
        "transformation",
        "dataset",
        "seed",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    color="max_mark_translation_scores",
    labels={
        "transformation": "Transformation",
        "dataset": "Dataset",
        "embed_apply_ln": "Embed LN",
        "transform_apply_ln": "Transform LN",
        "unembed_apply_ln": "Unembed LN",
        "max_mark_translation_scores": "Max Score",
    },
    title="Max Mark Translation Score by LayerNorm",
    annotation_text="So here it seems like seed is the parameter that is making the "
    "most difference on our test metric. As such, averaging over this for our next "
    "graph may let us see the best performing combinations.",
)

# %%
create_parallel_categories_plot(
    df=original_df,
    groupby_conditions=[
        "transformation",
        "dataset",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    dimensions=[
        "transformation",
        "dataset",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    color="max_mark_translation_scores",
    labels={
        "transformation": "Transformation",
        "dataset": "Dataset",
        "embed_apply_ln": "Embed LN",
        "transform_apply_ln": "Transform LN",
        "unembed_apply_ln": "Unembed LN",
        "max_mark_translation_scores": "Max Score",
    },
    title="Max Mark Translation Score by LayerNorm, Averaging Over Seed",
    annotation_text="Hmm, I don't think I see a massive difference here, the scores "
    "are all between 0.3 and 0.45. Then again, there are only two seeds here. Let's "
    "actually do the same plot but for our big run where we did",
)

# %%
# fetching data for another set of runs
original_df = fetch_wandb_runs_as_df(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-04-06", "chungus"],
)
# %%
wandb_run_config_variables_that_change = (
    pd.json_normalize(original_df["config"])
    .nunique()
    .loc[lambda x: x > 1]
    .index.tolist()
)

for variable in wandb_run_config_variables_that_change:
    print(variable)

# %%
# drop runs where max_mark_translation_scores is NaN
runs_df = original_df.dropna(subset=["max_mark_translation_scores"])

# %% averaging over seed
create_parallel_categories_plot(
    df=runs_df,
    groupby_conditions=[
        "transformation",
        "dataset",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    dimensions=[
        "transformation",
        "dataset",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    color="max_mark_translation_scores",
    labels={
        "transformation": "Transformation",
        "dataset": "Dataset",
        "embed_apply_ln": "Embed LN",
        "transform_apply_ln": "Transform LN",
        "unembed_apply_ln": "Unembed LN",
        "max_mark_translation_scores": "Max Score",
    },
    title="Chungus Run - Max Mark Translation Score, Averaging Over Seed",
    annotation_text="Seems like translation usually did quite poorly. Let's yoink it.",
)

# %%
# chungus run - max mark translation score, averaging over seed. translation, uncentered
# and biased rotation yoinked
create_parallel_categories_plot(
    df=runs_df.query("`transformation` != 'translation'"),
    groupby_conditions=[
        "transformation",
        "dataset",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    dimensions=[
        "transformation",
        "dataset",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    color="max_mark_translation_scores",
    labels={
        "transformation": "Transformation",
        "dataset": "Dataset",
        "embed_apply_ln": "Embed LN",
        "transform_apply_ln": "Transform LN",
        "unembed_apply_ln": "Unembed LN",
        "max_mark_translation_scores": "Max Score",
    },
    title="Chungus Run - Max Mark Translation Score, Averaging Over Seed, "
    "Translation Yoinked",
    annotation_text="Now it seems uncentered rotation and biased rotation also did "
    "quite poorly. Let's yoink them as well.",
)

# %%
# chungus run - max mark translation score, averaging over seed. translation, uncentered
# and biased rotation yoinked
create_parallel_categories_plot(
    df=runs_df.query(
        "`transformation` not in ['translation', 'biased_rotation', 'uncentered_rotation']"
    ),
    groupby_conditions=[
        "transformation",
        "dataset",
        "embed_apply_ln",
        "unembed_apply_ln",
    ],
    dimensions=[
        "transformation",
        "dataset",
        "embed_apply_ln",
        "unembed_apply_ln",
    ],
    color="max_mark_translation_scores",
    labels={
        "transformation": "Transformation",
        "dataset": "Dataset",
        "embed_apply_ln": "Embed LN",
        "unembed_apply_ln": "Unembed LN",
        "max_mark_translation_scores": "Max Score",
    },
    title="Chungus Run - Max Mark Translation Score, Averaging Over Seed, "
    "Translation, Uncentered Rotation and Biased Rotation Yoinked",
    annotation_text="I think this makes sense, given that we are unembedding with "
    "layernorm, you get good performance if you apply layernorm to the embedding"
    "beforehand.",
)

# %%
# Create a new column to represent the combination of transform_apply_ln and
# unembed_apply_ln
runs_df["LN Combination"] = (
    runs_df["transform_apply_ln"].astype(str)
    + " & "
    + runs_df["unembed_apply_ln"].astype(str)
)

avg_scores = (
    runs_df.groupby(["LN Combination", "transformation"])["max_mark_translation_scores"]
    .mean()
    .reset_index()
)

# Create the grouped bar chart
fig = px.bar(
    avg_scores,
    x="transformation",
    y="max_mark_translation_scores",
    color="LN Combination",
    barmode="group",
    title="Average Max Mark Translation Scores by LN Application Combination",
    labels={
        "max_mark_translation_scores": "Average Max Score",
        "transformation": "Transformation",
        "LN Combination": "LayerNorm Application Combination",
    },
)
fig.show()

# %%
# and the same chart with error bars
fig = px.bar(
    avg_scores,
    x="transformation",
    y="max_mark_translation_scores",
    color="LN Combination",
    barmode="group",
    error_y="max_mark_translation_scores",
    title="Average Max Mark Translation Scores by LN Application Combination",
    labels={
        "max_mark_translation_scores": "Average Max Score",
        "transformation": "Transformation",
        "LN Combination": "LayerNorm Application Combination",
    },
)
fig.show()

# %%
(
    runs_df.assign(
        **{
            "LN Combination": (
                runs_df["embed_apply_ln"].astype(str)
                + " & "
                + runs_df["transform_apply_ln"].astype(str)
                + " & "
                + runs_df["unembed_apply_ln"].astype(str)
            )
        }
    )
    .groupby(["LN Combination", "transformation", "dataset"])[
        "max_mark_translation_scores"
    ]
    .agg(["mean", "std"])
    .reset_index()
    .pipe(
        lambda df: px.bar(
            df,
            x="transformation",
            y="mean",
            error_y="std",
            color="LN Combination",
            facet_row="dataset",
            barmode="group",
            title="Average Max Mark Translation Scores by LN Application Combination "
            "and Dataset",
            labels={
                "mean": "Average Max Score",
                "std": "Standard Deviation",
                "transformation": "Transformation",
                "LN Combination": "LayerNorm Application Combination",
                "dataset": "Dataset",
            },
        )
    )
    .show()
)

# %%
