# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wandb
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.formula.api import ols

from auto_embeds.utils.misc import dynamic_text_wrap

# %%
api = wandb.Api()
filters = {"$and": [{"tags": "actual"}, {"tags": "2024-04-15 effect of layernorm 7"}]}
runs = api.runs("jl2191/language-transformations", filters=filters)

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
    history_list.append(run_value.history(samples=10000, pandas=False))
    # history_list.append(run_value.scan_history())

# %%
original_df = pd.DataFrame(
    {
        "name": name_list,
        "summary": summary_list,
        "config": config_list,
        "history": history_list,
    }
)
# %%
# Filter out runs with empty 'history' or 'summary' and issue a warning if any are found
# Using lambda functions to avoid 'len' not defined error in query method
filtered_df = original_df[
    original_df["history"].map(lambda x: len(x) > 0)
    | original_df["summary"].map(lambda x: len(x) > 0)
]
num_filtered_out = len(original_df) - len(filtered_df)
if num_filtered_out > 0:
    print(
        f"Warning: {num_filtered_out} run(s) with empty 'history' or 'summary' were "
        "removed. This is likely because they are still in progress."
    )
original_df = filtered_df

original_df
# %%
# Constructing a DataFrame with specific configurations and calculated scores
runs_df = (
    # stealing columns but processed
    original_df.assign(
        run_name=lambda df: df["name"],
        dataset=lambda df: df["config"].apply(lambda x: x["name"]),
        seed=lambda df: df["config"].apply(lambda x: x["seed"]),
        transformation=lambda df: df["config"].apply(lambda x: x["transformation"]),
        embed_apply_ln=lambda df: df["config"].apply(lambda x: x["embed_apply_ln"]),
        transform_apply_ln=lambda df: df["config"].apply(
            lambda x: x["transform_apply_ln"]
        ),
        unembed_apply_ln=lambda df: df["config"].apply(lambda x: x["unembed_apply_ln"]),
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
    # give them their proper name :)
    .replace(
        {
            "transformation": {"rotation": "Rotation", "linear_map": "Linear Map"},
            "dataset": {
                "wikdict_en_fr_extracted": "wikdict_en_fr",
                "cc_cedict_zh_en_extracted": "cc_cedict_zh_en",
            },
        }
    )
)
# %%
# drop runs where max_mark_translation_scores is NaN
runs_df = runs_df.dropna(subset=["max_mark_translation_scores"])


# %%
def create_parallel_categories_plot(
    df, groupby_conditions, dimensions, color, labels, title, annotation_text
):
    """
    Creates and displays a parallel categories plot based on the provided parameters.

    Args:
        df (pd.DataFrame): The DataFrame to plot.
        groupby_conditions (list): Conditions to group the DataFrame by.
        dimensions (list): The dimensions to use in the plot.
        color (str): The column name to color the lines by.
        labels (dict): A mapping of column names to display labels.
        title (str): The title of the plot.
        annotation_text (str): Text for the annotation to add to the plot.
    """
    # Group and reset index for plotting
    plot_df = df.groupby(groupby_conditions)[color].mean().reset_index()

    # Create the plot
    fig = (
        px.parallel_categories(
            plot_df,
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


# Example usage for the first plot
create_parallel_categories_plot(
    df=runs_df,
    groupby_conditions=[
        "transformation",
        "dataset",
        "seed",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
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
    annotation_text="So here it seems like seed is the parameter that is making the most difference on our test metric. As such, averaging over this for our next graph may let us see the best performing combinations.",
)

# Example usage for the second plot
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
    title="Max Mark Translation Score by LayerNorm, Averaging Over Seed",
    annotation_text="Hmm, I don't think I see a massive difference here, the scores are all between 0.3 and 0.45. Then again, there are only two seeds here. Let's actually do the same plot but for our big run where we did",
)

# %%
api = wandb.Api()
filters = {"$and": [{"tags": "actual"}, {"tags": "2024-04-06"}, {"tags": "chungus"}]}
runs = api.runs("jl2191/language-transformations", filters=filters)

name_list, summary_list, config_list, history_list = [], [], [], []

for run_value in tqdm(runs, desc="Processing runs"):
    # .name is the human-readable name of the run.
    name_list.append(run_value.name)

    # .summary contains output keys/values for metrics such as accuracy.
    # We call ._json_dict to omit large files
    summary_list.append(run_value.summary._json_dict)

    # .config contains the hyperparameters.
    # We remove special values that start with _.
    config_list.append(
        {k: v for k, v in run_value.config.items() if not k.startswith("_")}
    )

    # added me-self
    history_list.append(run_value.history(samples=10000, pandas=False))

# %%
original_df = pd.DataFrame(
    {
        "name": name_list,
        "summary": summary_list,
        "config": config_list,
        "history": history_list,
    }
)

# Filter out runs with empty 'history' or 'summary' and issue a warning if any are found
# Using lambda functions to avoid 'len' not defined error in query method
filtered_df = original_df[
    original_df["history"].map(lambda x: len(x) > 0)
    | original_df["summary"].map(lambda x: len(x) > 0)
]
num_filtered_out = len(original_df) - len(filtered_df)
if num_filtered_out > 0:
    print(
        f"Warning: {num_filtered_out} run(s) with empty 'history' or 'summary' were "
        "removed. This is likely because they are still in progress."
    )
original_df = filtered_df
# %%
changing_config_variables = (
    pd.json_normalize(original_df["config"])
    .nunique()
    .loc[lambda x: x > 1]
    .index.tolist()
)

for variable in changing_config_variables:
    print(variable)

# %%
# Constructing a DataFrame with specific configurations and calculated scores
runs_df = (
    # stealing columns but processed
    original_df.assign(
        run_name=lambda df: df["name"],
        dataset=lambda df: df["config"].apply(lambda x: x["name"]),
        seed=lambda df: df["config"].apply(lambda x: x["seed"]),
        transformation=lambda df: df["config"].apply(lambda x: x["transformation"]),
        embed_apply_ln=lambda df: df["config"].apply(lambda x: x["embed_apply_ln"]),
        transform_apply_ln=lambda df: df["config"].apply(
            lambda x: x["transform_apply_ln"]
        ),
        unembed_apply_ln=lambda df: df["config"].apply(lambda x: x["unembed_apply_ln"]),
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
    # give them their proper name :)
    .replace(
        {
            "transformation": {"rotation": "Rotation", "linear_map": "Linear Map"},
            "dataset": {
                "wikdict_en_fr_extracted": "wikdict_en_fr",
                "cc_cedict_zh_en_extracted": "cc_cedict_zh_en",
            },
        }
    )
)
# %%
# drop runs where max_mark_translation_scores is NaN
runs_df = runs_df.dropna(subset=["max_mark_translation_scores"])

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
    "Translation, Uncentered Rotation and Biased Rotation Yoinked",
    annotation_text="Now it seems uncentered rotation and biased rotation also did quite poorly. Let's yoink them as well.",
)

# %%
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
    "Translation Yoinked",
    annotation_text="I think this makes sense, given that we are unembedding with layernorm, "
    "you get good performance if you apply layernorm to the embedding"
    "beforehand.",
)

# %%
# Create a new column to represent the combination of transform_apply_ln and unembed_apply_ln
runs_df["LN Combination"] = (
    runs_df["transform_apply_ln"].astype(str)
    + " & "
    + runs_df["unembed_apply_ln"].astype(str)
)

# Calculate the average max_mark_translation_scores for each LN Combination
# You might want to group by another variable (e.g., transformation or dataset) if you're comparing across those as well
avg_scores = (
    runs_df.groupby(["LN Combination", "transformation"])["max_mark_translation_scores"]
    .mean()
    .reset_index()
)

# Create the grouped bar chart
fig = px.bar(
    avg_scores,
    x="transformation",  # or 'dataset' or any other variable you're interested in
    y="max_mark_translation_scores",
    color="LN Combination",  # Differentiates the bars based on the LN Combination
    barmode="group",
    title="Average Max Mark Translation Scores by LN Application Combination",
    labels={
        "max_mark_translation_scores": "Average Max Score",
        "transformation": "Transformation",  # or 'Dataset' or your variable of interest
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
