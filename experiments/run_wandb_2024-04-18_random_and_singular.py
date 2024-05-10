# %%
# import numpy as np
import pandas as pd
import plotly.express as px

from auto_embeds.utils.misc import (
    dynamic_text_wrap,
)
from auto_embeds.utils.neptune import fetch_neptune_runs, process_neptune_runs_df
from auto_embeds.utils.plot import create_parallel_categories_plot

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass

# %%
## run that was specifically for random and singular plural
# fetching data and creating DataFrame
original_df = fetch_neptune_runs(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-04-17 random and singular plural", "run group 2"],
    get_artifacts=True,
)

# %%
runs_df = process_neptune_runs_df(
    original_df,
    custom_labels={
        "transformation": {"rotation": "Rotation", "linear_map": "Linear Map"},
        "dataset": {
            "wikdict_en_fr_extracted": "wikdict_en_fr",
            "cc_cedict_zh_en_extracted": "cc_cedict_zh_en",
        },
    },
)

# %%
neptune_run_config_variables_that_change = (
    pd.json_normalize(runs_df["config"]).nunique().loc[lambda x: x > 1]
)

for variable in neptune_run_config_variables_that_change.index:
    unique_values = pd.json_normalize(runs_df["config"])[variable].unique()
    print(f"{variable}: {unique_values}")

# %%
create_parallel_categories_plot(
    df=runs_df,
    dimensions=[
        "transformation",
        "dataset",
        "seed",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    color="test_loss",
    labels={
        "transformation": "Transformation",
        "dataset": "Dataset",
        "embed_apply_ln": "Embed LN",
        "transform_apply_ln": "Transform LN",
        "unembed_apply_ln": "Unembed LN",
    },
    title="Max Test Accuracy by LayerNorm",
    annotation_text="So here it seems like seed is the parameter that is making the "
    "most difference on our test metric. As such, averaging over this for our next "
    "graph may let us see the best performing combinations.",
)

# %%
# Plotting train and test loss as separate line graphs
# Validate that 'train_loss' and 'test_loss' columns exist in the DataFrame
if "train_loss" not in runs_df.columns or "test_loss" not in runs_df.columns:
    raise ValueError("DataFrame must contain 'train_loss' and 'test_loss' columns.")

# query = None
# if query:
#     runs_df = runs_df.query(query)

# %%
# train_loss plot
fig = (
    runs_df.explode(column=["epoch", "test_loss", "train_loss"])
    .melt(
        id_vars=[
            "name",
            "dataset",
            "seed",
            "transformation",
            "embed_apply_ln",
            "unembed_apply_ln",
            "epoch",
        ],
        value_vars=[
            "train_loss",
            "test_loss",
        ],
        var_name="loss_type",
        value_name="loss",
    )
    .pipe(
        lambda df: px.line(
            data_frame=df,
            x="epoch",
            y="loss",
            color="transformation",
            facet_col="loss_type",
            title="Train and Test Loss By Transformation",
            labels={
                "name": "Name",
                "dataset": "Dataset",
                "seed": "Seed",
                "transformation": "Transformation",
                "embed_apply_ln": "Embed LN",
                "unembed_apply_ln": "Unembed LN",
                "loss_type": "Loss Type",
                "loss": "Loss",
                "epoch": "Epoch",
            },
            hover_data=[
                "name",
                "dataset",
                "seed",
                "transformation",
                "embed_apply_ln",
                "unembed_apply_ln",
            ],
        )
        .update_traces(
            opacity=0.8,
        )
        .add_annotation(
            text=dynamic_text_wrap(
                "Very weird how for one of the Random Word Pair seeds, it seems to "
                "converge in terms of both train and test loss! My guess is that this "
                "is one particular seed? Zooming in and hovering over, shows that it "
                "is the linear layer!",
                800,
            ),
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.4,
            font=dict(size=12),
        )
        .update_layout(
            autosize=True,
            height=500,
            margin=dict(b=150),
        )
        .show(config={"responsive": True})
    )
)

# %%
fig = (
    runs_df.explode(column=["epoch", "test_loss", "train_loss"])
    .melt(
        id_vars=[
            "name",
            "dataset",
            "seed",
            "transformation",
            "embed_apply_ln",
            "unembed_apply_ln",
            "epoch",
        ],
        value_vars=[
            "train_loss",
            "test_loss",
        ],
        var_name="loss_type",
        value_name="loss",
    )
    .pipe(
        lambda df: px.line(
            data_frame=df,
            x="epoch",
            y="loss",
            color="transformation",
            facet_col="loss_type",
            title="Train and Test Loss By Dataset",
            labels={
                "name": "Name",
                "dataset": "Dataset",
                "seed": "Seed",
                "transformation": "Transformation",
                "embed_apply_ln": "Embed LN",
                "unembed_apply_ln": "Unembed LN",
                "loss_type": "Loss Type",
                "loss": "Loss",
                "epoch": "Epoch",
            },
            hover_data=[
                "name",
                "dataset",
                "seed",
                "transformation",
                "embed_apply_ln",
                "unembed_apply_ln",
            ],
        )
        .update_traces(
            opacity=0.8,
        )
        .add_annotation(
            text=dynamic_text_wrap(
                "Doesn't seem to be!",
                800,
            ),
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.4,
            font=dict(size=12),
        )
        .update_layout(
            autosize=True,
            height=500,
            margin=dict(b=150),
        )
        .show(config={"responsive": True})
    )
)
# %%
# train_loss plot
fig = px.line(
    runs_df.explode(column=["epoch", "test_loss", "train_loss"]),
    x="epoch",
    y="train_loss",
    color="dataset",
    title="Train Loss By Dataset",
    labels={"epoch": "Epoch", "train_loss": "Train Loss", "dataset": "Dataset"},
).update_layout(autosize=True)
fig.show(config={"responsive": True})

# test_loss plot
fig = px.line(
    runs_df.explode(column=["epoch", "test_loss", "train_loss"]),
    x="epoch",
    y="test_loss",
    color="dataset",
    title="Test Loss By Dataset",
    labels={"epoch": "Epoch", "test_loss": "Test Loss", "dataset": "Dataset"},
).update_layout(autosize=True)
fig.show(config={"responsive": True})
