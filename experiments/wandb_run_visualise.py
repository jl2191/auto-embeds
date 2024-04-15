# %%
import pandas as pd
import plotly.express as px
import wandb
import plotly.graph_objects as go


# %%
api = wandb.Api()
filters = {"tags": "after crash again"}
runs = api.runs("jl2191/language-transformations", filters=filters)

name_list, summary_list, config_list, history_list = [], [], [], []

for run_value in runs:
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
runs_df = (
    pd.DataFrame(
        {
            "name": name_list,
            "transformation": [config["transformation"] for config in config_list],
            "embed_apply_ln": [config["embed_apply_ln"] for config in config_list],
            "transform_apply_ln": [
                config["transform_apply_ln"] for config in config_list
            ],
            "unembed_apply_ln": [config["unembed_apply_ln"] for config in config_list],
            "mark_translation_scores": original_df["history"].apply(
                lambda history: [
                    step["mark_translation_score"]
                    for step in history
                    if step["mark_translation_score"] is not None
                ]
            ),
        }
    )
    .assign(
        avg_mark_translation_scores=lambda df: df["mark_translation_scores"].apply(
            lambda x: sum(x) / len(x) if len(x) > 0 else 0
        ),
        max_mark_translation_scores=lambda df: df["mark_translation_scores"].apply(max),
    )
    .astype(
        {
            "embed_apply_ln": str,
            "transform_apply_ln": str,
            "unembed_apply_ln": str,
        }
    )
    .replace({"transformation": {"rotation": "Rotation", "linear_map": "Linear Map"}})
)
# %%
runs_df

# %%
# parallel categories diagram (mean)
# fig = px.parallel_categories(
#     runs_df,
#     dimensions=[
#         "transformation",
#         "embed_apply_ln",
#         "transform_apply_ln",
#         "unembed_apply_ln",
#     ],
#     color="avg_mark_translation_scores",
#     # color_continuous_scale=px.colors.sequential.Inferno,
#     labels={
#         "transformation": "Transformation",
#         "embed_apply_ln": "Embed LN",
#         "transform_apply_ln": "Transform LN",
#         "unembed_apply_ln": "Unembed LN",
#         "avg_mark_translation_scores": "Mean Score",
#     },
#     title="Average Mark Translation Score by LayerNorm (Linear Map)",
# ).show(config={"responsive": True})

# %%
# parallel categories diagram (max)
fig = px.parallel_categories(
    runs_df,
    dimensions=[
        "transformation",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    color="max_mark_translation_scores",
    # color_continuous_scale=px.colors.sequential.Inferno,
    labels={
        "transformation": "Transformation",
        "embed_apply_ln": "Embed LN",
        "transform_apply_ln": "Transform LN",
        "unembed_apply_ln": "Unembed LN",
        "max_mark_translation_scores": "Max Score",
    },
    title="Max Mark Translation Score by LayerNorm",
).show(config={"responsive": True})

# %% only for rotation
fig = px.parallel_categories(
    runs_df.query("transformation == 'Rotation'"),
    dimensions=[
        "transformation",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    color="max_mark_translation_scores",
    labels={
        "transformation": "Transformation",
        "embed_apply_ln": "Embed LN",
        "transform_apply_ln": "Transform LN",
        "unembed_apply_ln": "Unembed LN",
        "max_mark_translation_scores": "Max Score",
    },
    title="Max Mark Translation Score by LayerNorm",
).show(config={"responsive": True})
