# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wandb

from auto_embeds.utils.misc import dynamic_text_wrap

# %%
api = wandb.Api()
filters = {"$and": [{"tags": "actual"}, {"tags": "2024-04-15 effect of layernorm 7"}]}
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
# parallel categories diagram (max)
fig = (
    px.parallel_categories(
        runs_df,
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
            "embed_apply_ln": "Embed LN",
            "transform_apply_ln": "Transform LN",
            "unembed_apply_ln": "Unembed LN",
            "max_mark_translation_scores": "Max Score",
        },
        title="Max Mark Translation Score by LayerNorm",
    )
    .update_traces(arrangement="freeform")
    .add_annotation(
        text=dynamic_text_wrap(
            "",
            700,
        ),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.20,
        font=dict(size=13),
    )
    # .update_annotations(automargin=True)
    # .update_layout(
    #     automargin=True,
    #     autosize=True,
    # )
    .show(config={"responsive": True})
)
# %%
## only for rotation
fig = (
    px.parallel_categories(
        runs_df.query("transformation == 'Rotation'"),
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
            "embed_apply_ln": "Embed LN",
            "transform_apply_ln": "Transform LN",
            "unembed_apply_ln": "Unembed LN",
            "max_mark_translation_scores": "Max Score",
        },
        title="Max Mark Translation Score by LayerNorm",
    )
    # Dynamically adding annotation with automatic line breaks for extensive text
    .add_annotation(
        text=dynamic_text_wrap(
            "Same plot but only visualising for rotations.",
            700,
        ),
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.20,
        font=dict(size=13),
    )
    # .update_annotations(automargin=True)
    .update_layout(
        # automargin=True,
        # autosize=True,
    ).show(config={"responsive": True})
)
# %%
# Used for checking NaNs: Modify queries and columns_to_check as needed
# columns_to_check = ["avg_mark_translation_scores", "max_mark_translation_scores"]
# (
#     runs_df.query(
#         "dataset == 'random' or dataset == 'singular_plural'"
#     )  # Modify query as needed
#     .assign(
#         total_entries=lambda x: len(x),
#         nan_counts=lambda x: x[columns_to_check].isna().sum(axis=1),
#         possible_nans=lambda x: len(columns_to_check),
#     )
#     .pipe(
#         lambda x: print(
#             f"{x['nan_counts'].sum()}/{x['total_entries'].iloc[0] * x['possible_nans'].iloc[0]} "
#             f"({(x['nan_counts'].sum() / (x['total_entries'].iloc[0] * x['possible_nans'].iloc[0])) * 100:.2f}%) NaNs accounted for."
#         )
#     )
# )
