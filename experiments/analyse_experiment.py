# %%
import pandas as pd
import plotly.express as px
import shap
from IPython.core.getipython import get_ipython
from sklearn.ensemble import RandomForestRegressor

from auto_embeds.utils.neptune import (
    fetch_neptune_runs_df,
    process_neptune_runs_df,
)
from auto_embeds.utils.plot import create_parallel_categories_plot
from experiments.configure_experiment import experiment_config
from experiments.scratch_funcs import display_top_runs_table

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass


# %%
project_name = "mars/language-transformations"
# tags = ["2024-05-29 new metrics sweep", "run group 1"]
visualise_all_run_groups = True
tags = experiment_config["neptune"]["tags"]
if visualise_all_run_groups:
    tags = [tag for tag in tags if "run group" not in tag]

# fetch_neptune_runs_df.clear_cache()
original_df = fetch_neptune_runs_df(
    project_name=project_name,
    tags=tags,
    get_artifacts=False,
)

# %%
(
    runs_df,
    config_column_names,
    changed_configs_column_names,
    result_column_names,
) = process_neptune_runs_df(original_df)
excluded_substrings = ["dataset/"]
excluded_exact_matches = ["embed_ln", "unembed_ln"]

runs_df["test_cos_sim"] = -runs_df["cos_sim_test_loss"]
runs_df.drop(columns=["cos_sim_test_loss"], inplace=True)
result_column_names = [
    col.replace("cos_sim_test_loss", "test_cos_sim") for col in result_column_names
]

runs_df["train_cos_sim"] = runs_df["train_metrics/cos_sim"]
runs_df.drop(columns=["train_metrics/cos_sim"], inplace=True)
result_column_names = [
    col.replace("train_metrics/cos_sim", "train_cos_sim") for col in result_column_names
]

runs_df["train_mse_loss"] = runs_df["train_metrics/mse_loss"]
runs_df.drop(columns=["train_metrics/mse_loss"], inplace=True)
result_column_names = [
    col.replace("train_metrics/mse_loss", "train_mse_loss")
    for col in result_column_names
]

config_column_names = [
    col
    for col in config_column_names
    if not any(sub in col for sub in excluded_substrings)
    and col not in excluded_exact_matches
]
changed_configs_column_names = [
    col
    for col in changed_configs_column_names
    if not any(sub in col for sub in excluded_substrings)
    and col not in excluded_exact_matches
]

numerical_result_column_names = [
    col for col in result_column_names if runs_df[col].dtype == "float64"
]

numerical_filter = changed_configs_column_names + numerical_result_column_names

runs_df["model_name"] = runs_df["model_name"].str.replace(
    "bigscience/", "", regex=False
)
runs_df["seed"] = runs_df["seed"].astype(str)

# ic(config_column_names)
# ic(changed_configs_column_names)
# ic(result_column_names)

custom_labels = (
    {
        "transformation": {
            "rotation": "Rotation",
            "linear_map": "Linear Map",
            "translation": "Translation",
            "analytical_rotation": "Analytical Rotation",
            "analytical_translation": "Analytical Translation",
            "uncentered_rotation": "Uncentered Rotation",
        },
        "dataset": {
            "wikdict_en_fr_extracted": "wikdict_en_fr",
            "cc_cedict_zh_en_extracted": "cc_cedict_zh_en",
            "random_word_pairs": "Random Word Pairs",
            "singular_plural_pairs": "Singular Plural Pairs",
        },
    },
)

runs_df["transformation"] = runs_df["transformation"].apply(
    lambda x: x.replace("analytical_", "") if x.startswith("analytical_") else x
)

metrics_interpretation = {
    "mark_translation_acc": "higher_better",
    "test_cos_sim": "higher_better",
    "mse_test_loss": "lower_better",
    "expected_metrics/expected_kabsch_rmsd": "lower_better",
    "pred_same_as_input": "lower_better",
}


# %%
# Global variable to control figure display ranges
SHOW_PLOT_RANGE = (0, 99)


def should_show_plot(plot_index):
    """Check if the plot index is within the specified range to show."""
    start, end = SHOW_PLOT_RANGE
    return start <= plot_index <= end


plot_index = 1

show_tables = False

# %%
df = runs_df
# sort_by = "mark_translation_acc"
# sort_by = "mse_test_loss"
sort_by = "test_cos_sim"
# ascending = True
ascending = False
if show_tables:
    display_top_runs_table(
        df=df,
        metrics=[
            "test_accuracy",
            "test_cos_sim",
            "mse_test_loss",
            "mark_translation_acc",
            "pred_same_as_input",
            # "test_cos_sim_diff.p-value",
            # "test_cos_sim_diff.correlation_coefficient",
        ],
        changed_configs_column_names=changed_configs_column_names,
        sort_by=sort_by,
        metric_interpretation=metrics_interpretation,
        ascending=ascending,
    )

# %%
df = runs_df
# sort_by = "mark_translation_acc"
# sort_by = "mse_test_loss"
sort_by = "test_cos_sim"
# ascending = True
ascending = False
if show_tables:
    display_top_runs_table(
        df=df,
        metrics=[
            "test_accuracy",
            "test_cos_sim",
            "mse_test_loss",
            "mark_translation_acc",
            "pred_same_as_input",
            # "test_cos_sim_diff.p-value",
            # "test_cos_sim_diff.correlation_coefficient",
        ],
        changed_configs_column_names=changed_configs_column_names,
        sort_by=sort_by,
        metric_interpretation=metrics_interpretation,
        ascending=ascending,
    )

# %%
# TODO: okay so seems like we are indeed overfutting? although actually, some of them
# have a negative test_cos_sim_diff.correlation_coefficient. which are the runs that
# have this?

# %%
# TODO: unsure how to handle seeds

# %%
title = "bar chart of mean mark_translation_acc by model, transformation and dataset"
model_order = ["bloom-560m", "bloom-1b1", "bloom-3b", "gpt-2", "gpt-2 medium"]
df = (
    runs_df.groupby(["transformation", "dataset", "model_name"], as_index=False)
    .agg(
        mean_mark_translation_acc=("mark_translation_acc", "mean"),
        std_mark_translation_acc=("mark_translation_acc", "std"),
    )
    .sort_values(
        by="model_name",
        key=lambda x: x.map({model: i for i, model in enumerate(model_order)}),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="mean_mark_translation_acc",
    color="dataset",
    error_y="std_mark_translation_acc",
    title=title,
    labels={"mean_mark_translation_acc": "mean mark translation accuracy"},
    barmode="group",
    facet_col="model_name",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1


# %%
title = "bar chart of mean test_cos_sim by model, transformation and dataset"
model_order = ["bloom-560m", "bloom-1b1", "bloom-3b", "gpt-2", "gpt-2 medium"]
df = (
    runs_df.groupby(["transformation", "dataset", "model_name"], as_index=False)
    .agg(
        mean_test_cos_sim=(
            "test_cos_sim",
            "mean",
        ),
        std_test_cos_sim=("test_cos_sim", "std"),
    )
    .sort_values(
        by="model_name",
        key=lambda x: x.map({model: i for i, model in enumerate(model_order)}),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="mean_test_cos_sim",
    color="dataset",
    error_y="std_test_cos_sim",
    title=title,
    labels={"mean_test_cos_sim": "mean test cosine similarity"},
    barmode="group",
    facet_col="model_name",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
title = "bar chart of mean test accuracy by model, transformation and dataset"
model_order = ["bloom-560m", "bloom-1b1", "bloom-3b", "gpt-2", "gpt-2 medium"]
df = (
    runs_df.groupby(["transformation", "dataset", "model_name"], as_index=False)
    .agg(
        mean_test_accuracy=("test_accuracy", "mean"),
        std_test_accuracy=("test_accuracy", "std"),
    )
    .sort_values(
        by="model_name",
        key=lambda x: x.map({model: i for i, model in enumerate(model_order)}),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="mean_test_accuracy",
    color="dataset",
    error_y="std_test_accuracy",
    title=title,
    labels={"mean_test_accuracy": "mean test accuracy"},
    barmode="group",
    facet_col="model_name",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
title = (
    "bar chart of expected (train) cosine similarity "
    "by model, transformation and dataset"
)
model_order = ["bloom-560m", "bloom-1b1", "bloom-3b", "gpt-2", "gpt-2 medium"]
df = (
    runs_df.groupby(["transformation", "dataset", "model_name"], as_index=False)
    .agg(
        mean_train_cos_sim=("train_cos_sim", "mean"),
        std_train_cos_sim=("train_cos_sim", "std"),
    )
    .sort_values(
        by="model_name",
        key=lambda x: x.map({model: i for i, model in enumerate(model_order)}),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="mean_train_cos_sim",
    color="dataset",
    error_y="std_train_cos_sim",
    title=title,
    labels={"mean_train_cos_sim": "mean train cosine similarity"},
    barmode="group",
    facet_col="model_name",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
title = "bar chart of expected (train) mse by model, transformation and dataset"
model_order = ["bloom-560m", "bloom-1b1", "bloom-3b", "gpt-2", "gpt-2 medium"]
df = (
    runs_df.groupby(["transformation", "dataset", "model_name"], as_index=False)
    .agg(
        mean_train_mse_loss=("train_mse_loss", "mean"),
        std_train_mse_loss=("train_mse_loss", "std"),
    )
    .sort_values(
        by="model_name",
        key=lambda x: x.map({model: i for i, model in enumerate(model_order)}),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="mean_train_mse_loss",
    color="dataset",
    error_y="std_train_mse_loss",
    title=title,
    labels={"mean_train_mse_loss": "mean train mse loss"},
    barmode="group",
    facet_col="model_name",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1


# %%
# title = (
#     "bar chart of mean verify_correlation_coefficient "
#     "by model, transformation and verify seed"
# )
# model_order = [
#     "bloom-560m",
#     "bloom-1b1",
#     "bloom-3b",
#     "gpt-2",
#     "gpt-2 medium",
#     "gpt-2 large",
# ]
# df = (
#     runs_df.query("dataset == 'cc_cedict_zh_en_extracted'")
#     .groupby(["transformation", "dataset", "model_name", "seed"], as_index=False)
#     .agg(
#         verify_correlation_coefficient=(
#             "test_cos_sim_diff.correlation_coefficient",
#             lambda x: -x.mean(),
#         ),
#         std_verify_correlation_coefficient=(
#             "test_cos_sim_diff.correlation_coefficient",
#             "std",
#         ),
#     )
#     .sort_values(
#         by="model_name",
#         key=lambda x: x.map({model: i for i, model in enumerate(model_order)}),
#     )
# )

# fig = px.bar(
#     df,
#     x="transformation",
#     y="verify_correlation_coefficient",
#     color="seed",
#     error_y="std_verify_correlation_coefficient",
#     title=title,
#     labels={"verify_correlation_coefficient": "verify correlation coefficient"},
#     barmode="group",
#     facet_col="model_name",
# )

# if should_show_plot(plot_index):
#     fig.show(config={"responsive": True})
# plot_index += 1
# %%
# for the chinese dataset, let's split by the different layernorm settings
title = (
    "mean mark_translation_acc by transformation, embed_ln, and unembed_ln. "
    "chinese dataset only."
)
df = (
    runs_df.query("dataset == 'cc_cedict_zh_en_extracted'")
    .groupby(
        ["transformation", "embed_ln_weights", "unembed_ln_weights"], as_index=False
    )
    .agg(
        mean_mark_translation_acc=("mark_translation_acc", "mean"),
        std_mark_translation_acc=("mark_translation_acc", "std"),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="mean_mark_translation_acc",
    color="transformation",
    error_y="std_mark_translation_acc",
    title=title,
    labels={"mean_mark_translation_acc": "mean mark translation accuracy"},
    barmode="group",
    facet_col="embed_ln_weights",
    facet_row="unembed_ln_weights",
    height=900,
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
title = (
    "mean mark_translation_acc by transformation, embed_ln, and unembed_ln. "
    "french dataset only."
)
df = (
    runs_df.query("dataset == 'wikdict_en_fr_extracted'")
    .groupby(
        ["transformation", "embed_ln_weights", "unembed_ln_weights"], as_index=False
    )
    .agg(
        mean_mark_translation_acc=("mark_translation_acc", "mean"),
        std_mark_translation_acc=("mark_translation_acc", "std"),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="mean_mark_translation_acc",
    color="transformation",
    error_y="std_mark_translation_acc",
    title=title,
    labels={"mean_mark_translation_acc": "mean mark translation accuracy"},
    barmode="group",
    facet_col="embed_ln_weights",
    facet_row="unembed_ln_weights",
    height=900,
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1


# %%
title = (
    "mean cosine similarity by transformation, embed_ln, and unembed_ln. "
    "french dataset only"
)
df = (
    runs_df.query("dataset == 'wikdict_en_fr_extracted'")
    .groupby(
        ["transformation", "embed_ln_weights", "unembed_ln_weights"], as_index=False
    )
    .agg(
        mean_test_cos_sim=("test_cos_sim", lambda x: x.mean()),
        std_test_cos_sim=("test_cos_sim", lambda x: x.std()),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="mean_test_cos_sim",
    color="transformation",
    error_y="std_test_cos_sim",
    title=title,
    labels={"mean_test_cos_sim": "mean test cosine similarity"},
    barmode="group",
    facet_col="embed_ln_weights",
    facet_row="unembed_ln_weights",
    height=900,
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
# %%
annotation_text = """
"""
df = (
    runs_df
    # .query("seed == ")
    # .query("loss_function == 'mse_loss'")
    # .query("loss_function == 'cos_sim'")
    .query("dataset == 'cc_cedict_zh_en_extracted'")
    # .query("dataset == 'wikdict_en_fr_extracted'")
    # .query("embed_ln_weights == 'model_weights'")
    # .query("unembed_ln_weights == 'model_weights'")
    # .query(
    #     "transformation == 'torch_scale_analytical' or "
    #     "transformation == 'torch_analytical'"
    # )
    # .query(
    #     "transformation == 'analytical_rotation' or "
    #     "transformation == 'roma_analytical'"
    # )
    # .query(
    #     "transformation == 'roma_scale_analytical' or "
    #     "transformation == 'roma_analytical'"
    # )
    # .query(
    #     "transformation == 'roma_analytical' or "
    #     "transformation == 'analytical_rotation'"
    # )
    # .query("transformation.str.endswith('analytical')")
    .filter(items=numerical_filter)
    # .pipe(lambda x: (display(x), x)[1])
    # .pipe(lambda x: (print(x.dtypes), x)[1])
    .groupby(
        [
            "model_name",
            "dataset",
            "transformation",
            "embed_ln_weights",
            "unembed_ln_weights",
            "seed",
        ]
    )
    # .pipe(lambda x: (print(x), x)[1])
    .mean().reset_index()
)
sort_by = "mark_translation_acc"
# sort_by = "mse_test_loss"
# sort_by = "cos_sim_test_loss"
# ascending = True
ascending = False
if show_tables:
    display_top_runs_table(
        df=df,
        metrics=[
            "test_cos_sim",
            "mark_translation_acc",
            "mse_test_loss",
            "pred_same_as_input",
            "test_accuracy",
            # "test_cos_sim_diff.p-value",
            # "test_cos_sim_diff.correlation_coefficient",
        ],
        changed_configs_column_names=[
            col for col in changed_configs_column_names if col != "seed"
        ],
        sort_by=sort_by,
        metric_interpretation=metrics_interpretation,
        ascending=ascending,
    )

# %%
title = "parallel categories plot for mark_translation_acc. chinese dataset only"
annotation_text = """
"""
df = runs_df.query("dataset == 'cc_cedict_zh_en_extracted'")[
    changed_configs_column_names + ["mark_translation_acc"]
]
fig = create_parallel_categories_plot(
    df,
    dimensions=changed_configs_column_names,
    color="mark_translation_acc",
    title=title,
    annotation_text=annotation_text,
    invert_colors=True,
    parcat_kwargs={"height": 800},
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
title = "parallel categories plot for test_cos_sim. chinese dataset only"
annotation_text = """
"""
df = runs_df.query("dataset == 'cc_cedict_zh_en_extracted'")[
    changed_configs_column_names + ["test_cos_sim"]
]
fig = create_parallel_categories_plot(
    df,
    dimensions=changed_configs_column_names,
    color="test_cos_sim",
    title=title,
    annotation_text=annotation_text,
    invert_colors=True,
    parcat_kwargs={"height": 800},
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
title = "parallel categories plot for test_cos_sim"
annotation_text = """
"""
df = runs_df[changed_configs_column_names + ["test_cos_sim"]]
fig = create_parallel_categories_plot(
    df,
    dimensions=changed_configs_column_names,
    color="test_cos_sim",
    title=title,
    annotation_text=annotation_text,
    invert_colors=True,
    parcat_kwargs={"height": 800},
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
# Prepare data
X = runs_df[changed_configs_column_names]
X = pd.get_dummies(X, prefix_sep=": ")
y = runs_df["test_accuracy"]

# SHAP for Feature Importance
model = RandomForestRegressor().fit(X, y)
X_numeric = X.astype(float)
explainer = shap.Explainer(model, X_numeric)
shap_values = explainer(X_numeric)
shap.summary_plot(shap_values, X_numeric)

# %% translation
title = "parallel categories plot for test_accuracy, gpt models only"
annotation_text = """
"""
df = (
    runs_df[
        [col for col in changed_configs_column_names if col != "embed_ln_weights"]
        + ["test_accuracy"]
    ]
    .query("dataset == 'wikdict_en_fr_extracted'")
    .query("model_name == 'gpt2' or model_name == 'gpt2-medium'")
)
fig = create_parallel_categories_plot(
    df,
    dimensions=[
        col for col in changed_configs_column_names if col != "embed_ln_weights"
    ],
    color="test_accuracy",
    title=title,
    annotation_text=annotation_text,
    invert_colors=True,
    parcat_kwargs={"height": 800},
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
title = "parallel categories plot for cos_sim. translation only."
annotation_text = """
"""
df = (
    runs_df[changed_configs_column_names + ["test_cos_sim"]]
    .query("dataset == 'wikdict_en_fr_extracted'")
    .query("transformation == 'translation'")
)
fig = create_parallel_categories_plot(
    df,
    dimensions=changed_configs_column_names,
    color="test_cos_sim",
    title=title,
    annotation_text=annotation_text,
    invert_colors=True,
    parcat_kwargs={"height": 800},
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
title = ""
