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

visualise_all_run_groups = True
tags = experiment_config["neptune"]["tags"]
if visualise_all_run_groups:
    tags = [tag for tag in tags if "run group" not in tag]

fetch_neptune_runs_df.clear_cache()
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

runs_df = runs_df.query('transformation != "analytical_rotation"')

rename_labels = {
    "transformation": {
        "roma_scale_analytical": "analytical_rotation_scale",
        "roma_analytical": "analytical_rotation",
    }
}
runs_df.replace(rename_labels, inplace=True)

runs_df["transformation"] = runs_df["transformation"].apply(
    lambda x: x.replace("analytical_", "") if x.startswith("analytical_") else x
)


metrics_interpretation = {
    "mark_translation_acc": "higher_better",
    "cosine_similarity_test_loss": "lower_better",
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

# %%
df = runs_df
# sort_by = "mark_translation_acc"
# sort_by = "mse_test_loss"
sort_by = "cosine_similarity_test_loss"
ascending = True
# ascending = False
display_top_runs_table(
    df=df,
    metrics=[
        "cosine_similarity_test_loss",
        "mark_translation_acc",
        "mse_test_loss",
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
# scatter plot of mark_translation_acc by transformation and dataset
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
    title="bar chart of mean mark_translation_acc by model, transformation and dataset",
    labels={"mean_mark_translation_acc": "mean mark translation accuracy"},
    barmode="group",
    facet_col="model_name",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
# scatter plot of mark_translation_acc by transformation + dataset for gpt2 models only
df = (
    runs_df.query("model_name == 'gpt2' or model_name == 'gpt2-medium'").groupby(
        ["transformation", "dataset", "model_name"], as_index=False
    )
    # .agg(
    #     mean_cosine_similarity=("cosine_similarity_test_loss", lambda x: -x.mean()),
    #     std_cosine_similarity=("cosine_similarity_test_loss", lambda x: x.std()),
    # )
)

fig = px.bar(
    df,
    x="transformation",
    y="cosine_similarity_test_loss",
    color="dataset",
    title="bar chart of mean mark_translation_acc by model, transformation and dataset",
    labels={"mean_mark_translation_acc": "mean mark translation accuracy"},
    barmode="group",
    facet_col="model_name",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
# scatter plot of mark_translation_acc by transformation and dataset
model_order = ["bloom-560m", "bloom-1b1", "bloom-3b", "gpt-2", "gpt-2 medium"]
df = (
    runs_df.groupby(["transformation", "dataset", "model_name"], as_index=False)
    .agg(
        cosine_similarity_test_loss=(
            "cosine_similarity_test_loss",
            lambda x: -x.mean(),
        ),
        std_cosine_similarity_test_loss=("cosine_similarity_test_loss", "std"),
    )
    .sort_values(
        by="model_name",
        key=lambda x: x.map({model: i for i, model in enumerate(model_order)}),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="cosine_similarity_test_loss",
    color="dataset",
    error_y="std_cosine_similarity_test_loss",
    title="bar chart of mean cosine similarity by model, transformation and dataset",
    labels={"mean_mark_translation_acc": "mean mark translation accuracy"},
    barmode="group",
    facet_col="model_name",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
# scatter plot of mark_translation_acc by transformation and dataset
model_order = ["bloom-560m", "bloom-1b1", "bloom-3b", "gpt-2", "gpt-2 medium"]
df = (
    runs_df.query("dataset == 'cc_cedict_zh_en_extracted'")
    .groupby(["transformation", "dataset", "model_name", "seed"], as_index=False)
    .agg(
        verify_correlation_coefficient=(
            "test_cos_sim_diff.correlation_coefficient",
            lambda x: -x.mean(),
        ),
        std_verify_correlation_coefficient=(
            "test_cos_sim_diff.correlation_coefficient",
            "std",
        ),
    )
    .sort_values(
        by="model_name",
        key=lambda x: x.map({model: i for i, model in enumerate(model_order)}),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="verify_correlation_coefficient",
    color="seed",
    error_y="std_verify_correlation_coefficient",
    title=(
        "bar chart of mean verify_correlation_coefficient by model, transformation "
        "and verify seed."
    ),
    labels={"verify_correlation_coefficient": "verify correlation coefficient"},
    barmode="group",
    facet_col="model_name",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
# %%
# TODO: for the chinese dataset, let's split by the different layernorm settings
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
    title=(
        "mean mark_translation_acc by transformation, embed_ln, and unembed_ln. "
        "chinese dataset only."
    ),
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
    title=(
        "mean mark_translation_acc by transformation, embed_ln, and unembed_ln. "
        "french dataset only."
    ),
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
df = (
    runs_df.query("dataset == 'wikdict_en_fr_extracted'")
    .groupby(
        ["transformation", "embed_ln_weights", "unembed_ln_weights"], as_index=False
    )
    .agg(
        mean_cosine_similarity=("cosine_similarity_test_loss", lambda x: -x.mean()),
        std_cosine_similarity=("cosine_similarity_test_loss", lambda x: x.std()),
    )
)

fig = px.bar(
    df,
    x="transformation",
    y="mean_cosine_similarity",
    color="transformation",
    error_y="std_cosine_similarity",
    title=(
        "mean cosine similarity by transformation, embed_ln, and unembed_ln. "
        "french dataset only"
    ),
    labels={"mean_cosine_similarity": "mean cosine similarity"},
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
    # .query("loss_function == 'cosine_similarity'")
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
# sort_by = "cosine_similarity_test_loss"
# ascending = True
ascending = False
display_top_runs_table(
    df=df,
    metrics=[
        "cosine_similarity_test_loss",
        "mark_translation_acc",
        "mse_test_loss",
        "pred_same_as_input",
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
annotation_text = """
huh interesting, it seems like on mark_translation_loss the analytical_linear_map is
performing very badly!
"""
df = runs_df[changed_configs_column_names + ["mark_translation_acc"]]
fig = create_parallel_categories_plot(
    df,
    dimensions=changed_configs_column_names,
    color="mark_translation_acc",
    title="parallel categories plot for mark_translation_acc",
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
y = runs_df["mark_translation_acc"]

# SHAP for Feature Importance
model = RandomForestRegressor().fit(X, y)
X_numeric = X.astype(float)
explainer = shap.Explainer(model, X_numeric)
shap_values = explainer(X_numeric)
shap.summary_plot(shap_values, X_numeric)

# %%
# Prepare data
X = runs_df[changed_configs_column_names]
X = pd.get_dummies(X, prefix_sep=": ")
y = runs_df["test_cos_sim_diff.correlation_coefficient"]

# %%
# SHAP for Feature Importance
model = RandomForestRegressor().fit(X, y)
X_numeric = X.astype(float)
explainer = shap.Explainer(model, X_numeric)
shap_values = explainer(X_numeric)
shap.summary_plot(shap_values, X_numeric)

# %% translation

annotation_text = """
"""
df = (
    runs_df[changed_configs_column_names + ["mark_translation_acc"]].query(
        "dataset == 'wikdict_en_fr_extracted'"
    )
    # .query("transformation == 'translation'")
    .query("model_name == 'gpt2' or model_name == 'gpt2-medium'")
)
fig = create_parallel_categories_plot(
    df,
    dimensions=changed_configs_column_names,
    color="mark_translation_acc",
    title="parallel categories plot for mark_translation_acc, gpt models only",
    annotation_text=annotation_text,
    invert_colors=True,
    parcat_kwargs={"height": 800},
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
annotation_text = """
"""
df = (
    runs_df.assign(cosine_similarity=lambda x: -x["cosine_similarity_test_loss"])[
        changed_configs_column_names + ["cosine_similarity"]
    ]
    .query("dataset == 'wikdict_en_fr_extracted'")
    .query("transformation == 'translation'")
)
fig = create_parallel_categories_plot(
    df,
    dimensions=changed_configs_column_names,
    color="cosine_similarity",
    title="parallel categories plot for cosine_similarity",
    annotation_text=annotation_text,
    invert_colors=True,
    parcat_kwargs={"height": 800},
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
annotation_text = """
"""
df = (
    runs_df.assign(cosine_similarity=lambda x: -x["cosine_similarity_test_loss"])[
        changed_configs_column_names + ["cosine_similarity"]
    ]
    .query("dataset == 'wikdict_en_fr_extracted'")
    .query("model_name == 'gpt2' or model_name == 'gpt2-medium'")
    .query("transformation != 'translation'")
    # .query("seed == '3'")
)
fig = create_parallel_categories_plot(
    df,
    dimensions=changed_configs_column_names,
    color="cosine_similarity",
    title=(
        "parallel categories plot for cosine_similarity, for gpt2 models on the "
        "french dataset"
    ),
    annotation_text=annotation_text,
    invert_colors=True,
    parcat_kwargs={"height": 800},
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
