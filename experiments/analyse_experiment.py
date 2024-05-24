# %%
import pandas as pd
from IPython.core.getipython import get_ipython

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
    tags = [tag for tag in tags if tag != "run group"]

# fetch_neptune_runs_df.clear_cache()
original_df = fetch_neptune_runs_df(
    project_name=project_name,
    tags=tags,
    get_artifacts=True,
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
        "test_cos_sim_diff.p-value",
        "test_cos_sim_diff.correlation_coefficient",
    ],
    changed_configs_column_names=changed_configs_column_names,
    sort_by=sort_by,
    metric_interpretation=metrics_interpretation,
    ascending=ascending,
)

# %%
annotation_text = """
"""
df = (
    runs_df
    # .query("seed == ")
    # .query("loss_function == 'mse_loss'")
    # .query("loss_function == 'cosine_similarity'")
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
    .query("dataset == 'wikdict_en_fr_extracted'")
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
import shap
from sklearn.ensemble import RandomForestRegressor

# Prepare data
X = runs_df[changed_configs_column_names]
display(X)
X = pd.get_dummies(X, prefix_sep=": ")
display(X)
y = runs_df["mark_translation_acc"]

# %%
# SHAP for Feature Importance
model = RandomForestRegressor().fit(X, y)
X_numeric = X.astype(float)
explainer = shap.Explainer(model, X_numeric)
shap_values = explainer(X_numeric)
shap.summary_plot(shap_values, X_numeric)

# %%
# Correlation Analysis
df = runs_df[changed_configs_column_names + ["mark_translation_acc"]]
df = pd.get_dummies(df)
correlation = df.corr()["mark_translation_acc"].sort_values(ascending=False)
print("Correlation with mark_translation_acc:\n", correlation)

# %%
# Feature Importance using RandomForestRegressor
importances = model.feature_importances_
feature_importance = pd.Series(importances, index=X.columns).sort_values(
    ascending=False
)
print("Feature importance:\n", feature_importance)
