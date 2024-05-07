# %%
import plotly.express as px

from auto_embeds.utils.plot import create_parallel_categories_plot
from auto_embeds.utils.wandb import (
    fetch_wandb_runs,
    get_difference_df,
    list_changed_configs,
    process_wandb_runs_df,
)

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass

# %%
original_df = fetch_wandb_runs(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-05-05 analytical and ln", "experiment 2"],
    get_artifacts=True,
)

changed_configs = list_changed_configs(original_df)
configs_that_change_names = [
    key for key in changed_configs.keys() if key not in ["embed_ln", "unembed_ln"]
]
configs_that_change_values = [changed_configs[key] for key in configs_that_change_names]

# %%
runs_df = process_wandb_runs_df(original_df)
runs_df = runs_df.drop(columns=["summary", "config", "history"]).reset_index(drop=True)
runs_df = runs_df.query("dataset != 'singular_plural_pairs'")

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

# %%
runs_df = runs_df.query(
    "dataset not in ['singular_plural_pairs', 'random_word_pairs']"
).reset_index(drop=True)


# %%
def plot_difference(
    query,
    comparison_name,
    comparison_values,
    metric,
    annotation_text,
    invert_colors=False,
):
    # Only perform the query if it is provided, otherwise use the full dataframe
    if query:
        filtered_df = runs_df.query(query)
    else:
        filtered_df = runs_df

    difference_df = get_difference_df(
        df=filtered_df,
        configs_that_change_names=configs_that_change_names,
        comparison_name=comparison_name,
        comparison_values=comparison_values,
        metric=metric,
    )
    negative_values_exist = (difference_df["difference"] < 0).any()
    if negative_values_exist:
        print(f"There are negative differences in {metric}.")
    else:
        print(f"All differences in {metric} are non-negative.")
    fig = create_parallel_categories_plot(
        difference_df,
        dimensions=[
            name for name in configs_that_change_names if name != comparison_name
        ],
        color="difference",
        title=f"Parallel Categories Plot for {metric}",
        annotation_text=annotation_text,
        difference=True,
        invert_colors=True,
    )
    return fig


# %%
fig = plot_difference(
    query="loss_function == 'cosine_similarity'",
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="cosine_similarity_test_loss",
    annotation_text="filters: loss_function = cosine_similarity",
)

# %%
fig = plot_difference(
    query="loss_function == 'mse_loss'",
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="mse_test_loss",
    annotation_text="filters: loss_function = mse_loss",
)

# %%
fig = plot_difference(
    query="",
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="cosine_similarity_test_loss",
    annotation_text="filters: None",
)

# %%
fig = plot_difference(
    query="",
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="mse_test_loss",
    annotation_text="filters: None",
)

# %%
fig = create_parallel_categories_plot(
    runs_df,
    dimensions=configs_that_change_names,
    color="cosine_similarity_test_loss",
    title="Parallel Categories Plot for cosine_similarity_test_loss",
    annotation_text="filters: None",
    invert_colors=True,
)

# %%
fig = create_parallel_categories_plot(
    runs_df,
    dimensions=configs_that_change_names,
    color="mse_test_loss",
    title="Parallel Categories Plot for mse_test_loss",
    annotation_text="filters: None",
)
# %%
fig = px.bar(
    runs_df,
    x="test_accuracy",
    y="mark_translation_acc",
    color="transformation",
    title="Test Accuracy vs Mark Translation Accuracy",
).show()

# %%
fig = px.bar(
    runs_df,
    x="transformation",
    y="cosine_similarity_test_loss",
    color="transformation",
    title="Test Accuracy vs Mark Translation Accuracy",
).show()
