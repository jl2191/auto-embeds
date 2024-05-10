# %%
from auto_embeds.utils.neptune import (
    fetch_neptune_runs,
    list_changed_configs,
    process_neptune_runs_df,
)
from auto_embeds.utils.plot import create_parallel_categories_plot
from experiments.scratch_funcs import plot_difference

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass

# %%
original_df = fetch_neptune_runs(
    project_name="jl2191/language-transformations",
    tags=[
        "actual",
        "2024-05-09 translation and linear map",
        "experiment 1",
        "run group 3",
    ],
    get_artifacts=True,
)

changed_configs = list_changed_configs(original_df)
configs_that_change_names = ["dataset"] + [
    key for key in changed_configs.keys() if key not in ["embed_ln", "unembed_ln"]
]
configs_that_change_values = [changed_configs[key] for key in configs_that_change_names]

# %%
runs_df = process_neptune_runs_df(original_df, has_plot=False)
runs_df = runs_df.drop(columns=["summary", "config", "history"]).reset_index(drop=True)
runs_df = runs_df.query("dataset not in ['singular_plural_pairs', 'random_word_pairs']")

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
# Global variable to control figure display ranges
SHOW_PLOT_RANGE = (0, 99)


def should_show_plot(plot_index):
    """Check if the plot index is within the specified range to show."""
    start, end = SHOW_PLOT_RANGE
    return start <= plot_index <= end


plot_index = 1

# %%
"""
"""
df = (
    runs_df[configs_that_change_names + ["mark_translation_acc"]].query(
        "dataset == 'cc_cedict_zh_en_extracted'"
    )
    # .query("embed_ln_weights != model_weights")
    # .query("loss_function == 'cosine_similarity'")
    # .dropna()
    # .reset_index(drop=True)
)
fig = create_parallel_categories_plot(
    df,
    dimensions=configs_that_change_names,
    color="mark_translation_acc",
    title="Parallel Categories Plot for mark_translation_acc",
    annotation_text="",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
"""
"""

# %%
df = (
    runs_df[configs_that_change_names + ["cosine_similarity_test_loss"]].query(
        "dataset == 'cc_cedict_zh_en_extracted'"
    )
    # .query("embed_ln_weights != model_weights")
    # .query("loss_function == 'cosine_similarity'")
    # .dropna()
    # .reset_index(drop=True)
)
fig = create_parallel_categories_plot(
    df,
    dimensions=configs_that_change_names,
    color="cosine_similarity_test_loss",
    title="Parallel Categories Plot for cosine_similarity_test_loss",
    annotation_text="",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
print(
    """
    """
)

# %%
df = (
    runs_df[configs_that_change_names + ["mse_test_loss"]]
    # .query("transformation in ['analytical_linear_map']")
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
    # .query("embed_ln_weights != model_weights")
    # .query("loss_function == 'cosine_similarity'")
    # .dropna()
    # .reset_index(drop=True)
)
fig = create_parallel_categories_plot(
    df,
    dimensions=configs_that_change_names,
    color="mse_test_loss",
    title="Parallel Categories Plot for mse_test_loss",
    annotation_text="",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
print(
    """
    hmm it seems like the case that analytical_linear_map is indeed doing as well as the
    linear map for the mse loss but this does not translate over to mark_translation_acc
    """
)

# %%
"""
"""
df = (
    runs_df[configs_that_change_names + ["mark_translation_acc"]]
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
    # .query("embed_ln_weights != model_weights")
    # .query("loss_function == 'cosine_similarity'")
    # .dropna()
    # .reset_index(drop=True)
)
fig = create_parallel_categories_plot(
    df,
    dimensions=configs_that_change_names,
    color="mark_translation_acc",
    title="Parallel Categories Plot for mark_translation_acc",
    annotation_text="",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
"""
"""

# %%
fig = plot_difference(
    df=runs_df[configs_that_change_names + ["mse_test_loss"]],
    query="loss_function == 'mse_loss'",
    configs_that_change_names=configs_that_change_names,
    comparison_name="transformation",
    comparison_values=("analytical_linear_map", "linear_map"),
    metric="mse_test_loss",
    annotation_text="",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
