# %%
from IPython.core.getipython import get_ipython

from auto_embeds.utils.neptune import (
    fetch_neptune_runs_df,
    process_neptune_runs_df,
)
from experiments.configure_experiment import experiment_config
from experiments.scratch_funcs import display_top_runs_table

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass


# %%
project_name = "mars/language-transformations"

visualise_all_run_groups = False
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

metrics_interpretation = {
    "mark_translation_acc": "higher_better",
    "cosine_similarity_test_loss": "lower_better",
    "mse_test_loss": "lower_better",
    "expected_metrics/expected_kabsch_rmsd": "lower_better",
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
annotation_text = """
"""
df = (
    runs_df
    # .query("seed == ")
    # .query("loss_function == 'mse_loss'")
    # .query("loss_function == 'cosine_similarity'")
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
    # .query("dataset == 'wikdict_en_fr_extracted'")
    # .query("embed_ln_weights == 'model_weights'")
    # .query("unembed_ln_weights == 'model_weights'")
    # .query(
    #     "transformation == 'torch_scale_analytical' or "
    #     "transformation == 'torch_analytical'"
    # )
    # .query(
    #     "transformation == 'roma_scale_analytical' or "
    #     "transformation == 'roma_analytical'"
    # )
    .query(
        "transformation == 'roma_analytical' or "
        "transformation == 'analytical_rotation'"
    )
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
        "test_cos_sim_diff.p-value",
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
    # .query("loss_function == 'mse_loss'")
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
    # .query("dataset == 'wikdict_en_fr_extracted'")
    # .query("embed_ln_weights == 'model_weights'")
    # .query("unembed_ln_weights == 'model_weights'")
    # .query(
    #     "transformation == 'torch_scale_analytical' or "
    #     "transformation == 'torch_analytical'"
    # )
    # .query(
    #     "transformation == 'linear_map' or transformation == 'analytical_linear_map'"
    # )
    # .query(
    #     "transformation == 'translation' or transformation == 'analytical_translation'"
    # )
    # .query("loss_function == 'cosine_similarity'")
    # .query(
    #     "transformation == 'roma_scale_analytical' or "
    #     "transformation == 'roma_analytical'"
    # )
    .query("transformation.str.endswith('analytical')")
    # .query("transformation.str.contains('roma')")
)
# sort_by = "mse_test_loss"
# sort_by = "cosine_similarity_test_loss"
sort_by = "mark_translation_acc"
# ascending = True
ascending = False
# top_n=500
display_top_runs_table(
    df=df,
    metrics=[
        "cosine_similarity_test_loss",
        "mark_translation_acc",
        "mse_test_loss",
        "test_cos_sim_diff.p-value",
    ],
    changed_configs_column_names=changed_configs_column_names,
    sort_by=sort_by,
    metric_interpretation=metrics_interpretation,
    ascending=ascending,
)
