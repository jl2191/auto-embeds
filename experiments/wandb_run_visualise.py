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
difference_df = get_difference_df(
    runs_df,
    configs_that_change_names,
    comparison_config=("rotation", "analytical_rotation"),
    "cosine_similarity_test_loss",
)
fig = create_parallel_categories_plot(
    difference_df,
    dimensions=[name for name in configs_that_change_names if name != "transformation"],
    color="difference",
    title="Parallel Categories Plot for Cosine Similarity Test Loss",
    annotation_text="Parallel Categories Plot for Cosine Similarity Test Loss",
)

# %%
query = "difference > 0"
difference_df = get_difference_df(
    runs_df, configs_that_change_names, "mark_translation_acc"
)
fig = create_parallel_categories_plot(
    difference_df.query(query),
    dimensions=[name for name in configs_that_change_names if name != "transformation"],
    color="difference",
    title="Parallel Categories Plot for Cosine Similarity Test Loss",
    annotation_text="Parallel Categories Plot for Cosine Similarity Test Loss",
)

# %%

query = "transformation in ['rotation', 'analytical_rotation'] and unembed_ln_weights in ['no_ln']"
fig = create_parallel_categories_plot(
    runs_df.query(query),
    dimensions=[
        "transformation",
        "embed_ln_weights",
        "unembed_ln_weights",
    ],
    color="test_accuracy",
    title="Parallel Categories Plot for Test Accuracy",
    annotation_text="Parallel Categories Plot for Test Accuracy",
)

# %%
query = "transformation in ['rotation', 'analytical_rotation'] and unembed_ln_weights in ['no_ln']"
fig = create_parallel_categories_plot(
    runs_df.query(query),
    dimensions=[
        "transformation",
        "embed_ln_weights",
        "unembed_ln_weights",
    ],
    color="mark_translation_acc",
    title="Parallel Categories Plot for Mark Translation Accuracy",
    annotation_text="Parallel Categories Plot for Mark Translation Accuracy",
)

# %%

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
