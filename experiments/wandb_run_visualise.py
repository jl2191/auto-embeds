# %%
import plotly.express as px

from auto_embeds.utils.plot import create_parallel_categories_plot
from auto_embeds.utils.wandb import fetch_wandb_runs, process_wandb_runs_df

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass

# %%
original_df = fetch_wandb_runs(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-05-03 analytical and ln", "experiment 5"],
    get_artifacts=True,
)

runs_df = process_wandb_runs_df(original_df)

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
fig = create_parallel_categories_plot(
    runs_df,
    dimensions=[
        "transformation",
        "seed",
        "embed_weight",
        "embed_ln_weights",
        "unembed_weight",
        "unembed_ln_weights",
    ],
    color="test_accuracy",
    title="Parallel Categories Plot for Test Accuracy",
    annotation_text="Parallel Categories Plot for Test Accuracy",
)

# %%
fig = create_parallel_categories_plot(
    runs_df,
    dimensions=[
        "transformation",
        "seed",
        "embed_weight",
        "embed_ln_weights",
        "unembed_weight",
        "unembed_ln_weights",
    ],
    color="mark_translation_acc",
    title="Parallel Categories Plot for Mark Translation Accuracy",
    annotation_text="Parallel Categories Plot for Mark Translation Accuracy",
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
