# %%

import plotly.express as px

from auto_embeds.utils.neptune import (
    fetch_neptune_runs_df,
    process_neptune_runs_df,
)
from auto_embeds.utils.plot import (
    create_parallel_categories_plot,
)
from experiments.scratch_funcs import display_top_runs_table, plot_difference

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass

# %%
# fetch_neptune_runs_df.clear_cache()
original_df = fetch_neptune_runs_df(
    project_name="mars/language-transformations",
    tags=[
        "actual",
        "2024-05-14 rotation trials",
        "experiment 2",
    ],
    get_artifacts=True,
)

# %%
runs_df, config_column_names, changed_configs_column_names, result_column_names = (
    process_neptune_runs_df(original_df)
)
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
sort_by = "mse_test_loss"
# sort_by = "cosine_similarity_test_loss"
# sort_by = "mark_translation_acc"
ascending = True
# ascending = False
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

# %%
annotation_text = """
"""
df = runs_df.query("embed_ln_weights != 'default_weights'").query(
    "unembed_ln_weights != 'default_weights'"
)
fig = plot_difference(
    df=df,
    configs_that_change_names=changed_configs_column_names,
    comparison_name="transformation",
    comparison_values=("analytical_linear_map", "linear_map"),
    metric="mse_test_loss",
    annotation_text=annotation_text,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
# %%
annotation_text = """
"""
df = (
    runs_df.query("loss_function == 'mse_loss'")
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
    .query("dataset == 'wikdict_en_fr_extracted'")
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
    # .query("transformation == 'kabsch_analytical'")
)
sort_by = "mark_translation_acc"
# sort_by = "mse_test_loss"
# ascending = True
ascending = False
display_top_runs_table(
    df=df,
    metrics=[
        "cosine_similarity_test_loss",
        "mark_translation_acc",
        "mse_test_loss",
        "test_cos_sim_diff.p-value",
        "expected_metrics/expected_kabsch_rmsd",
    ],
    changed_configs_column_names=changed_configs_column_names,
    sort_by=sort_by,
    metric_interpretation=metrics_interpretation,
    ascending=ascending,
)

# %%
import plotly.io as pio

plot_json = runs_df.query("run_id == 'LAN-939'")["cos_sims_trend_plot"].iloc[0]
cos_sims_trend_plot = pio.from_json(plot_json).show()
# %%
annotation_text = """
"""
df = runs_df.query("loss_function == 'mse_loss'")
display_top_runs_table(
    df=df,
    metrics=[
        "cosine_similarity_test_loss",
        "mark_translation_acc",
        "mse_test_loss",
        "test_cos_sim_diff.p-value",
        "expected_metrics/expected_kabsch_rmsd",
    ],
    changed_configs_column_names=changed_configs_column_names,
    sort_by="expected_metrics/expected_kabsch_rmsd",
    metric_interpretation=metrics_interpretation,
    ascending=False,
)
# %%

# %%
annotation_text = """
"""
df = runs_df.query("loss_function == 'mse_loss'")
display_top_runs_table(
    df=df,
    metrics=["cosine_similarity_test_loss", "mark_translation_acc", "mse_test_loss"],
    changed_configs_column_names=changed_configs_column_names,
    sort_by="mse_test_loss",
    metric_interpretation=metrics_interpretation,
    ascending=True,
)
# %%
# plotting correlation between mse_test_loss and mark_translation_acc with facets for
# embed_ln and unembed_ln
df = (
    runs_df
    # .query("embed_ln_weights == 'model_weights' or embed_ln_weights == 'default_weights'")
    # .query("mark_translation_acc > 0.2")
    # .query("unembed_ln == 'model_weights'")
    # .query(
    #     "transformation == 'linear_map' or transformation == 'analytical_linear_map'"
    # )
    # .query(
    #     "transformation == 'translation' or transformation == 'analytical_translation'"
    # )
)
df_mse = df.query("loss_function == 'mse_loss'")
fig_mse = px.scatter(
    df_mse,
    x="mse_test_loss",
    y="mark_translation_acc",
    color="transformation",
    facet_col="embed_ln",
    facet_row="unembed_ln",
    trendline="ols",
    title="correlation between mse_test_loss and mark_translation_acc",
    height=600,
    log_x=True,
)
if should_show_plot(plot_index):
    fig_mse.show(config={"responsive": True})
plot_index += 1

df_cosine = df.query("loss_function == 'cosine_similarity'")
fig_cosine = px.scatter(
    df_cosine,
    x="cosine_similarity_test_loss",
    y="mark_translation_acc",
    color="transformation",
    facet_col="embed_ln",
    facet_row="unembed_ln",
    trendline="ols",
    title="correlation between cosine_similarity_test_loss and mark_translation_acc",
    height=600,
)
if should_show_plot(plot_index):
    fig_cosine.show(config={"responsive": True})
plot_index += 1

# %%
# plotting correlation between mse_test_loss and mark_translation_acc with facets for
# embed_ln and unembed_ln
df = (
    runs_df
    # .query("embed_ln_weights == 'model_weights' or embed_ln_weights == 'default_weights'")
    # .query("mark_translation_acc > 0.2")
    # .query("unembed_ln == 'model_weights'")
    # .query(
    #     "transformation == 'linear_map' or transformation == 'analytical_linear_map'"
    # )
    # .query(
    #     "transformation == 'translation' or transformation == 'analytical_translation'"
    # )
    .query("transformation.str.endswith('analytical')").query(
        "transformation != 'kabsch_analytical'"
    )
)
df_mse = df.query("loss_function == 'mse_loss'")
fig_mse = px.scatter(
    df_mse,
    x="mse_test_loss",
    y="mark_translation_acc",
    color="transformation",
    facet_col="embed_ln",
    facet_row="unembed_ln",
    trendline="ols",
    title="correlation between mse_test_loss and mark_translation_acc",
    height=600,
    log_x=True,
)
if should_show_plot(plot_index):
    fig_mse.show(config={"responsive": True})
plot_index += 1

df_cosine = df.query("loss_function == 'cosine_similarity'")
fig_cosine = px.scatter(
    df_cosine,
    x="cosine_similarity_test_loss",
    y="mark_translation_acc",
    color="transformation",
    facet_col="embed_ln",
    facet_row="unembed_ln",
    trendline="ols",
    title="correlation between cosine_similarity_test_loss and mark_translation_acc",
    height=600,
)
if should_show_plot(plot_index):
    fig_cosine.show(config={"responsive": True})
plot_index += 1

# %%
df_mse = runs_df.query("transformation != 'torch_scale_analytical'")
df_mse = df_mse.query(
    "transformation not in ['analytical_translation', 'analytical_linear_map']"
)
fig_mse = (
    px.scatter(
        df_mse,
        x="mse_test_loss",
        y="mark_translation_acc",
        color="transformation",
        facet_col="embed_ln",
        facet_row="unembed_ln",
        trendline="ols",
        title="correlation between mse_test_loss and mark_translation_acc",
        height=600,
        log_x=True,
    )
    .update_layout(margin=dict(b=125))
    .add_annotation(
        text="excluding analytical_translation and analytical_linear_map",
        align="left",
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13),
        x=0,
        y=-0.25,
    )
)
if should_show_plot(plot_index):
    fig_mse.show(config={"responsive": True})
plot_index += 1

# %%
# plotting correlation between mse_test_loss and mark_translation_acc with facets for
# embed_ln and unembed_ln
df_mse = runs_df.query("loss_function == 'mse_loss'")
# categorize transformations based on the presence of 'analytical' in their names
df_mse["color_group"] = df_mse["transformation"].apply(
    lambda x: "analytical" if "analytical" in x else "learned"
)
fig_mse = px.scatter(
    df_mse,
    x="mse_test_loss",
    y="mark_translation_acc",
    color="color_group",
    facet_col="embed_ln",
    facet_row="unembed_ln",
    trendline="ols",
    title="correlation between mse_test_loss and mark_translation_acc",
    height=600,
)
if should_show_plot(plot_index):
    fig_mse.show(config={"responsive": True})
plot_index += 1

# plotting correlation between cosine_similarity_test_loss and mark_translation_acc
# with facets for embed_ln and unembed_ln
df_cosine = runs_df.query("loss_function == 'cosine_similarity'")
# categorize transformations based on the presence of 'analytical' in their names
df_cosine["color_group"] = df_cosine["transformation"].apply(
    lambda x: "analytical" if "analytical" in x else "learned"
)
fig_cosine = px.scatter(
    df_cosine,
    x="cosine_similarity_test_loss",
    y="mark_translation_acc",
    color="color_group",
    facet_col="embed_ln",
    facet_row="unembed_ln",
    trendline="ols",
    title="correlation between cosine_similarity_test_loss and mark_translation_acc",
    height=600,
)
if should_show_plot(plot_index):
    fig_cosine.show(config={"responsive": True})
plot_index += 1
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
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
annotation_text = """
oh, this looks good for our analytical least squares solution! let's do this as well
for mse loss.
"""
fig = plot_difference(
    df=runs_df,
    configs_that_change_names=changed_configs_column_names,
    comparison_name="transformation",
    comparison_values=("analytical_linear_map", "linear_map"),
    metric="cosine_similarity_test_loss",
    annotation_text=annotation_text,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
annotation_text = """
"""
df = (
    runs_df.query("transformation == 'translation' or transformation == 'identity'")
    # .query("loss_function == 'mse_loss'"),
    # .query("transformation == 'torch_analytical'"),
)
fig = plot_difference(
    df=df,
    configs_that_change_names=changed_configs_column_names,
    comparison_name="dataset",
    comparison_values=("wikdict_en_fr_extracted", "cc_cedict_zh_en_extracted"),
    metric="mark_translation_acc",
    annotation_text=annotation_text,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%

# %%
annotation_text = """
and it seems like when it comes to our mark_translation_acc metric, there is no run
between the two where the analytical_linear_map is better than the learned linear_map!
"""
fig = plot_difference(
    df=runs_df,
    configs_that_change_names=changed_configs_column_names,
    comparison_name="transformation",
    comparison_values=("analytical_linear_map", "linear_map"),
    metric="mark_translation_acc",
    annotation_text=annotation_text,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
