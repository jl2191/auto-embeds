# %%

import pandas as pd
import plotly.express as px

from auto_embeds.utils.neptune import (
    fetch_neptune_runs_df,
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
original_df = fetch_neptune_runs_df(
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
runs_df = process_neptune_runs_df(original_df)
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

fig = plot_difference(
    df=runs_df,
    query="loss_function == 'cosine_similarity'",
    configs_that_change_names=configs_that_change_names,
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="cosine_similarity_test_loss",
    annotation_text="filters: loss_function = cosine_similarity",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1


# %%
fig = plot_difference(
    df=runs_df,
    query="loss_function == 'mse_loss'",
    configs_that_change_names=configs_that_change_names,
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="mse_test_loss",
    annotation_text="filters: loss_function = mse_loss",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
fig = plot_difference(
    df=runs_df,
    query="",
    configs_that_change_names=configs_that_change_names,
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="cosine_similarity_test_loss",
    annotation_text="filters: None",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
fig = plot_difference(
    df=runs_df,
    query="",
    configs_that_change_names=configs_that_change_names,
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="mse_test_loss",
    annotation_text="filters: None",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
df = (
    runs_df[configs_that_change_names + ["cosine_similarity_test_loss"]]
    .query("transformation in ['analytical_rotation', 'rotation']")
    .dropna()
    .reset_index(drop=True)
)
fig = create_parallel_categories_plot(
    df,
    dimensions=configs_that_change_names,
    color="cosine_similarity_test_loss",
    title="Parallel Categories Plot for cosine_similarity_test_loss",
    annotation_text="filters: transformation in ['analytical_rotation', 'rotation']",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
df = (
    runs_df[configs_that_change_names + ["mark_translation_acc"]]
    .query("transformation in ['analytical_rotation', 'rotation']")
    .dropna()
    .reset_index(drop=True)
)
fig = plot_difference(
    df=df,
    query="",
    configs_that_change_names=configs_that_change_names,
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="mark_translation_acc",
    annotation_text="filters: None",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
df = (
    runs_df[configs_that_change_names + ["mark_translation_acc"]]
    .query(
        "transformation in ['analytical_rotation', 'rotation'] and dataset == 'cc_cedict_zh_en_extracted'"
    )
    .dropna()
    .reset_index(drop=True)
)
fig = create_parallel_categories_plot(
    df,
    dimensions=configs_that_change_names,
    color="mark_translation_acc",
    title="Parallel Categories Plot for mark_translation_acc",
    annotation_text="filters: None",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%

# %%

df = (
    runs_df[configs_that_change_names + ["mark_translation_acc"]]
    .query("transformation in ['analytical_rotation', 'rotation']")
    .query("dataset == 'cc_cedict_zh_en_extracted'")
    # .query("embed_ln_weights != model_weights")
    # .query("loss_function == 'cosine_similarity'")
    # .dropna()
    # .reset_index(drop=True)
)
fig = plot_difference(
    df=df,
    query="",
    configs_that_change_names=configs_that_change_names,
    comparison_name="transformation",
    comparison_values=("analytical_rotation", "rotation"),
    metric="mark_translation_acc",
    annotation_text="filters: None",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
"""
what roughly overall is the best performing in terms of mark_translation_acc? is it
different for cosine_similarity and mse?
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
    annotation_text="filters: None",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
"""
linear map obviously does best, so let's get rid of that
"""

# %%
"""
getting rid of linear map
"""
df = (
    runs_df[configs_that_change_names + ["mark_translation_acc"]].query(
        "transformation != 'linear_map'"
    )
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
    annotation_text="filters: None",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
"""
okay, seems like now the best performing is the analytical rotation
"""
# %%
"""
my guess is that there is a correlation between mark_translation_acc and the loss
functions. i am unsure which one is "better" though, cosine_similarity or mse. let's
plot them both and see if we can see any difference.
"""
df = (
    runs_df[
        configs_that_change_names
        + ["mark_translation_acc", "cosine_similarity_test_loss", "mse_test_loss"]
    ]
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
    # .query("embed_ln_weights != model_weights")
    # .query("loss_function == 'cosine_similarity'")
    # .dropna()
    # .reset_index(drop=True)
)
# Plotting the differences in mark_translation_acc for cosine_similarity and mse
df_cosine = df.query("loss_function == 'cosine_similarity'")
df_mse = df.query("loss_function == 'mse_loss'")

fig_cosine = px.scatter(
    df_cosine,
    x="cosine_similarity_test_loss",
    y="mark_translation_acc",
    color="transformation",
    trendline="ols",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
fig_mse = px.scatter(
    df_mse,
    x="mse_test_loss",
    y="mark_translation_acc",
    color="transformation",
    trendline="ols",
)
if should_show_plot(plot_index):
    fig_mse.show(config={"responsive": True})
plot_index += 1

fig_cosine.show(config={"responsive": True})
print(
    """
for cosine similarity, more negative values are more performant. as such, we would want
to see a negative trendline. translation has a very negative trendline but some of them
seem to have a fairly positive one like linear map.
"""
)
if should_show_plot(plot_index):
    fig_cosine.show(config={"responsive": True})
plot_index += 1

print(
    """
for when we are using mse as our loss function, if there is a good correspondence
between mark translation loss and mse loss, we would want to see a negative gradient
(as you have a bigger mse, you perform worse on mark translation. however, this does
not seem to be the case). as such, i think we should use cosine similarity the thing we
optimise. actually, we should be calculating the overall trendline, not splitting by
transformation!!!
"""
)
# %%
"""
actual comparison of the two loss functions
"""
df = (
    runs_df[
        configs_that_change_names
        + ["mark_translation_acc", "cosine_similarity_test_loss", "mse_test_loss"]
    ]
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
    # .query("embed_ln_weights != model_weights")
    # .query("loss_function == 'cosine_similarity'")
    # .dropna()
    # .reset_index(drop=True)
)
# Plotting the differences in mark_translation_acc for cosine_similarity and mse
df_cosine = df.query("loss_function == 'cosine_similarity'")
df_mse = df.query("loss_function == 'mse_loss'")

fig_cosine = px.scatter(
    df_cosine,
    x="cosine_similarity_test_loss",
    y="mark_translation_acc",
    trendline="ols",
)
fig_mse = px.scatter(
    df_mse,
    x="mse_test_loss",
    y="mark_translation_acc",
    trendline="ols",
)

if should_show_plot(plot_index):
    fig_cosine.show(config={"responsive": True})
plot_index += 1
print(
    """
for cosine similarity, more negative values are more performant. as such, we would want
to see a negative trendline. translation has a very negative trendline but some of them
seem to have a fairly positive one like linear map.
"""
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
print(
    """
for when we are using mse as our loss function, if there is a good correspondence
between mark translation loss and mse loss, we would want to see a negative gradient
(as you have a bigger mse, you perform worse on mark translation. however, this does
not seem to be the case). as such, i think we should use cosine similarity the thing we
optimise. actually, we should be calculating the overall trendline, not splitting by
transformation!!!
"""
)

# %%
"""
actually, what's the effect of layernorm on this? let's only do this for layernorm =
model_weights
"""
df = (
    runs_df[
        configs_that_change_names
        + ["mark_translation_acc", "cosine_similarity_test_loss", "mse_test_loss"]
    ]
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
    # .query("embed_ln_weights != model_weights")
    # .query("loss_function == 'cosine_similarity'")
    # .dropna()
    # .reset_index(drop=True)
)
# Plotting the differences in mark_translation_acc for cosine_similarity and mse
df_cosine = df.query(
    "loss_function == 'cosine_similarity' and embed_ln_weights == 'model_weights'"
)
df_mse = df.query("loss_function == 'mse_loss' and embed_ln_weights == 'model_weights'")

fig_cosine = px.scatter(
    df_cosine,
    x="cosine_similarity_test_loss",
    y="mark_translation_acc",
    trendline="ols",
)
fig_mse = px.scatter(
    df_mse,
    x="mse_test_loss",
    y="mark_translation_acc",
    trendline="ols",
)
if should_show_plot(plot_index):
    fig_cosine.show(config={"responsive": True})
plot_index += 1
if should_show_plot(plot_index):
    fig_mse.show(config={"responsive": True})
plot_index += 1
print(
    """
    it seems like once we set embed_ln_weights = model_weights and unembed_ln_weights,
    the difference between the two vanishes. i think this is because embed_ln_weights
    is super important when we use mse because otherwise the magnitudes of the
    embeddings are so large that the loss is so high that it doesn't matter how
    close the embeddings are to each other. if this is the case then as long as we have
    embed_ln_weights = model_weights, we should be good to go. let's see if this is the
    case by doing a parcat plot for the correlation coefficients.
"""
)

# %%
df = runs_df[
    configs_that_change_names
    + ["mark_translation_acc", "cosine_similarity_test_loss", "mse_test_loss"]
].query("loss_function == 'mse_loss'")
print(df)
# Calculate correlation coefficients for each combination of embed_ln_weights and
# unembed_ln_weights
unique_combinations = df[["embed_ln_weights", "unembed_ln_weights"]].drop_duplicates()
correlation_results = []
for index, row in unique_combinations.iterrows():
    filtered_df = df.query(
        f"embed_ln_weights == '{row['embed_ln_weights']}' and "
        f"unembed_ln_weights == '{row['unembed_ln_weights']}'"
    )
    correlation = (
        filtered_df[["mark_translation_acc", "mse_test_loss"]].corr().iloc[0, 1]
    )
    correlation_results.append(
        {
            "embed_ln_weights": row["embed_ln_weights"],
            "unembed_ln_weights": row["unembed_ln_weights"],
            "correlation_coefficient": correlation,
        }
    )
correlation_df = pd.DataFrame(correlation_results)
fig_correlation = create_parallel_categories_plot(
    correlation_df,
    dimensions=["embed_ln_weights", "unembed_ln_weights"],
    color="correlation_coefficient",
    title="Parallel Categories Plot for Correlation Coefficients",
    annotation_text="Correlation between mark_translation_acc and mse_test_loss",
    invert_colors=False,
)
if should_show_plot(plot_index):
    fig_correlation.show()
plot_index += 1
print(
    """
    yep, as i suspected! using model_weights for embed_ln_weights gives us the best
    correlation between mse_test_loss and mark_translation_acc.
"""
)

# %%
print(
    """
    wait, is it the case that if you layernorm, there should be no difference between
    linear map and rotation?
    """
)
df = (
    runs_df[configs_that_change_names + ["mark_translation_acc"]]
    .query("embed_ln_weights != 'no_ln'")
    .query("unembed_ln_weights != 'no_ln'")
    .query("transformation in ['linear_map', 'rotation']")
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
    annotation_text="filters: None",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1
# %%
print(
    """
    cosine similarity test loss may be the more informative thing to look at
    """
)
df = (
    runs_df[configs_that_change_names + ["cosine_similarity_test_loss"]]
    .query("embed_ln_weights != 'no_ln'")
    .query("unembed_ln_weights != 'no_ln'")
    .query("transformation in ['linear_map', 'rotation']")
    .query("loss_function == 'cosine_similarity'")
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
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
    annotation_text="filters: None",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1


# %%
print(
    """
    how about for just default weights?
    """
)
df = (
    runs_df[configs_that_change_names + ["cosine_similarity_test_loss"]]
    .query("embed_ln_weights == 'default_weights'")
    .query("unembed_ln_weights == 'default_weights'")
    .query("transformation in ['linear_map', 'rotation']")
    .query("loss_function == 'cosine_similarity'")
    # .query("dataset == 'cc_cedict_zh_en_extracted'")
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
    annotation_text="filters: None",
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1


# %%
print(
    """
    and for mark_translation_acc?
    """
)
df = (
    runs_df[configs_that_change_names + ["mark_translation_acc"]]
    .query("embed_ln_weights == 'default_weights'")
    .query("unembed_ln_weights == 'default_weights'")
    .query("transformation in ['linear_map', 'rotation']")
    .query("loss_function == 'cosine_similarity'")
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
    annotation_text="filters: None",
    invert_colors=True,
)
if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
"""
revisiting our test_loss and mark_translation_acc scatter plots, i just want to check
it is the case that there are no runs where we get better mse or cosine sim test loss
but lower mark_translation_acc (i think this should be the case especially when
embed_ln_weights = model_weights and unembed_ln_weights = model_weights).
"""
df = (
    runs_df[
        configs_that_change_names
        + ["mark_translation_acc", "cosine_similarity_test_loss", "mse_test_loss"]
    ].query("transformation not in ['identity']")
    # .query("transformation == 'linear_map'")
    # .query("embed_ln_weights == 'model_weights'")
    # .query("unembed_ln_weights == 'model_weights'")
)
# Plotting the differences in mark_translation_acc for cosine_similarity and mse
df_cosine = df.query("loss_function == 'cosine_similarity'")
df_mse = df.query("loss_function == 'mse_loss'")

fig_cosine = px.scatter(
    df_cosine,
    x="cosine_similarity_test_loss",
    y="mark_translation_acc",
    color="transformation",
    trendline="ols",
)
if should_show_plot(plot_index):
    fig_cosine.show(config={"responsive": True})
plot_index += 1

fig_mse = px.scatter(
    df_mse,
    x="mse_test_loss",
    y="mark_translation_acc",
    color="transformation",
    trendline="ols",
)
if should_show_plot(plot_index):
    fig_mse.show(config={"responsive": True})
plot_index += 1

print(
    """
"""
)
print(
    """
"""
)
