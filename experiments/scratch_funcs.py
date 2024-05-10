# %%

from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from auto_embeds.utils.plot import create_parallel_categories_plot


def plot_matrix_visualizations(results_df):
    """Plot the first three matrices from the results dataframe as heatmaps."""
    # Number of matrices to plot
    num_matrices = len(results_df["matrix"])

    # Create subplots
    fig = make_subplots(rows=num_matrices, cols=1)

    # Add each matrix as a heatmap to the subplots
    for i, matrix in enumerate(results_df["matrix"], start=1):
        fig.add_trace(
            go.Heatmap(z=matrix, colorscale="Viridis", showscale=True), row=i, col=1
        )

    # Update layout to make each plot square
    fig.update_layout(
        height=1024 * num_matrices, width=1024, title_text="Matrix Visualizations"
    )

    # Return the figure
    return fig


def get_difference_df(
    df: pd.DataFrame,
    configs_that_change_names: List[str],
    comparison_name: str,
    comparison_values: Tuple[str, str],
    metric: str,
) -> pd.DataFrame:
    """
    Computes the difference between two configurations for a specified metric.

    Args:
        df: DataFrame containing the data.
        configs_that_change_names: List of configuration names that change.
        comparison_config: A tuple containing the query strings for the two configs to
                           compare. Each element should be a valid DataFrame query
                           string that uniquely identifies each configuration subset.
        metric: The metric for which the difference is to be calculated.

    Returns:
        A DataFrame with the differences computed between the specified configurations.
    """
    first_value, second_value = comparison_values
    # Filter out columns with unhashable data types (e.g., lists, dicts)
    hashable_columns = list(configs_that_change_names) + [metric]
    # Create subsets for each configuration in the comparison
    df_first_config = (
        df[hashable_columns]
        .query(f"{comparison_name} == @first_value")
        .reset_index(drop=True)
    )
    df_second_config = (
        df[hashable_columns]
        .query(f"{comparison_name} == @second_value")
        .reset_index(drop=True)
    )

    # Merge on all other parameters
    merged_df = pd.merge(
        df_first_config,
        df_second_config,
        on=[
            col for col in hashable_columns if col != metric and col != comparison_name
        ],
        suffixes=[f"_{first_value}", f"_{second_value}"],
    )

    # Calculate the difference
    merged_df["difference"] = (
        merged_df[f"{metric}_{second_value}"] - merged_df[f"{metric}_{first_value}"]
    )

    return merged_df


# %%
def plot_difference(
    df,
    query,
    configs_that_change_names,
    comparison_name,
    comparison_values,
    metric,
    annotation_text,
    invert_colors=False,
):
    # Only perform the query if it is provided, otherwise use the full dataframe
    if query:
        filtered_df = df.query(query)
    else:
        filtered_df = df

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
        invert_colors=invert_colors,
    )
    return fig
