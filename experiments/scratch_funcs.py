# %%

from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from auto_embeds.utils.misc import calculate_gradient_color
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
    df: pd.DataFrame,
    configs_that_change_names: List[str],
    comparison_name: str,
    comparison_values: Tuple[str, str],
    metric: str,
    annotation_text: str = "",
    invert_colors: bool = False,
) -> go.Figure:
    """
    Generates a plot showing the differences in a specified metric between two
    configurations.

    Args:
        df: DataFrame containing the data.
        configs_that_change_names: List of configuration names that change.
        comparison_name: The name of the configuration parameter to compare.
        comparison_values: A tuple containing the values of the configurations to compare.
        metric: The metric for which the differences are plotted.
        annotation_text: Text to annotate the plot with. Defaults to an empty string.
        invert_colors: A boolean flag to invert colors in the plot.

    Returns:
        A plotly.graph_objects.Figure object representing the difference plot.
    """
    difference_df = get_difference_df(
        df=df,
        configs_that_change_names=configs_that_change_names,
        comparison_name=comparison_name,
        comparison_values=comparison_values,
        metric=metric,
    )
    first_value, second_value = comparison_values
    num_negative_differences = (difference_df["difference"] < 0).sum()
    num_positive_differences = (difference_df["difference"] > 0).sum()
    status_icon = (
        "✅" if num_negative_differences == 0 or num_positive_differences == 0 else "❌"
    )
    rprint(
        f"{status_icon} There are {num_negative_differences} negative differences and "
        f"{num_positive_differences} positive differences in {metric} "
        f"between {first_value} and {second_value}."
    )
    fig = create_parallel_categories_plot(
        difference_df,
        dimensions=[
            name for name in configs_that_change_names if name != comparison_name
        ],
        color="difference",
        title=f"difference in {metric} between {first_value} and {second_value}",
        annotation_text=annotation_text,
        difference=True,
        invert_colors=invert_colors,
    )
    return fig


def display_top_runs_table(
    df,
    metrics,
    changed_configs_column_names,
    sort_by,
    top_n=None,
    record_console=True,
    metric_interpretation=None,
    ascending=False,
):
    """
    Displays a table of the top runs based on specified metrics.

    Args:
        df: DataFrame containing the data.
        metrics: List of metrics to display in the table.
        changed_configs_column_names: List of configuration names that change.
        sort_by: Metric name to sort the table by.
        top_n: Number of top runs to display. If None, displays all runs.
        record_console: Whether to record the console output.
        metric_interpretation: Optional dictionary mapping metrics to 'higher_better'
                               or 'lower_better' for gradient colouring. Defaults to
                               None.
        ascending: Boolean to control the sorting order. Defaults to False.

    Returns:
        None, prints a table to the console.
    """
    # Select and sort the top runs
    columns_to_display = changed_configs_column_names + metrics
    if top_n is None:
        top_runs_df = df.sort_values(by=sort_by, ascending=ascending)[
            columns_to_display
        ]
    else:
        top_runs_df = (
            df.nlargest(top_n, sort_by) if ascending else df.nsmallest(top_n, sort_by)
        )
    top_runs_df.reset_index(inplace=True, drop=True)

    # Setup console for output
    console = Console(record=record_console, width=900)
    table = Table(show_header=True, header_style="bold")
    for column in top_runs_df.columns:
        if column in metrics:
            table.add_column(column, justify="center", width=8, max_width=12)
        else:
            table.add_column(column, justify="left", width=18, max_width=24)

    # Prepare data for display with color coding
    min_values = top_runs_df.min(numeric_only=True)
    max_values = top_runs_df.max(numeric_only=True)
    global_color_mapping = create_global_color_mapping(top_runs_df)

    # Ensure metric_interpretation is a dictionary if not provided
    if metric_interpretation is None:
        metric_interpretation = {}

    for _, row in top_runs_df.iterrows():
        table.add_row(
            *format_row(
                row,
                top_runs_df.columns,
                min_values,
                max_values,
                global_color_mapping,
                metric_interpretation,
            )
        )

    console.print(table)
    console.print(f"\nTotal runs displayed: {len(top_runs_df)}")


def create_global_color_mapping(df):
    """Create a color mapping for object-type columns in the dataframe."""
    all_unique_values = set(
        val
        for column in df.select_dtypes(include="object").columns
        for val in df[column].unique()
    )
    colors = qualitative.Plotly * (
        (len(all_unique_values) // len(qualitative.Plotly)) + 1
    )
    return dict(zip(all_unique_values, colors))


def format_row(
    row, columns, min_values, max_values, color_mapping, metric_interpretation
):
    """Format each row of the dataframe for display in the table."""
    row_data = []
    for index, item in enumerate(row):
        column_name = columns[index]
        if pd.isna(item):
            row_data.append("[grey]-[/grey]")
        elif isinstance(item, bool):
            color = "green" if item else "red"
            row_data.append(f"[{color}]{item}[/{color}]")
        elif isinstance(item, (int, float)):
            interpretation = metric_interpretation.get(column_name, "higher_better")
            reverse = interpretation == "lower_better"
            color = calculate_gradient_color(
                item, min_values[column_name], max_values[column_name], reverse
            )
            formatted_item = (
                f"[{color}]{format(item, '.3f')}[/{color}]"
                if column_name != "index"
                else f"[{color}]{item}[/]"
            )
            row_data.append(formatted_item)
        else:
            color = color_mapping.get(item, "grey")
            row_data.append(f"[{color}]{item}[/{color}]")
    return row_data
