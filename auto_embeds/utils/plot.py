import logging

import plotly.express as px

from auto_embeds.utils.misc import dynamic_text_wrap


def create_parallel_categories_plot(
    df,
    dimensions,
    color,
    title,
    annotation_text,
    groupby_conditions=None,
    query=None,
    labels=None,
    difference=None,
    invert_colors=False,  # New argument to invert the color scale
):
    """
    Creates and displays a parallel categories plot based on the provided parameters,
    with options to filter the DataFrame using a query string and to group the DataFrame
    by specified conditions. Rows where the color column is NA are filtered out.
    If 'difference' is True, the color scale will explicitly mark 0 and highlight
    negative values. If 'invert_colors' is True, the color scale will be inverted.

    Args:
        df (pd.DataFrame): The DataFrame to plot.
        dimensions (list): The dimensions to use in the plot.
        color (str): The column name to color the lines by.
        title (str): The title of the plot.
        annotation_text (str): Text for the annotation to add to the plot.
        groupby_conditions (list, optional): Conditions to group the DataFrame by.
        query (str, optional): A query string to filter the DataFrame before plotting.
        labels (dict, optional): A mapping of column names to display labels.
            Defaults to a predefined dictionary.
        difference (bool, optional): If True, use a diverging colour scale and set scale
            midpoint to 0 to highlight differences between postive and negative values.
        invert_colors (bool, optional): If True, invert the color scale.
    """

    # Apply query if provided
    if query:
        df = df.query(query)

    # Filter out rows where the color column is NA and log the action
    filtered_df = df.dropna(subset=[color])
    num_filtered = len(df) - len(filtered_df)
    if num_filtered > 0:
        logging.info(f"Filtered out {num_filtered} rows with NA in '{color}' column.")

    df = filtered_df

    # Use the DataFrame directly for plotting, applying groupby conditions if provided
    if groupby_conditions:
        df = df.groupby(groupby_conditions)[color].mean().reset_index()

    color_scale = px.colors.diverging.Tealrose
    if invert_colors:
        color_scale = color_scale[::-1]  # Invert the color scale

    fig = (
        px.parallel_categories(
            df,
            dimensions=dimensions,
            color=color,
            labels=labels,
            title=title,
            color_continuous_scale=color_scale if difference else None,
            color_continuous_midpoint=0 if difference else None,
        )
        .update_traces(arrangement="freeform")
        .add_annotation(
            text=dynamic_text_wrap(annotation_text, 600),
            align="left",
            xref="paper",
            yref="paper",
            showarrow=False,
            x=0,
            y=-0.25,
            font=dict(size=13),
        )
        .update_layout(autosize=True)
    )

    fig.show(config={"responsive": True})
