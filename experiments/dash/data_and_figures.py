# %%
import json

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from auto_embeds.utils.wandb import fetch_wandb_runs, process_wandb_runs_df


def generate_train_loss_figure(color_var, df, highlighted_name=None):
    """
    Generates the figure based on the color variable.
    Highlights the plot corresponding to the highlighted_name if provided.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Train Loss", "Test Loss"),
        shared_xaxes=True,
        shared_yaxes=True,
    )
    train_df = df.query("loss_type == 'train_loss'")
    test_df = df.query("loss_type == 'test_loss'")
    unique_vals = df[color_var].unique()
    colors = px.colors.qualitative.Plotly

    for i, val in enumerate(unique_vals):

        def add_loss_trace(
            fig, df, val, colors, i, loss_type, row, col, opacity, width, showlegend
        ):
            """
            Adds a trace for loss to the figure with specified opacity and line width.

            Args:
                fig: The figure object to add the trace to.
                df: The dataframe containing the data for the trace.
                val: The value used for the name and legend group.
                colors: The list of colors to use for the traces.
                i: The index used to select the color from the colors list.
                loss_type: Specifies if the trace is for 'train_loss' or 'test_loss'.
                row: The row position of the subplot in the figure.
                col: The column position of the subplot in the figure.
                opacity: The opacity level for the trace.
                width: The line width for the trace.
                showlegend: Whether to show the legend for the trace.
            """
            fig.add_trace(
                go.Scatter(
                    x=df["epoch"],
                    y=df["loss"],
                    mode="lines",
                    name=f"{val}",
                    legendgroup=f"{val}",
                    showlegend=showlegend,
                    line=dict(color=colors[i % len(colors)], width=width),
                    opacity=opacity,
                    hoverinfo="text",
                    text=df.apply(
                        lambda row: "<br>".join(
                            [
                                f'Name: {row["name"]}',
                                f'Dataset: {row["dataset"]}',
                                f'Seed: {row["seed"]}',
                                f'Transformation: {row["transformation"]}',
                                f'Embed Weight: {row["embed_weight"]}',
                                f'Embed LN: {row["embed_ln"]}',
                                f'Embed LN Weights: {row["embed_ln_weights"]}',
                                f'Unembed Weight: {row["unembed_weight"]}',
                                f'Unembed LN: {row["unembed_ln"]}',
                                f'Unembed LN Weights: {row["unembed_ln_weights"]}',
                                f'Epoch: {row["epoch"]}',
                                f'Loss: {row["loss"]}',
                            ]
                        ),
                        axis=1,
                    ),
                    customdata=df["name"],
                ),
                row=row,
                col=col,
            )

        if highlighted_name:
            add_loss_trace(
                fig=fig,
                df=train_df.query(f"{color_var} == @val and name != @highlighted_name"),
                val=val,
                colors=colors,
                i=i,
                loss_type="train_loss",
                row=1,
                col=1,
                opacity=0.4,
                width=2,
                showlegend=True,
            )
            add_loss_trace(
                fig=fig,
                df=test_df.query(f"{color_var} == @val and name != @highlighted_name"),
                val=val,
                colors=colors,
                i=i,
                loss_type="test_loss",
                row=1,
                col=2,
                opacity=0.4,
                width=2,
                showlegend=False,
            )
            add_loss_trace(
                fig=fig,
                df=train_df.query(f"{color_var} == @val and name == @highlighted_name"),
                val=val,
                colors=colors,
                i=i,
                loss_type="train_loss",
                row=1,
                col=1,
                opacity=1.0,
                width=4,
                showlegend=False,
            )
            add_loss_trace(
                fig=fig,
                df=test_df.query(f"{color_var} == @val and name == @highlighted_name"),
                val=val,
                colors=colors,
                i=i,
                loss_type="test_loss",
                row=1,
                col=2,
                opacity=1.0,
                width=4,
                showlegend=False,
            )
        else:
            df_train = train_df.query(f"{color_var} == @val")
            df_test = test_df.query(f"{color_var} == @val")
            add_loss_trace(
                fig, df_train, val, colors, i, "train_loss", 1, 1, 0.8, 2, True
            )
            add_loss_trace(
                fig, df_test, val, colors, i, "test_loss", 1, 2, 0.8, 2, False
            )

    max_epoch = exploded_df_fig_1["epoch"].max()
    fig.update_layout(
        title_text="Loss Over Epochs by Selected Variable",
        xaxis=dict(range=[0, max_epoch + 1]),
        yaxis=dict(range=[-1, 0.2]),
        margin=dict(l=50, r=300, t=100, b=50),
    )

    return fig


def generate_cos_sims_trend_figure(data):
    data = json.loads(data)
    fig = pio.from_json(data["plotly_json"])
    fig.add_annotation(
        text=f"Run Name: {data['run_name']}",
        xref="paper",
        yref="paper",
        x=0,
        y=1.26,
        showarrow=False,
        font=dict(size=12, color="grey"),
        align="left",
    )
    return fig


def generate_bar_chart_figure(df, color_var, x_var, y_var, title):
    fig = px.bar(df, x=x_var, y=y_var, color=color_var, title=title)
    return fig


def generate_parallel_categories_plot(
    df,
    dimensions,
    color,
    title,
    annotation_text,
    groupby_conditions=None,
    query=None,
    labels=None,
):
    """
    Creates and displays a parallel categories plot based on the provided parameters,
    with options to filter the DataFrame using a query string and to group the DataFrame
    by specified conditions. Rows where the color column is NA are filtered out.

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
    """

    # Apply query if provided
    if query:
        df = df.query(query)

    # Filter out rows where the color column is NA and log the action
    filtered_df = df.dropna(subset=[color])
    num_filtered = len(df) - len(filtered_df)
    if num_filtered > 0:
        print(f"Filtered out {num_filtered} rows with NA in '{color}' column.")

    df = filtered_df

    # Use the DataFrame directly for plotting, applying groupby conditions if provided
    if groupby_conditions:
        df = df.groupby(groupby_conditions)[color].mean().reset_index()

    fig = (
        px.parallel_categories(
            df,
            dimensions=dimensions,
            color=color,
            labels=labels,
            title=title,
        )
        .update_traces(arrangement="freeform")
        .add_annotation(
            text=annotation_text,
            align="left",
            xref="paper",
            yref="paper",
            showarrow=False,
            x=0,
            y=-0.25,
            font=dict(size=13),
        )
    )
    return fig


original_df = fetch_wandb_runs(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-04-29 analytical and ln", "experiment 3"],
    get_artifacts=True,
)

runs_df = process_wandb_runs_df(
    original_df,
    custom_labels={
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

exploded_df_fig_1 = (
    runs_df.drop([col for col in runs_df.columns if "mark_translation" in col], axis=1)
    .assign(
        epoch=lambda df: df["epoch"].apply(lambda x: x[:10]),
        test_loss=lambda df: df["test_loss"].apply(lambda x: x[:10]),
        train_loss=lambda df: df["train_loss"].apply(lambda x: x[:10]),
    )
    .explode(column=["epoch", "test_loss", "train_loss"])
    .melt(
        id_vars=[
            "name",
            "dataset",
            "seed",
            "transformation",
            "embed_weight",
            "embed_ln",
            "embed_ln_weights",
            "unembed_ln",
            "unembed_ln_weights",
            "unembed_weight",
            "epoch",
        ],
        value_vars=["train_loss", "test_loss"],
        var_name="loss_type",
        value_name="loss",
    )
)
figure_1_train_test_loss = generate_train_loss_figure("dataset", exploded_df_fig_1)
figure_1_train_test_loss_refreshed = generate_train_loss_figure(
    "dataset", exploded_df_fig_1
)
figure_2_default_data = {
    "plotly_json": runs_df["cos_sims_trend_plot"].iloc[0],
    "run_name": runs_df["name"].iloc[0],
}
figure_2_cos_sims_trend = generate_cos_sims_trend_figure(
    json.dumps(figure_2_default_data)
)
filtered_df_fig_1 = exploded_df_fig_1

figure_3_analytical_solutions_mark_translation_acc = generate_bar_chart_figure(
    runs_df,
    color_var="transformation",
    x_var="transformation",
    y_var="mark_translation_acc",
    title="Mark Translation Accuracy by Selected Variable",
)

figure_4_analytical_solutions_test_accuracy = generate_bar_chart_figure(
    runs_df,
    color_var="transformation",
    x_var="transformation",
    y_var="test_accuracy",
    title="Test Accuracy by Selected Variable",
)

figure_5_analytical_solutions_parcat_test_accuracy = generate_parallel_categories_plot(
    runs_df,
    dimensions=[
        "transformation",
        "seed",
        "embed_weight",
        "embed_ln",
        "embed_ln_weights",
        "unembed_weight",
        "unembed_ln",
        "unembed_ln_weights",
    ],
    color="test_accuracy",
    title="Parallel Categories Plot for Test Accuracy",
    annotation_text="Parallel Categories Plot for Test Accuracy",
)
