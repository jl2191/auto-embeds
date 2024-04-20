# Import necessary libraries
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback_context, dcc, html
from plotly.subplots import make_subplots

from auto_embeds.utils.misc import (
    fetch_wandb_runs,
    process_wandb_runs_df,
)

# Load and preprocess data outside of callbacks
original_df = fetch_wandb_runs(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-04-17 random and singular plural", "run group 2"],
)
runs_df = process_wandb_runs_df(
    original_df,
    custom_labels={
        "transformation": {
            "rotation": "Rotation",
            "linear_map": "Linear Map",
            "translation": "Translation",
        },
        "dataset": {
            "wikdict_en_fr_extracted": "wikdict_en_fr",
            "cc_cedict_zh_en_extracted": "cc_cedict_zh_en",
            "random_word_pairs": "Random Word Pairs",
            "singular_plural_pairs": "Singular Plural Pairs",
        },
    },
)

# Explode and melt the DataFrame outside the callback
preprocessed_df = runs_df.explode(column=["epoch", "test_loss", "train_loss"]).melt(
    id_vars=[
        "name",
        "dataset",
        "seed",
        "transformation",
        "embed_apply_ln",
        "unembed_apply_ln",
        "epoch",
    ],
    value_vars=["train_loss", "test_loss"],
    var_name="loss_type",
    value_name="loss",
)

# draw the graphs


def generate_figure_1(color_var, df, highlighted_name=None):
    """
    Generates the figure based on the color variable.
    Highlights the plot corresponding to the highlighted_name if provided.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Test Loss"))
    train_df = df.query("`loss_type` == 'train_loss'")
    test_df = df.query("`loss_type` == 'test_loss'")
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
                                f'Embed Apply LN: {row["embed_apply_ln"]}',
                                f'Unembed Apply LN: {row["unembed_apply_ln"]}',
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

    max_epoch = preprocessed_df["epoch"].max()
    fig.update_layout(
        height=600,
        title_text="Loss Over Epochs by Selected Variable",
        xaxis=dict(range=[0, max_epoch + 1]),
        yaxis=dict(range=[-1, 0.2]),
    )

    return fig


figure_1 = generate_figure_1("dataset", preprocessed_df)

# set the layout

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Language Transformation Results", className="text-center"),
                width=12,
                style={"marginBottom": "20px", "marginTop": "20px"},
            )
        ),
        # Figure 1
        dbc.Row(
            [
                dbc.Col(
                    html.H2("Figure 1: Loss Over Epochs by Selected Variable"), width=12
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=figure_1,
                        id="loss-plot",
                        clear_on_unhover=True,
                    ),
                    width=12,
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Label(
                                "Group By:",
                                htmlFor="color-dropdown",
                                className="text-center",
                            ),
                            dcc.Dropdown(
                                id="color-dropdown",
                                options=[
                                    {"label": col, "value": col}
                                    for col in [
                                        "name",
                                        "dataset",
                                        "seed",
                                        "transformation",
                                        "embed_apply_ln",
                                        "unembed_apply_ln",
                                    ]
                                ],
                                value="dataset",
                            ),
                        ],
                        style={
                            "width": "50%",
                            "margin": "0 auto",
                            "marginBottom": "20px",
                        },
                    ),
                ],
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Label(
                                "Filter Query:",
                                htmlFor="text-filter",
                                className="text-center",
                                style={"marginBottom": "10px"},
                            ),
                            dcc.Input(
                                id="text-filter",
                                type="text",
                                debounce=True,
                                style={"height": "40px", "width": "100%"},
                            ),
                        ],
                        style={"width": "50%", "margin": "0 auto"},
                    ),
                    dbc.Tooltip(
                        "Enter a pandas dataframe query string to filter the results. "
                        "Example \"transformation in ['Rotation', 'Linear Map'] and "
                        'seed = 10" or "unembed_apply_ln == True".',
                        target="text-filter",
                    ),
                ],
                width=12,
                style={"marginBottom": "20px"},
            )
        ),
    ],
    fluid=True,
    className="py-3",
)

# set the callbacks


@app.callback(
    Output("loss-plot", "figure"),
    [
        Input("color-dropdown", "value"),
        Input("loss-plot", "hoverData"),
        Input("text-filter", "value"),
    ],
    prevent_initial_call=True,
)
def update_plots(color_var, hoverData, text_filter):
    """Update loss plot based on color, hover event, or text filter."""
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Filter dataframe based on text input
    filtered_df = preprocessed_df.query(text_filter) if text_filter else preprocessed_df

    if triggered_id == "color-dropdown" or triggered_id == "text_filter":
        # Generate figure based on the color variable
        modified_figure_1 = generate_figure_1(color_var, filtered_df)
        return modified_figure_1
    else:
        if hoverData:
            hovered_run_name = hoverData["points"][0]["customdata"]
            print(hovered_run_name)
            # Highlight the corresponding plot in both subplots
            modified_figure_1 = generate_figure_1(
                color_var, filtered_df, highlighted_name=hovered_run_name
            )
            return modified_figure_1
        else:
            return generate_figure_1(color_var, filtered_df)


if __name__ == "__main__":
    app.run_server(debug=True)
