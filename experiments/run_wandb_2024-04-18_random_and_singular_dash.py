# Import necessary libraries
import json

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
from plotly.subplots import make_subplots

from auto_embeds.utils.misc import fetch_wandb_runs, process_wandb_runs_df

# Load and preprocess data outside of callbacks
original_df = fetch_wandb_runs(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-04-17 random and singular plural", "run group 2"],
    get_artifacts=True,
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
exploded_df_fig_1 = runs_df.explode(column=["epoch", "test_loss", "train_loss"]).melt(
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

filtered_df_fig_1 = exploded_df_fig_1

# draw the graphs


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
        align="left",  # Align text to the left
    )
    return fig


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
                    html.Div(
                        id="figure-1-train-test-loss-container",
                        children=[
                            dcc.Graph(
                                figure=figure_1_train_test_loss,
                                id="figure-1-train-test-loss",
                                clear_on_unhover=True,
                            )
                        ],
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
        # Figure 2 - Cos Sims Trend Plot
        dbc.Row(
            [
                dbc.Col(
                    html.H2("Figure 2: Cosine Similarity Trend Plot"),
                    width=12,
                    style={"marginTop": "20px"},
                ),
                dbc.Col(
                    dcc.Graph(
                        figure=figure_2_cos_sims_trend,
                        id="figure-2-cos-sims-trend",
                        clear_on_unhover=True,
                    ),
                    width=12,
                    style={"marginBottom": "20px"},
                ),
            ]
        ),
        # Hidden div for storing the state of whether the train test loss plot trace was
        # clicked
        html.Div(id="train-test-loss-plot-trace-clicked", style={"display": "none"}),
        html.Div(id="train-test-loss-plot-trace-hovered", style={"display": "none"}),
    ],
    fluid=True,
    className="py-3",
)

# set the callbacks


@app.callback(
    Output("figure-1-train-test-loss", "clickData"),
    Output("train-test-loss-plot-trace-clicked", "children", allow_duplicate=True),
    [Input("figure-1-train-test-loss-container", "n_clicks")],
    [
        State("figure-1-train-test-loss", "clickData"),
        State("train-test-loss-plot-trace-clicked", "children"),
    ],
    prevent_initial_call=True,
)
def figure_1_container_clicked(n_clicks, clickData, trace_is_clicked):
    # if a trace was previously clicked, then then the user clicks anywhere in the
    # container again, we reset clickData.
    if trace_is_clicked:
        return None, None
    return no_update


@app.callback(
    [
        Output("figure-1-train-test-loss", "figure"),
        Output("train-test-loss-plot-trace-clicked", "children", allow_duplicate=True),
        Output("train-test-loss-plot-trace-hovered", "children"),
    ],
    [
        Input("color-dropdown", "value"),
        Input("text-filter", "value"),
        Input("figure-1-train-test-loss", "hoverData"),
        Input("figure-1-train-test-loss", "clickData"),
    ],
    [
        State("train-test-loss-plot-trace-clicked", "children"),
    ],
    prevent_initial_call=True,
)
def update_train_test_loss_plot(
    color_var, text_filter, hoverData, clickData, clicked_trace_data
):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # we show the highlighted line on both charts on hover, as well as display the
    # corresponding cos sims trend plot.

    # if we click on a trace, we "lock" in, meaning hovers will now have any effects.
    # this allows people to then move the mouse to check out the cos sims trend plot.

    filtered_df = (
        exploded_df_fig_1.query(text_filter) if text_filter else exploded_df_fig_1
    )
    figure_1_train_test_loss_refreshed = generate_train_loss_figure(
        color_var, filtered_df
    )

    if "color-dropdown" in triggered_id:
        modified_figure_1 = generate_train_loss_figure(color_var, filtered_df)
        figure_1_train_test_loss_refreshed = modified_figure_1
        return modified_figure_1, None, None
    elif "text-filter" in triggered_id:
        modified_figure_1 = generate_train_loss_figure(color_var, filtered_df)
        figure_1_train_test_loss_refreshed = modified_figure_1
        return modified_figure_1, None, None
    elif "figure-1-train-test-loss" in triggered_id:
        if clickData:
            # check if click is on a data point.
            if clickData["points"]:
                clicked_run_name = clickData["points"][0]["customdata"]
                clicked_trace_data = json.dumps(
                    {
                        "plotly_json": runs_df.query("name == @clicked_run_name")[
                            "cos_sims_trend_plot"
                        ].iloc[0],
                        "run_name": clicked_run_name,
                    }
                )
                modified_figure_1 = generate_train_loss_figure(
                    color_var, filtered_df, highlighted_name=clicked_run_name
                )
                return (
                    modified_figure_1,
                    clicked_trace_data,
                    no_update,
                )
            else:
                raise ValueError("There is click data but no points.")
        elif hoverData:
            hovered_run_name = hoverData["points"][0]["customdata"]
            modified_figure_1 = generate_train_loss_figure(
                color_var, filtered_df, highlighted_name=hovered_run_name
            )
            hovered_trace_data = json.dumps(
                {
                    "plotly_json": runs_df.query("name == @hovered_run_name")[
                        "cos_sims_trend_plot"
                    ].iloc[0],
                    "run_name": hovered_run_name,
                }
            )

            return (
                modified_figure_1,
                None,
                hovered_trace_data,
            )
        else:
            # as we have clear_on_unhover, this means that if the user has not clicked
            # on particular trace to lock the view, we return the plot to its original
            # state (without highlights).
            return figure_1_train_test_loss_refreshed, no_update, None


@app.callback(
    Output("figure-2-cos-sims-trend", "figure"),
    [
        Input("figure-1-train-test-loss", "hoverData"),
        Input("train-test-loss-plot-trace-clicked", "children"),
        Input("train-test-loss-plot-trace-hovered", "children"),
    ],
    prevent_initial_call=True,
)
def update_cos_sims_trend_plot(hoverData, clicked_trace_data, hovered_trace_data):

    if clicked_trace_data:
        return generate_cos_sims_trend_figure(clicked_trace_data)
    elif hovered_trace_data:
        return generate_cos_sims_trend_figure(hovered_trace_data)
    else:
        default_data = json.dumps(
            {
                "plotly_json": runs_df["cos_sims_trend_plot"].iloc[0],
                "run_name": runs_df["name"].iloc[0],
            }
        )
        return generate_cos_sims_trend_figure(default_data)


if __name__ == "__main__":
    app.run_server(debug=True)
