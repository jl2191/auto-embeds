import dash_bootstrap_components as dbc
from dash import dcc, html

from experiments.dash.data_and_figures import (
    figure_1_train_test_loss,
    figure_2_cos_sims_trend,
)

layout = dbc.Container(
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
                                        "embed_weights",
                                        "embed_ln",
                                        "embed_ln_weights",
                                        "unembed_weights",
                                        "unembed_ln",
                                        "unembed_ln_weights",
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
        # Filter and Badge for displaying the number of runs
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.Label(
                            "Filter Query:",
                            htmlFor="text-filter",
                            className="text-center",
                            style={"marginBottom": "10px"},
                        ),
                        dbc.InputGroup(
                            [
                                dcc.Input(
                                    id="text-filter",
                                    type="text",
                                    debounce=True,
                                    style={"flex": "1"},
                                ),
                                dbc.Badge(
                                    "0 Runs",  # Default text to ensure visibility
                                    id="current-runs-badge",
                                    color="info",
                                    className="ml-1",
                                ),
                            ],
                            style={"width": "100%"},
                        ),
                    ],
                    style={"width": "50%", "margin": "0 auto"},
                ),
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
