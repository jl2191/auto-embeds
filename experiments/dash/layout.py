import dash_bootstrap_components as dbc
from dash import dcc, html

from experiments.dash.data_and_figures import (
    figure_1_train_test_loss,
    figure_2_cos_sims_trend,
    figure_3_analytical_solutions_mark_translation_acc,
    figure_4_analytical_solutions_test_accuracy,
    figure_5_analytical_solutions_parcat_test_accuracy,
)


def create_title():
    return dbc.Row(
        dbc.Col(
            html.H1("Language Transformation Results", className="text-center"),
            width=12,
            style={"marginBottom": "20px", "marginTop": "20px"},
        )
    )


def create_figure_1():
    return dbc.Row(
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
    )


def create_group_by_dropdown():
    options = [
        "name",
        "dataset",
        "seed",
        "transformation",
        "embed_weight",
        "embed_ln",
        "embed_ln_weights",
        "unembed_weight",
        "unembed_ln",
        "unembed_ln_weights",
    ]
    return dbc.Row(
        dbc.Col(
            html.Div(
                [
                    html.Label(
                        "Group By:",
                        htmlFor="color-dropdown",
                        className="text-center",
                    ),
                    dcc.Dropdown(
                        id="color-dropdown",
                        options=[{"label": col, "value": col} for col in options],
                        value="dataset",
                    ),
                ],
                style={"width": "50%", "margin": "0 auto", "marginBottom": "20px"},
            ),
            width=12,
        )
    )


def create_filter_query():
    return dbc.Row(
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
                            dbc.Input(
                                id="text-filter",
                                type="text",
                                debounce=True,
                                style={"flex": "1", "height": "40px"},
                                placeholder="Enter filter query",
                            ),
                            dbc.Badge(
                                "0 Runs",
                                id="current-runs-badge",
                                color="light",
                                className="ml-1",
                                style={
                                    "alignItems": "center",
                                    "padding": "0 15px 2px 15px",
                                    "display": "flex",
                                    "fontSize": "12px",
                                },
                            ),
                        ],
                        style={"width": "100%"},
                    ),
                    dbc.Tooltip(
                        "Enter a pandas dataframe query string to filter the results. "
                        "Example \"transformation in ['Rotation', 'Linear Map'] and "
                        'seed = 10" or "unembed_apply_ln == True".',
                        target="text-filter",
                    ),
                ],
                style={"width": "50%", "margin": "0 auto"},
            ),
            width=12,
            style={"marginBottom": "20px"},
        )
    )


def create_figure_2():
    return dbc.Row(
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
    )


def create_hidden_divs():
    return html.Div(
        [
            html.Div(
                id="train-test-loss-plot-trace-clicked", style={"display": "none"}
            ),
            html.Div(
                id="train-test-loss-plot-trace-hovered", style={"display": "none"}
            ),
        ]
    )


def create_figure_3():
    return dbc.Row(
        [
            dbc.Col(
                html.H2("Figure 3: Mark Translation Accuracy by Selected Variable"),
                width=12,
            ),
            dbc.Col(
                html.Div(
                    id="figure-3-analytical-solutions-mark-translation-accuracy-container",
                    children=[
                        dcc.Graph(
                            figure=figure_3_analytical_solutions_mark_translation_acc,
                            id="figure-3-analytical-solutions-mark-translation-accuracy",
                            clear_on_unhover=True,
                        )
                    ],
                ),
                width=12,
            ),
        ]
    )


def create_figure_4():
    return dbc.Row(
        [
            dbc.Col(
                html.H2("Figure 4: Test Accuracy by Selected Variable"),
                width=12,
            ),
            dbc.Col(
                html.Div(
                    id="figure-4-analytical-solutions-test-accuracy-container",
                    children=[
                        dcc.Graph(
                            figure=figure_4_analytical_solutions_test_accuracy,
                            id="figure-4-analytical-solutions-test-accuracy",
                            clear_on_unhover=True,
                        )
                    ],
                ),
                width=12,
            ),
        ]
    )


def create_figure_5():
    return dbc.Row(
        [
            dbc.Col(
                html.H2("Figure 5: Parcat Test Accuracy by Selected Variable"),
                width=12,
            ),
            dbc.Col(
                html.Div(
                    id="figure-5-analytical-solutions-parcat-test-accuracy-container",
                    children=[
                        dcc.Graph(
                            figure=figure_5_analytical_solutions_parcat_test_accuracy,
                            id="figure-5-analytical-solutions-parcat-test-accuracy",
                            clear_on_unhover=True,
                        )
                    ],
                ),
                width=12,
            ),
        ]
    )


layout = dbc.Container(
    [
        create_title(),
        create_figure_1(),
        create_group_by_dropdown(),
        create_filter_query(),
        create_figure_2(),
        create_hidden_divs(),
        create_figure_3(),
        create_figure_4(),
        create_figure_5(),
    ],
    fluid=True,
    className="py-3",
)
