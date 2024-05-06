import json

from dash import Input, Output, State, callback_context, no_update

from experiments.dash.data_and_figures import (
    exploded_df_fig_1,
    generate_cos_sims_trend_figure,
    generate_train_loss_figure,
    runs_df,
)


def figure_1_container_clicked(n_clicks, clickData, trace_is_clicked):
    # if a trace was previously clicked, then then the user clicks anywhere in the
    # container again, we reset clickData.
    if trace_is_clicked:
        return None, None
    return no_update


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
    modified_figure_1 = generate_train_loss_figure(color_var, filtered_df)

    if "color-dropdown" in triggered_id:
        return modified_figure_1, None, None
    elif "text-filter" in triggered_id:
        return modified_figure_1, None, None
    elif "figure-1-train-test-loss" in triggered_id:
        if clickData:
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
            return modified_figure_1, no_update, None


def update_cos_sims_trend_plot(hoverData, clicked_trace_data, hovered_trace_data):
    if clicked_trace_data:
        return generate_cos_sims_trend_figure(clicked_trace_data)
    elif hovered_trace_data:
        return generate_cos_sims_trend_figure(hovered_trace_data)
    else:
        default_data = json.dumps(
            {
                "plotly_json": runs_df["cos_sims_trend_plot"].iloc[0],
                "run_name": runs_df["run_name"].iloc[0],
            }
        )
        return generate_cos_sims_trend_figure(default_data)


def update_run_count(text_filter):
    filtered_df = (
        exploded_df_fig_1.query(text_filter) if text_filter else exploded_df_fig_1
    )
    count = filtered_df["run_name"].nunique()
    return f"Showing {count} Runs"


def register_callbacks(app):
    app.callback(
        [
            Output("figure-1-train-test-loss", "clickData"),
            Output(
                "train-test-loss-plot-trace-clicked", "children", allow_duplicate=True
            ),
        ],
        [Input("figure-1-train-test-loss-container", "n_clicks")],
        [
            State("figure-1-train-test-loss", "clickData"),
            State("train-test-loss-plot-trace-clicked", "children"),
        ],
        prevent_initial_call=True,
    )(figure_1_container_clicked)

    app.callback(
        [
            Output("figure-1-train-test-loss", "figure"),
            Output(
                "train-test-loss-plot-trace-clicked", "children", allow_duplicate=True
            ),
            Output("train-test-loss-plot-trace-hovered", "children"),
        ],
        [
            Input("color-dropdown", "value"),
            Input("text-filter", "value"),
            Input("figure-1-train-test-loss", "hoverData"),
            Input("figure-1-train-test-loss", "clickData"),
        ],
        [State("train-test-loss-plot-trace-clicked", "children")],
        prevent_initial_call=True,
    )(update_train_test_loss_plot)

    app.callback(
        Output("figure-2-cos-sims-trend", "figure"),
        [
            Input("figure-1-train-test-loss", "hoverData"),
            Input("train-test-loss-plot-trace-clicked", "children"),
            Input("train-test-loss-plot-trace-hovered", "children"),
        ],
        prevent_initial_call=True,
    )(update_cos_sims_trend_plot)

    app.callback(
        Output("current-runs-badge", "children"), Input("text-filter", "value")
    )(update_run_count)
