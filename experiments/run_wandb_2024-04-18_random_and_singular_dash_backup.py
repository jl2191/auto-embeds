# Import necessary libraries
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback_context, dcc, html
from plotly.subplots import make_subplots

from auto_embeds.utils.neptune import fetch_neptune_runs, process_neptune_runs_df

# Load and preprocess data outside of callbacks
original_df = fetch_neptune_runs(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-04-17 random and singular plural", "run group 2"],
)
runs_df = process_neptune_runs_df(
    original_df,
    custom_labels={
        "transformation": {"rotation": "Rotation", "linear_map": "Linear Map"},
        "dataset": {
            "wikdict_en_fr_extracted": "wikdict_en_fr",
            "cc_cedict_zh_en_extracted": "cc_cedict_zh_en",
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

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Language Transformation Results", className="text-center"),
                width=12,
                style={"marginTop": "20px"},
            )
        ),
        # Figure 1
        dbc.Row(
            [
                dbc.Col(
                    html.H2("Figure 1: Loss Over Epochs by Selected Variable"), width=12
                ),
                dbc.Col(dcc.Graph(id="loss-plot"), width=12),
                dbc.Col(
                    html.P(
                        "This plot allows users to visualize how the loss changes over epochs, with the ability to group or color the results by various variables."
                    ),
                    width=12,
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
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
                width={"size": 6, "offset": 3},
            )
        ),
        # Figure 2 Placeholder
        dbc.Row(
            [
                dbc.Col(html.H2("Figure 2: Placeholder Title"), width=12),
                dbc.Col(dcc.Graph(id="placeholder-plot"), width=12),
                dbc.Col(
                    html.P("Placeholder explanation for Figure 2."),
                    width=12,
                ),
            ]
        ),
        # Add more figures in a similar structure here
    ],
    fluid=True,
    className="py-3",
)


def generate_figure_1(color_var, highlighted_name=None):
    """
    Generates the figure based on the color variable.
    Highlights the plot corresponding to the highlighted_name if provided.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Test Loss"))
    train_df = preprocessed_df.query("`loss_type` == 'train_loss'")
    test_df = preprocessed_df.query("`loss_type` == 'test_loss'")
    unique_vals = preprocessed_df[color_var].unique()
    colors = px.colors.qualitative.Plotly

    for i, val in enumerate(unique_vals):
        # Plot all but highlighted_name
        df_filtered_train = train_df.query(
            f"{color_var} == @val and name != @highlighted_name"
        )
        df_filtered_test = test_df.query(
            f"{color_var} == @val and name != @highlighted_name"
        )

        # Add train loss trace for all but highlighted name
        fig.add_trace(
            go.Scatter(
                x=df_filtered_train["epoch"],
                y=df_filtered_train["loss"],
                mode="lines",
                name=f"{val}",
                legendgroup=f"{val}",
                showlegend=True,
                line=dict(color=colors[i % len(colors)]),
                opacity=0.5,
                hoverinfo="text",
                text=df_filtered_train.apply(
                    lambda row: f'Name: {row["name"]}<br>Dataset: {row["dataset"]}<br>Seed: {row["seed"]}<br>Transformation: {row["transformation"]}<br>Embed Apply LN: {row["embed_apply_ln"]}<br>Unembed Apply LN: {row["unembed_apply_ln"]}<br>Epoch: {row["epoch"]}<br>Loss: {row["loss"]}',
                    axis=1,
                ),
                customdata=df_filtered_train["name"],
            ),
            row=1,
            col=1,
        )
        # add train loss trace for highlighted name
        df_filtered_train = train_df.query(
            f"{color_var} == @val and name == @highlighted_name"
        )
        fig.add_trace(
            go.Scatter(
                x=df_filtered_train["epoch"],
                y=df_filtered_train["loss"],
                mode="lines",
                name=f"{val}",
                legendgroup=f"{val}",
                line=dict(color=colors[i % len(colors)]),
                opacity=1.0,
                hoverinfo="text",
                text=df_filtered_train.apply(
                    lambda row: f'Name: {row["name"]}<br>Dataset: {row["dataset"]}<br>Seed: {row["seed"]}<br>Transformation: {row["transformation"]}<br>Embed Apply LN: {row["embed_apply_ln"]}<br>Unembed Apply LN: {row["unembed_apply_ln"]}<br>Epoch: {row["epoch"]}<br>Loss: {row["loss"]}',
                    axis=1,
                ),
                customdata=df_filtered_train["name"],
            ),
            row=1,
            col=1,
        )

        # Add test loss trace for all but highlighted name
        fig.add_trace(
            go.Scatter(
                x=df_filtered_test["epoch"],
                y=df_filtered_test["loss"],
                mode="lines",
                name=f"{val}",
                legendgroup=f"{val}",
                showlegend=False,
                line=dict(color=colors[i % len(colors)]),
                opacity=0.5,
                hoverinfo="text",
                text=df_filtered_test.apply(
                    lambda row: f'Name: {row["name"]}<br>Dataset: {row["dataset"]}<br>Seed: {row["seed"]}<br>Transformation: {row["transformation"]}<br>Embed Apply LN: {row["embed_apply_ln"]}<br>Unembed Apply LN: {row["unembed_apply_ln"]}<br>Epoch: {row["epoch"]}<br>Loss: {row["loss"]}',
                    axis=1,
                ),
                customdata=df_filtered_test["name"],
            ),
            row=1,
            col=2,
        )
        # add test loss trace for highlighted name
        df_filtered_test = test_df.query(
            f"{color_var} == @val and name == @highlighted_name"
        )
        fig.add_trace(
            go.Scatter(
                x=df_filtered_test["epoch"],
                y=df_filtered_test["loss"],
                mode="lines",
                name=f"{val}",
                legendgroup=f"{val}",
                line=dict(color=colors[i % len(colors)]),
                hoverinfo="text",
                text=df_filtered_test.apply(
                    lambda row: f'Name: {row["name"]}<br>Dataset: {row["dataset"]}<br>Seed: {row["seed"]}<br>Transformation: {row["transformation"]}<br>Embed Apply LN: {row["embed_apply_ln"]}<br>Unembed Apply LN: {row["unembed_apply_ln"]}<br>Epoch: {row["epoch"]}<br>Loss: {row["loss"]}',
                    axis=1,
                ),
                customdata=df_filtered_test["name"],
            ),
            row=1,
            col=2,
        )

    max_epoch = preprocessed_df["epoch"].max()
    fig.update_layout(
        height=600,
        title_text="Loss Over Epochs by Selected Variable",
        xaxis=dict(range=[0, max_epoch + 1]),
        yaxis=dict(range=[-1, 0.2]),
    )

    return fig


@app.callback(
    [Output("loss-plot", "figure"), Output("placeholder-plot", "figure")],
    [
        Input("color-dropdown", "value"),
        Input("loss-plot", "hoverData"),
        Input("placeholder-plot", "hoverData"),
    ],
    prevent_initial_call=True,
)
def update_plots(color_var, hoverData_loss_plot, hoverData_placeholder_plot):
    """
    Callback to update both the loss plot and the placeholder plot based on the color variable or hover event.
    """
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "color-dropdown":
        # Generate figure based on the color variable
        fig = generate_figure_1(color_var)
        return fig, fig
    else:
        # Handle hover event to highlight corresponding plot
        # Determine which plot triggered the hover event
        if triggered_id == "loss-plot":
            hoverData = hoverData_loss_plot
        else:
            hoverData = hoverData_placeholder_plot

        if hoverData:
            # Extract the hovered run name
            hovered_run_name = hoverData["points"][0]["customdata"]
            print(hovered_run_name)
            # Highlight the corresponding plot in both subplots
            fig = generate_figure_1(color_var, highlighted_name=hovered_run_name)
            return fig, fig


if __name__ == "__main__":
    app.run_server(debug=True)
