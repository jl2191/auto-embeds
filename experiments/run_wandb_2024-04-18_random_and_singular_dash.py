# Import necessary libraries
from dash import Dash, dcc, html, Input, Output, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import pandas as pd
from auto_embeds.utils.misc import (
    dynamic_text_wrap,
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

# Initialize the app with Dash Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

# App layout using Dash Bootstrap Components
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Dash App with Interactive Plots", className="text-center"),
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="color-dropdown",
                    options=[
                        {"label": col, "value": col} for col in preprocessed_df.columns
                    ],
                    value="dataset",
                ),
                width={"size": 6, "offset": 3},
            )
        ),
        dbc.Row(dbc.Col(dcc.Graph(id="loss-plot"), width=12)),
    ],
    fluid=True,
)


@app.callback(
    Output("loss-plot", "figure"),
    [
        Input("color-dropdown", "value"),
    ],
)
def update_dynamic_plot(color_var):
    # Filter the DataFrame for train and test loss separately
    train_df = preprocessed_df[preprocessed_df["loss_type"] == "train_loss"]
    test_df = preprocessed_df[preprocessed_df["loss_type"] == "test_loss"]

    # Create subplots: one for train_loss and another for test_loss
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Train Loss", "Test Loss"))

    # Plot train_loss on the left
    fig.add_trace(
        go.Scatter(
            x=train_df["epoch"],
            y=train_df["loss"],
            mode="lines+markers",
            name="Train Loss",
            marker_color="blue",
        ),
        row=1,
        col=1,
    )

    # Plot test_loss on the right
    fig.add_trace(
        go.Scatter(
            x=test_df["epoch"],
            y=test_df["loss"],
            mode="lines+markers",
            name="Test Loss",
            marker_color="red",
        ),
        row=1,
        col=2,
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)

    # Update layout for a better visual presentation
    fig.update_layout(
        height=600, width=1200, title_text="Train vs Test Loss Over Epochs"
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
