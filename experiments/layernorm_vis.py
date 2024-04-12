# %%
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch as t
import transformer_lens as tl
import umap

from auto_embeds.data import prepare_data
from auto_embeds.metrics import calc_canonical_angles
from auto_embeds.utils.custom_tqdm import tqdm

# Set environment variables
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Seed for reproducibility
# np.random.seed(1)
# t.manual_seed(1)
# t.cuda.manual_seed(1)

# %%
# Experiment configurations
models = ["bloom-560m"]
processings = [True, False]
layernorms = [True, False]
samplings = [
    "first",
    # "random",
    # "all",
]
n_embeddings = 5000

# Results storage
results = {}
import itertools

config = {
    "models": ["bloom-560m"],
    "processings": [True, False],
    "layernorms": [True, False],
    "samplings": ["first", "random", "all"],
    "n_embeddings": 1000,
    "vis_dims": [2, 3],  # 2d or 3d
    "umap_params": [
        {"n_neighbors": 15, "min_dist": 0.3, "metric": "cosine", "random_state": 1},
        {"n_neighbors": 10, "min_dist": 0.1, "metric": "euclidean", "random_state": 1},
    ],
}

# %%
config = {
    "models": ["bloom-560m"],
    "processings": [True, False],
    "layernorms": [True, False],  # Keep both options to compare
    "samplings": ["first"],
    "n_embeddings": 1000,
    "vis_dims": [3],  # 2d or 3d
    "umap_params": [
        {"n_neighbors": 15, "min_dist": 0.3, "metric": "cosine", "random_state": 1},
    ],
}
from math import prod

# Calculate the product of the lengths of the lists in the config dictionary
total_plots = prod(len(value) for value in config.values() if isinstance(value, list))
print(f"Total plots to be generated: {total_plots}")

# Results storage
results = {}
max_plots = None
# %%
# Main experiment loop
# Preload models with and without processing to avoid reloading for each configuration
# Initialize a variable to keep track of the last loaded model for optimization
last_loaded_model = None

# %%
# counter variable for limiting the number of plots for debugging
plot_counter = 0
for model_name, processing, layernorm, sample, vis_dim, umap_param in tqdm(
    itertools.product(
        config["models"],
        config["processings"],
        [False],  # Only iterate without layernorm here
        config["samplings"],
        config["vis_dims"],
        config["umap_params"],
    ),
    desc="Configurations",
):

    # Break out of the loop if the plot counter reaches the max_plots limit
    if max_plots is not None and plot_counter >= max_plots:
        break

    model_results = results.setdefault(model_name, {})
    # Check if the current model combination is the same as the last loaded model to
    # avoid unnecessary reloading
    model_needs_loading = (
        last_loaded_model is None
        or last_loaded_model["model_name"] != model_name
        or last_loaded_model["processing"] != processing
    )
    if model_needs_loading:
        model = (
            tl.HookedTransformer.from_pretrained(model_name)
            if processing
            else tl.HookedTransformer.from_pretrained_no_processing(model_name)
        )
        last_loaded_model = {
            "model_name": model_name,
            "processing": processing,
            "model": model,
        }

    # Ensure model.tokenizer is not None and is callable to satisfy linter
    if model.tokenizer is None or not callable(model.tokenizer):
        raise ValueError("model.tokenizer is not set or not callable")

    tok_ids = t.arange(model.cfg.d_vocab)[: config["n_embeddings"]]

    # Get the embeddings without layernorm
    embeddings = model.embed.W_E.detach().cpu()[tok_ids]
    embeddings_layernorm = t.nn.functional.layer_norm(embeddings, [model.cfg.d_model])

    # Convert both embeddings to numpy for UMAP
    embeddings = embeddings.numpy()
    embeddings_layernorm = embeddings_layernorm.numpy()

    # Get string representation of tokens
    tok_strs = [model.tokenizer.batch_decode([tok_id]) for tok_id in tok_ids]

    # Apply UMAP
    import timeit

    start_time = timeit.default_timer()
    reducer = umap.UMAP(n_components=vis_dim, **umap_param)
    # Apply UMAP to both embeddings and layernorm embeddings
    umap_embeddings = reducer.fit_transform(embeddings)
    umap_embeddings_layernorm = reducer.transform(embeddings_layernorm)
    elapsed = timeit.default_timer() - start_time
    print(f"UMAP fit_transform time: {elapsed} seconds")

    # Visualization with Plotly for both embeddings
    fig = None
    if vis_dim == 2:
        fig = px.scatter(
            umap_embeddings,
            x=0,
            y=1,
            title="UMAP Visualization of Word Embeddings",
            hover_data=[tok_ids, tok_strs],
            labels={"hover_data_0": "Token", "hover_data_1": "String"},
            color_discrete_sequence=["blue"],  # Original embeddings in blue
        )
        fig.add_scatter(
            x=umap_embeddings_layernorm[:, 0],
            y=umap_embeddings_layernorm[:, 1],
            mode="markers",
            marker=dict(color="red"),  # Layernorm embeddings in red
            name="Layernorm Applied",
        )
        fig.update_xaxes(title_text="Component 1")
        fig.update_yaxes(title_text="Component 2")
    elif vis_dim == 3:
        fig = px.scatter_3d(
            umap_embeddings,
            x=0,
            y=1,
            z=2,
            title="UMAP Visualization of Word Embeddings",
            hover_data=[tok_ids, tok_strs],
            labels={"hover_data_0": "Token", "hover_data_1": "String"},
            color_discrete_sequence=["blue"],  # Original embeddings in blue
        )
        fig.add_trace(
            go.Scatter3d(
                x=umap_embeddings_layernorm[:, 0],
                y=umap_embeddings_layernorm[:, 1],
                z=umap_embeddings_layernorm[:, 2],
                mode="markers",
                marker=dict(color="red"),  # Layernorm embeddings in red
                name="Layernorm Applied",
            )
        )
        fig.update_traces(marker_size=3)
        fig.update_layout(
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3",
            )
        )
    if fig:
        fig.add_annotation(
            text="<br>".join(
                [
                    f"model_name: {model_name}",
                    f"processing: {processing}",
                    f"layernorm: {layernorm}",
                    f"sample: {sample}",
                    f"vis_dim: {vis_dim}",
                ]
                + [f"{k}: {v}" for k, v in umap_param.items()]
            ),
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=1.45,
            y=0.6,
            font=dict(size=10),
        )
        fig.update_layout(margin=dict(r=160))
        fig.show()
        # Increment the plot counter after each plot
        plot_counter += 1
    print(f"Plotly time: {elapsed} seconds")

# %%
import plotly.graph_objects as go

# Calculate the norm of embeddings with and without LayerNorm
norm_all_embeds = t.norm(all_embeds, dim=-1).cpu().numpy()
norm_all_embeds_ln = t.norm(all_embeds_ln, dim=-1).cpu().numpy()

# Prepare data for bar chart plotting
norm_data = go.Bar(
    x=list(range(len(norm_all_embeds))),
    y=norm_all_embeds_ln,
    name="Norm of Embeddings without LayerNorm",
    marker=dict(color="blue"),
    opacity=0.75,
)

norm_data_ln = go.Bar(
    x=list(range(len(norm_all_embeds_ln))),
    y=norm_all_embeds_ln,
    name="Norm of Embeddings with LayerNorm",
    marker=dict(color="red"),
    opacity=0.75,
)

data = [norm_data, norm_data_ln]

# Create the bar chart
fig = go.Figure(data=data)
fig.update_layout(
    title="Norm of Embeddings with and without LayerNorm",
    xaxis_title="Embedding Index",
    yaxis_title="Norm",
    legend_title="Embedding Type",
    barmode="group",
)
fig.show()
