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
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

# %%

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
    "layernorms": [True, False],
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
        config["layernorms"],
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

    tok_ids = t.arange(model.cfg.d_vocab)

    # reducing from (250880, 1024) for speed
    if sample == "first":
        tok_ids = t.arange(model.cfg.d_vocab)[: config["n_embeddings"]]
    elif sample == "random":
        tok_ids = t.randperm(model.cfg.d_vocab)[: config["n_embeddings"]]
    elif sample == "all":
        pass

    # Get the embeddings
    embeddings = model.embed.W_E.detach().cpu()[tok_ids]
    if layernorm:
        embeddings = t.nn.functional.layer_norm(embeddings, [model.cfg.d_model])
    embeddings = embeddings.numpy()

    # Get string representation of tokens
    tok_strs = [model.tokenizer.batch_decode([tok_id]) for tok_id in tok_ids]

    # Apply UMAP
    import timeit

    # time
    start_time = timeit.default_timer()
    # time
    reducer = umap.UMAP(n_components=vis_dim, **umap_param)
    umap_embeddings = reducer.fit_transform(embeddings)
    ### time ###
    elapsed = timeit.default_timer() - start_time
    print(f"umap fit_transform time: {elapsed} seconds")
    ### time ###

    ### time ###
    start_time = timeit.default_timer()
    ### time ###
    # Visualization with Plotly
    fig = None
    if vis_dim == 2:
        fig = px.scatter(
            umap_embeddings,
            x=0,
            y=1,
            title="UMAP Visualization of Word Embeddings",
            hover_data=[tok_ids, tok_strs],
            labels={"hover_data_0": "Token", "hover_data_1": "String"},
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
    ### time ###
    elapsed = timeit.default_timer() - start_time
    print(f"plotly time: {elapsed} seconds")
    ### time ###

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming the Bloom model and tokenizer are initialized similarly to your existing setup
model_name = "bigscience/bloom-560m"
model.eval()  # Set the model to evaluation mode


# Function to get embeddings for words
def get_word_embeddings(words):
    inputs = model.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.detach().numpy()


# Function to visualize embeddings in 2D
def visualize_embeddings_2D(embeddings, labels):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(12, 7))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    for i, word in enumerate(labels):
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.show()


# Word Embeddings Visualization
words = ["man", "woman", "king", "queen", "apple", "orange"]
embeddings = get_word_embeddings(words)
visualize_embeddings_2D(embeddings, words)


# Function to get sentence embeddings
def get_sentence_embeddings(sentences):
    inputs = model.tokenizer(
        sentences, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.detach().numpy()


# Sentence Embeddings Visualization
sentences = [
    "The sky is blue today.",
    "The sea is blue today.",
    "This is a horrible idea!",
    "This was a horrible idea!",
    "This will be a horrible idea!",
]
sentence_embeddings = get_sentence_embeddings(sentences)
visualize_embeddings_2D(sentence_embeddings, sentences)

# Additional experiments as per the original file can be adapted similarly
