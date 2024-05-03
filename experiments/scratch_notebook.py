# %%
import datetime
import itertools
import json
import os

import pandas as pd
from transformers import PreTrainedTokenizer

from auto_embeds.analytical import initialize_manual_transform

# Set environment variables
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["AUTOEMBEDS_CACHING"] = "true"

import numpy as np
import torch as t
import transformer_lens as tl
import wandb
from icecream import ic
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_embeds.data import filter_word_pairs, get_dataset_path
from auto_embeds.embed_utils import (
    initialize_embed_and_unembed,
    initialize_loss,
    initialize_transform_and_optim,
    train_transform,
)
from auto_embeds.metrics import (
    evaluate_accuracy,
    mark_translation,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.misc import get_experiment_worker_config, is_notebook
from auto_embeds.verify import (
    calc_tgt_is_closest_embed,
    plot_cosine_similarity_trend,
    prepare_verify_analysis,
    prepare_verify_datasets,
    test_cos_sim_difference,
    verify_transform,
)

# Seed for reproducibility
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

# Configuration for overall experiments
experiment_config = {
    "wandb": {
        "notes": "blank",
        "tags": [
            f"{datetime.datetime.now():%Y-%m-%d}",
            f"{datetime.datetime.now():%Y-%m-%d} analytical solutions",
            "experiment 1",
            "run group 1",
            # "actual",
            # "test",
        ],
    },
    "models": [
        "bigscience/bloom-560m",
        # "bloom-3b",
        # "bloom-7b",
    ],
    "processings": [
        False,
    ],
    "datasets": [
        # {
        #     "name": "wikdict_en_fr_extracted",
        #     "min_length": 5,
        #     "capture_space": True,
        #     "capture_no_space": False,
        #     "mark_accuracy_path": "wikdict_en_fr_azure_validation",
        # },
        # {
        #     "name": "random_word_pairs",
        #     "min_length": 2,
        #     "capture_space": True,
        #     "capture_no_space": False,
        # },
        # {
        #     "name": "singular_plural_pairs",
        #     "min_length": 2,
        #     "capture_space": True,
        #     "capture_no_space": False,
        # },
        # {
        #     "name": "muse_en_fr_extracted",
        #     "min_length": 5,
        #     "capture_space": True,
        #     "capture_no_space": False,
        #     "mark_accuracy_path": "muse_en_fr_azure_validation",
        # },
        {
            "name": "cc_cedict_zh_en_extracted",
            "min_length": 2,
            "capture_space": False,
            "capture_no_space": True,
            "mark_accuracy_path": "cc_cedict_zh_en_azure_validation",
        },
        # {
        #     "name": "muse_zh_en_extracted_train",
        #     "min_length": 2,
        #     "capture_space": False,
        #     "capture_no_space": True,
        #     "mark_accuracy_path": "muse_zh_en_azure_validation",
        # },
    ],
    "transformations": [
        # "identity",
        "translation",
        "linear_map",
        # "biased_linear_map",
        # "uncentered_linear_map",
        # "biased_uncentered_linear_map",
        "rotation",
        # "biased_rotation",
        # "uncentered_rotation",
        "analytical_rotation",
        "analytical_translation",
    ],
    "train_batch_sizes": [128],
    "test_batch_sizes": [256],
    "top_k": [200],
    "top_k_selection_methods": [
        # "src_and_src",
        # "tgt_and_tgt",
        "top_src",
        # "top_tgt",
    ],
    "seeds": [2],
    # "embed_weight": ["model_weights"],
    # "embed_ln": [True, False],
    # "embed_ln_weights": ["default_weights", "model_weights"],
    # "unembed_weight": ["model_weights"],
    # "unembed_ln": [True, False],
    # "unembed_ln_weights": ["default_weights", "model_weights"],
    "embed_weight": ["model_weights"],
    "embed_ln": [True],
    # "embed_ln_weights": ["default_weights", "model_weights"],
    "embed_ln_weights": ["model_weights"],
    "unembed_weight": ["model_weights"],
    "unembed_ln": [True],
    "unembed_ln_weights": ["model_weights"],
    "n_epochs": [150],
    "weight_decay": [
        0,
        # 2e-5,
    ],
    "lr": [8e-5],
}

total_runs = 1
for value in experiment_config.values():
    if isinstance(value, list):
        total_runs *= len(value)

print(f"Total experiment runs calculated: {total_runs}")


config_dict = get_experiment_worker_config(
    experiment_config=experiment_config,
    split_parameter="datasets",
    n_splits=1,
    worker_id=0,
)
# Extracting 'wandb' configuration and generating all combinations of configurations
# as a list of lists
wandb_config = config_dict.pop("wandb")
config_values = [
    config_dict[entry] if entry != "datasets" else config_dict[entry]
    for entry in config_dict
]
config_list = list(itertools.product(*config_values))

# To prevent unnecessary reloading
last_model_config = None
last_dataset_config = None
model = None
model_weights = None

results = {
    "transformation": [],
    "matrix": [],
    "test_cos_sims": [],
    "mark_translation_accuracy": [],
    "cos_sim_trend_plot": [],
}

# %%
for (
    model_name,
    processing,
    dataset_config,
    transformation,
    train_batch_size,
    test_batch_size,
    top_k,
    top_k_selection_method,
    seed,
    embed_weight,
    embed_ln,
    embed_ln_weights,
    unembed_weight,
    unembed_ln,
    unembed_ln_weights,
    n_epoch,
    weight_decay,
    lr,
) in tqdm(config_list, total=len(config_list)):

    run_config = {
        "model_name": model_name,
        "processing": processing,
        "dataset": dataset_config,
        "transformation": transformation,
        "train_batch_size": train_batch_size,
        "test_batch_size": test_batch_size,
        "top_k": top_k,
        "top_k_selection_method": top_k_selection_method,
        "seed": seed,
        "embed_weight": embed_weight,
        "embed_ln": embed_ln,
        "embed_ln_weights": embed_ln_weights,
        "unembed_weight": unembed_weight,
        "unembed_ln": unembed_ln,
        "unembed_ln_weights": unembed_ln_weights,
        "n_epoch": n_epoch,
        "weight_decay": weight_decay,
        "lr": lr,
    }

    # Tokenizer setup
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name
    )  # type: ignore

    # Model setup
    current_model_config = (model_name, processing)
    if current_model_config != last_model_config:
        if processing:
            model = tl.HookedTransformer.from_pretrained(model_name)
        else:
            model = tl.HookedTransformer.from_pretrained_no_processing(model_name)
        last_model_config = current_model_config

        model_weights = {
            "W_E": model.W_E.detach().clone(),
            "embed.ln.w": model.embed.ln.w.detach().clone(),
            "embed.ln.b": model.embed.ln.b.detach().clone(),
            "ln_final.w": model.ln_final.w.detach().clone(),
            "ln_final.b": model.ln_final.b.detach().clone(),
            "W_U": model.W_U.detach().clone(),
            "b_U": model.b_U.detach().clone(),
        }

        del model

    # Initialize embed and unembed modules
    embed_module, unembed_module = initialize_embed_and_unembed(
        tokenizer=tokenizer,
        model_weights=model_weights,
        embed_weight=embed_weight,
        embed_ln=embed_ln,
        embed_ln_weights=embed_ln_weights,
        unembed_weight=unembed_weight,
        unembed_ln=unembed_ln,
        unembed_ln_weights=unembed_ln_weights,
    )

    d_model = model_weights["W_E"].shape[1]
    n_toks = model_weights["W_E"].shape[0]

    # Dataset filtering
    dataset_name = dataset_config["name"]
    file_path = get_dataset_path(dataset_name)
    with open(file_path, "r", encoding="utf-8") as file:
        word_pairs = json.load(file)

    current_dataset_config = dataset_config
    if current_dataset_config != last_dataset_config:
        all_word_pairs = filter_word_pairs(
            tokenizer=tokenizer,
            word_pairs=word_pairs,
            discard_if_same=True,
            min_length=dataset_config["min_length"],
            capture_space=dataset_config["capture_space"],
            capture_no_space=dataset_config["capture_no_space"],
            print_number=True,
            verbose_count=True,
        )
        last_dataset_config = current_dataset_config

    # Prepare datasets
    verify_learning = prepare_verify_analysis(
        tokenizer=tokenizer,
        embed_module=embed_module,
        all_word_pairs=all_word_pairs,
        seed=seed,
        keep_other_pair=True,
    )

    train_loader, test_loader = prepare_verify_datasets(
        verify_learning=verify_learning,
        batch_sizes=(train_batch_size, test_batch_size),
        top_k=top_k,
        top_k_selection_method=top_k_selection_method,
    )

    if "mark_accuracy_path" in dataset_config:
        azure_translations_path = get_dataset_path(dataset_config["mark_accuracy_path"])
    else:
        azure_translations_path = None

    # Initialize transformation and optimizer
    if "analytical" in transformation:
        transform, expected_metrics = initialize_manual_transform(
            transform_name=transformation,
            train_loader=train_loader,
        )
        optim = None
    else:
        transform, optim = initialize_transform_and_optim(
            d_model,
            transformation=transformation,
            optim_kwargs={"lr": lr, "weight_decay": weight_decay},
        )

    loss_module = initialize_loss("cosine_similarity")

    # Train transformation
    if optim is not None:
        transform, loss_history = train_transform(
            tokenizer=tokenizer,
            train_loader=train_loader,
            test_loader=test_loader,
            transform=transform,
            optim=optim,
            unembed_module=unembed_module,
            loss_module=loss_module,
            n_epochs=n_epoch,
            plot_fig=True,
            azure_translations_path=azure_translations_path,
        )

    results["transformation"].append(transformation)
    if "analytical" in transformation:
        results["matrix"].append(transform.transformations[0][1].detach().clone().cpu())
    elif "rotation" in transformation:
        results["matrix"].append(transform.rotation.weight.detach().clone().cpu())
    elif "translation" in transformation:
        results["matrix"].append(transform.translation.data.detach().clone().cpu())
    elif "linear" in transformation:
        results["matrix"].append(transform.linear.weight.detach().clone().cpu())
    else:
        raise ValueError(f"Unknown transformation: {transformation}")

    # Evaluate and log results
    test_accuracy = evaluate_accuracy(
        tokenizer=tokenizer,
        test_loader=test_loader,
        transformation=transform,
        unembed_module=unembed_module,
        exact_match=False,
        print_results=False,
        print_top_preds=False,
    )

    if azure_translations_path is None:
        mark_translation_acc = None
    else:
        mark_translation_acc = mark_translation(
            tokenizer=tokenizer,
            transformation=transform,
            unembed_module=unembed_module,
            test_loader=test_loader,
            azure_translations_path=azure_translations_path,
            print_results=False,
        )

    verify_results_dict = verify_transform(
        tokenizer=tokenizer,
        transformation=transform,
        test_loader=test_loader,
        unembed_module=unembed_module,
    )

    cos_sims_trend_plot = plot_cosine_similarity_trend(verify_results_dict)

    cos_sims_trend_plot.show(config={"responsive": True, "autosize": True})

    test_cos_sim_diff = test_cos_sim_difference(verify_results_dict)

    results["test_cos_sims"].append(verify_results_dict["cos_sims"].cpu())
    results["mark_translation_accuracy"].append(mark_translation_acc)
    results["cos_sim_trend_plot"].append(cos_sims_trend_plot)

# %%
results_df = pd.DataFrame(results)

# %%
results_df

# %%
results_df = results_df.assign(
    test_cos_sims=lambda df: df["test_cos_sims"].apply(lambda x: x.mean()),
    # matrix=lambda df: df["matrix"].apply(lambda x: x.numpy()),
)

# %%
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# %%


def plot_matrix_visualizations(results_df):
    """Plot the first three matrices from the results dataframe as heatmaps."""
    # Number of matrices to plot
    num_matrices = len(results_df["matrix"][0:3])

    # Create subplots
    fig = make_subplots(rows=num_matrices, cols=1)

    # Add each matrix as a heatmap to the subplots
    for i, matrix in enumerate(results_df["matrix"].iloc[0:3], start=1):
        fig.add_trace(
            go.Heatmap(z=matrix, colorscale="Viridis", showscale=True), row=i, col=1
        )

    # Update layout to make each plot square
    fig.update_layout(
        height=1024 * num_matrices, width=1024, title_text="Matrix Visualizations"
    )

    # Show the figure
    fig.show()


plot_matrix_visualizations(results_df)

# %%


def cosine_similarity_matrix(matrices):
    """
    Computes the cosine similarity matrix for a list of matrices.

    Args:
        matrices (list of t.Tensor): List of matrices to compute similarity.

    Returns:
        t.Tensor: A matrix of cosine similarities.
    """
    # Flatten each matrix to a vector for cosine similarity computation
    vectors = [mat.flatten() for mat in matrices]
    print([mat.shape for mat in matrices])
    # print([vec.shape for vec in vectors])
    similarity_matrix = t.nn.functional.cosine_similarity(
        t.stack(vectors).unsqueeze(1), t.stack(vectors).unsqueeze(0), dim=2
    )
    return similarity_matrix


def plot_similarity_heatmap(similarity_matrix, title="Cosine Similarity Heatmap"):
    """
    Plots a heatmap of the cosine similarity matrix.

    Args:
        similarity_matrix (t.Tensor): Cosine similarity matrix to plot.
        title (str): Title of the heatmap.
    """
    fig = px.imshow(
        similarity_matrix,
        labels=dict(x="Matrix Index", y="Matrix Index", color="Cosine Similarity"),
        x=np.arange(similarity_matrix.shape[0]),
        y=np.arange(similarity_matrix.shape[1]),
        title=title,
        color_continuous_scale="Viridis",  # Color scale can be adjusted
    )
    fig.update_xaxes(side="top")
    fig.show()


def visualize_matrix_similarity(df):
    """
    Extracts matrices from DataFrame, computes their cosine similarity, and plots it.

    Args:
        df (pd.DataFrame): DataFrame containing a column 'matrix' with matrices.
    """
    matrices = df["matrix"].tolist()
    similarity_matrix = cosine_similarity_matrix(matrices)
    plot_similarity_heatmap(similarity_matrix)


# Example usage within the module
visualize_matrix_similarity(results_df)

# %%


def frobenius_norm_difference(matrix1, matrix2):
    print(matrix1)
    return t.linalg.norm(matrix1 - matrix2, "fro")


def spectral_norm_difference(matrix1, matrix2):
    return t.linalg.norm(matrix1 - matrix2, 2)


def rotation_angle(matrix1, matrix2):
    R = t.matmul(matrix2.T, matrix1)
    theta = t.acos((t.trace(R) - 1) / 2)
    return theta


def compute_similarity_matrices(matrices, similarity_func):
    n = len(matrices)
    similarity_matrix = t.full((n, n), float("nan"))  # Initialize with np.nan
    for i in range(n):
        for j in range(i):
            similarity = similarity_func(matrices[i], matrices[j])
            similarity_matrix[i, j] = similarity
    return similarity_matrix


def plot_heatmap(similarity_matrix, title):
    fig = px.imshow(
        similarity_matrix.numpy(),
        labels=dict(x="Matrix Index", y="Matrix Index", color="Similarity Measure"),
        x=np.arange(similarity_matrix.shape[0]),
        y=np.arange(similarity_matrix.shape[1]),
        title=title,
        color_continuous_scale="Viridis",
    )
    fig.update_xaxes(side="top")
    fig.show()


# Extract matrices from the results DataFrame
matrices = results_df["matrix"].tolist()

# Compute similarity matrices
frobenius_sim_matrix = compute_similarity_matrices(matrices, frobenius_norm_difference)
spectral_sim_matrix = compute_similarity_matrices(matrices, spectral_norm_difference)
rotation_angle_matrix = compute_similarity_matrices(matrices, rotation_angle)

# Plot heatmaps
plot_heatmap(frobenius_sim_matrix, "Frobenius Norm Difference Heatmap")

# %%
plot_heatmap(spectral_sim_matrix, "Spectral Norm Difference Heatmap")
# %%
plot_heatmap(rotation_angle_matrix, "Rotation Angle Heatmap")

# %%
print(tokenizer.encode(" queen"))
# %%
# Testing the analogy "king - queen + woman = man" using word embeddings
king_vec = word_embeddings["king"]
queen_vec = word_embeddings["queen"]
woman_vec = word_embeddings["woman"]
man_vec = word_embeddings["man"]

# Compute the result of the analogy
result_vec = king_vec - queen_vec + woman_vec

# Find the closest word to the result vector
closest_word = find_closest_word(result_vec, word_embeddings)

# Check if the closest word is 'man'
is_correct_analogy = closest_word == "man"
print(f"Analogy 'king - queen + woman = man' test result: {is_correct_analogy}")

# %%
tgt_is_closest_embed = calc_tgt_is_closest_embed(
    tokenizer=tokenizer,
    all_word_pairs=all_word_pairs,
    embed_module=embed_module,
)

# %%
verify_results_dict

# %%
import plotly.graph_objects as go
import torch


def plot_difference_heatmap(matrix1, matrix2, title):
    """Plots the difference between two matrices using a heatmap.

    Args:
        matrix1 (torch.Tensor): First matrix.
        matrix2 (torch.Tensor): Second matrix.
        title (str): Title of the heatmap.
    """
    if matrix1.shape != matrix2.shape:
        raise ValueError("Both matrices must have the same dimensions.")

    # Compute the absolute difference between the two matrices
    difference_matrix = torch.abs(matrix1 - matrix2).cpu().numpy()

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=difference_matrix,
            x=list(range(matrix1.shape[1])),
            y=list(range(matrix1.shape[0])),
            colorscale="Viridis",
        )
    )

    fig.update_layout(title=title, xaxis_title="Index", yaxis_title="Index")
    fig.show()


# %%

plot_difference_heatmap(
    model_weights["embed.ln.w"], model_weights["ln_final.w"], "embed.ln.w - ln_final.w"
)


# %%
def plot_matrix_similarities(matrices):
    """Plots the cosine similarity and Euclidean distance heatmaps for given matrices.

    Args:
        matrices (dict): A dictionary of matrices.
    """
    matrix_names = list(matrices.keys())

    # Initialize matrices for cosine similarity and Euclidean distance
    cosine_similarity_matrix = t.zeros((len(matrices), len(matrices)))
    euclidean_distance_matrix = t.zeros((len(matrices), len(matrices)))

    # Calculate cosine similarity and Euclidean distance
    for i, mat1 in enumerate(matrix_names):
        for j, mat2 in enumerate(matrix_names):
            if i <= j:
                cosine_similarity_matrix[i, j] = t.tensor(float("nan"))
                euclidean_distance_matrix[i, j] = t.tensor(float("nan"))
            else:
                cosine_similarity_matrix[i, j] = t.nn.functional.cosine_similarity(
                    matrices[mat1], matrices[mat2], dim=-1
                ).mean()
                euclidean_distance_matrix[i, j] = t.norm(
                    matrices[mat1] - matrices[mat2], p=2
                )

    # Move matrices to CPU for plotting
    cosine_similarity_matrix = cosine_similarity_matrix.cpu().numpy()
    euclidean_distance_matrix = euclidean_distance_matrix.cpu().numpy()

    # Plot cosine similarity matrix using Plotly
    fig_cosine = go.Figure(
        data=go.Heatmap(
            z=cosine_similarity_matrix,
            x=matrix_names,
            y=matrix_names,
            colorscale="coolwarm",
        )
    )
    fig_cosine.update_layout(
        title="Matrix Cosine Similarity", xaxis_title="Matrix", yaxis_title="Matrix"
    )
    fig_cosine.show()

    # Plot Euclidean distance matrix using Plotly
    fig_euclidean = go.Figure(
        data=go.Heatmap(
            z=euclidean_distance_matrix,
            x=matrix_names,
            y=matrix_names,
            colorscale="Viridis",
        )
    )
    fig_euclidean.update_layout(
        title="Matrix Euclidean Distance", xaxis_title="Matrix", yaxis_title="Matrix"
    )
    fig_euclidean.show()


model_square_weights = {
    "W_E": model_weights["W_E"],
    "embed.ln.w": model_weights["embed.ln.w"],
    "ln_final.w": model_weights["ln_final.w"],
    "W_U": model_weights["W_U"],
    "b_U": model_weights["b_U"],
}

for matrix in model_square_weights:
    print(model_square_weights[matrix].shape)

# %%
plot_matrix_similarities(model_square_weights)

# %%
print(model_weights["embed.ln.w"])
print(model_weights["ln_final.w"])

# %%
t.norm(model_weights["embed.ln.w"], 1)
# %%
t.norm(model_weights["ln_final.w"], 1)
# %%
W_E = model_weights["W_E"]
embed_ln_w = model_weights["embed.ln.w"]
ln_final_w = model_weights["ln_final.w"]
W_U = model_weights["W_U"]
b_U = model_weights["b_U"]


# %%
def plot_tensor_histogram(tensor, tensor_name):
    """
    Plots a histogram for the values in a given tensor.

    Args:
        tensor (torch.Tensor): The tensor to plot.
        tensor_name (str): The name of the tensor, used for labeling the plot.
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=tensor.cpu().numpy(), name=tensor_name, opacity=0.75))
    fig.update_layout(
        title_text=f"Distribution of Values in {tensor_name}",
        xaxis_title_text="Value",
        yaxis_title_text="Count",
        bargap=0.2,
    )
    fig.show()


plot_tensor_histogram(embed_ln_w, "embed_ln_w")
plot_tensor_histogram(W_E, "W_E")
plot_tensor_histogram(ln_final_w, "ln_final_w")
plot_tensor_histogram(W_U, "W_U")
plot_tensor_histogram(b_U, "b_U")

# %%
