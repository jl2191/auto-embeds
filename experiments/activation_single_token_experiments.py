# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch as t
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import transformer_lens as tl
from IPython.core.getipython import get_ipython
import wandb

from auto_steer.steering_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    initialize_transform_and_optim,
    train_transform,
    run_and_gather_acts,
    filter_word_pairs,
    tokenize_word_pairs,
    save_acts,
    load_test_strings,
    initialize_loss,
    perform_steering_tests,
)
from auto_steer.utils.misc import repo_path_to_abs_path

ipython = get_ipython()
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)
try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("load_ext", "line_profiler")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except:
    pass

# %% model setup
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
model_caches_folder = repo_path_to_abs_path("datasets/model_caches")
token_caches_folder = repo_path_to_abs_path("datasets/token_caches")

# %% -----------------------------------------------------------------------------------
file_path = f"{datasets_folder}/wikdict/2_extracted/eng-fra.json"
with open(file_path, "r") as file:
    word_pairs = json.load(file)
random.seed(1)
random.shuffle(word_pairs)
split_index = int(len(word_pairs) * 0.95)
train_en_fr_pairs = word_pairs[:split_index]
test_en_fr_pairs = word_pairs[split_index:]

train_word_pairs = filter_word_pairs(
    model,
    train_en_fr_pairs,
    discard_if_same=True,
    min_length=4,
    # capture_diff_case=True,
    capture_space=True,
    capture_no_space=True,
    print_pairs=True,
    print_number=True,
    max_token_id=100_000,
    most_common_english=True,
    # most_common_french=True,
)

test_word_pairs = filter_word_pairs(
    model,
    test_en_fr_pairs,
    discard_if_same=True,
    min_length=4,
    # capture_diff_case=True,
    capture_space=True,
    capture_no_space=True,
    # print_pairs=True,
    print_number=True,
    max_token_id=100_000,
    most_common_english=True,
    # most_common_french=True,
)

train_en_toks, train_fr_toks, train_en_mask, train_fr_mask = tokenize_word_pairs(
    model, train_word_pairs
)
test_en_toks, test_fr_toks, test_en_mask, test_fr_mask = tokenize_word_pairs(
    model, test_word_pairs
)
# %%
t.save(
    {
        "en_toks": train_en_toks,
        "fr_toks": train_fr_toks,
        "en_mask": train_en_mask,
        "fr_mask": train_fr_mask,
    },
    f"{token_caches_folder}/wikdict-train-en-fr-tokens.pt",
)

t.save(
    {
        "en_toks": test_en_toks,
        "fr_toks": test_fr_toks,
        "en_mask": test_en_mask,
        "fr_mask": test_fr_mask,
    },
    f"{token_caches_folder}/wikdict-test-en-fr-tokens.pt",
)
# %%
train_data = t.load(f"{token_caches_folder}/wikdict-train-en-fr-tokens.pt")
test_data = t.load(f"{token_caches_folder}/wikdict-test-en-fr-tokens.pt")

train_en_toks = train_data["en_toks"]
train_fr_toks = train_data["fr_toks"]
train_en_mask = train_data["en_mask"]
train_fr_mask = train_data["fr_mask"]

test_en_toks = test_data["en_toks"]
test_fr_toks = test_data["fr_toks"]
test_en_mask = test_data["en_mask"]
test_fr_mask = test_data["fr_mask"]

# %%
train_en_embeds = (
    model.embed.W_E[train_en_toks].detach().clone()
)  # shape[batch, pos, d_model]
test_en_embeds = (
    model.embed.W_E[test_en_toks].detach().clone()
)  # shape[batch, pos, d_model]
train_fr_embeds = (
    model.embed.W_E[train_fr_toks].detach().clone()
)  # shape[batch, pos, d_model]
test_fr_embeds = (
    model.embed.W_E[test_fr_toks].detach().clone()
)  # shape[batch, pos, d_model]

# train_en_embeds = t.nn.functional.layer_norm(
#     model.embed.W_E[train_en_toks].detach().clone(), [model.cfg.d_model]
# )
# train_fr_embeds = t.nn.functional.layer_norm(
#     model.embed.W_E[train_fr_toks].detach().clone(), [model.cfg.d_model]
# )
# test_en_embeds = t.nn.functional.layer_norm(
#     model.embed.W_E[test_en_toks].detach().clone(), [model.cfg.d_model]
# )
# test_fr_embeds = t.nn.functional.layer_norm(
#     model.embed.W_E[test_fr_toks].detach().clone(), [model.cfg.d_model]
# )

train_dataset = TensorDataset(train_en_embeds, train_fr_embeds)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = TensorDataset(test_en_embeds, test_fr_embeds)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
# %%
# run = wandb.init(
#     project="single_token_tests",
# )
# %%

transformation_names = [
    "identity",
    "translation",
    "linear_map",
    "biased_linear_map",
    "uncentered_linear_map",
    "biased_uncentered_linear_map",
    "rotation",
    "biased_rotation",
    "uncentered_rotation",
]

for transformation_name in transformation_names:
    transform = None
    optim = None

    transform, optim = initialize_transform_and_optim(
        d_model,
        transformation=transformation_name,
        # optim_kwargs={"lr": 2e-4},
        optim_kwargs={"lr": 2e-4, "weight_decay": 2e-5},
    )
    loss_module = initialize_loss("cosine_similarity")

    if optim is not None:
        transform, loss_history = train_transform(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            transform=transform,
            optim=optim,
            loss_module=loss_module,
            n_epochs=200,
            # wandb=wandb,
        )
    else:
        print(f"nothing trained for {transformation_name}")
    print(f"{transformation_name}:")
    transform.eval()
    accuracy = evaluate_accuracy(
        model,
        test_loader,
        transform,
        exact_match=False,
        print_results=True,
    )
    print(f"{transformation_name}:")
    print(f"Correct Percentage: {accuracy * 100:.2f}%")
    print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))

# %% -----------------------------------------------------------------------------------
# %% gather activations

train_dataset = TensorDataset(
    train_en_toks, train_fr_toks, train_en_mask, train_fr_mask
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = TensorDataset(test_en_toks, test_fr_toks, test_en_mask, test_fr_mask)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

train_en_resids, train_fr_resids = run_and_gather_acts(
    model, train_loader, layers=[0, 1, 12, 18, 22, 23]
)
test_en_resids, test_fr_resids = run_and_gather_acts(
    model, test_loader, layers=[0, 1, 4, 6, 8, 10, 12, 14, 16, 18, 19, 22, 23]
)
# %% save activations
filename_base = "bloom-560m-wikdict-train"
save_acts(model_caches_folder, filename_base, train_en_resids, train_fr_resids)
filename_base = "bloom-560m-wikdict-test"
save_acts(model_caches_folder, filename_base, test_en_resids, test_fr_resids)

# %% load activations
train_en_resids = t.load(
    f"{model_caches_folder}/bloom-560m-wikdict-train-en-layers-[0, 1, 4, 6, 8, 10, 12, 14, 16, 18, 19, 22, 23].pt"
)
train_fr_resids = t.load(
    f"{model_caches_folder}/bloom-560m-wikdict-train-fr-layers-[0, 1, 4, 6, 8, 10, 12, 14, 16, 18, 19, 22, 23].pt"
)
test_en_resids = t.load(
    f"{model_caches_folder}/bloom-560m-wikdict-test-en-layers-[0, 1, 4, 6, 8, 10, 12, 14, 16, 18, 19, 22, 23].pt"
)
test_fr_resids = t.load(
    f"{model_caches_folder}/bloom-560m-wikdict-test-fr-layers-[0, 1, 4, 6, 8, 10, 12, 14, 16, 18, 19, 22, 23].pt"
)

# %%
train_en_resids = {
    layer: t.cat(train_en_resids[layer], dim=0) for layer in train_en_resids
}
train_fr_resids = {
    layer: t.cat(train_fr_resids[layer], dim=0) for layer in train_fr_resids
}
test_en_resids = {
    layer: t.cat(test_en_resids[layer], dim=0) for layer in test_en_resids
}
test_fr_resids = {
    layer: t.cat(test_fr_resids[layer], dim=0) for layer in test_fr_resids
}

# %% train en to fr residual in layers [0, 1, 4, 6, 8, 10, 12, 14, 16, 18, 19, 22, 23]

results = {"train": {}, "test": {}}

transformation_names = [
    # "identity",
    # "translation",
    "linear_map",
    # "biased_linear_map",
    # "uncentered_linear_map",
    # "biased_uncentered_linear_map",
    # "rotation",
    # "biased_rotation",
    # "uncentered_rotation",
]
for transformation_name in transformation_names:

    results["train"][transformation_name] = {}
    results["test"][transformation_name] = {}

    for layer in train_en_resids:

        train_dataset = TensorDataset(train_en_resids[layer], train_fr_resids[layer])
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_dataset = TensorDataset(test_en_resids[layer], test_fr_resids[layer])
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

        transform = None
        optim = None

        transform, optim = initialize_transform_and_optim(
            d_model,
            transformation=transformation_name,
            optim_kwargs={"lr": 2e-4, "weight_decay": 1e-5},
            # optim_kwargs={"lr": 1e-4, "weight_decay": 1e-5}
            # optim_kwargs={"lr": 5e-4}
            # optim_kwargs={"lr": 2e-4}
            # optim_kwargs={"lr": 1e-4}
            # optim_kwargs={"lr": 5e-4, "weight_decay": 2e-5}
        )

        loss_module = initialize_loss("cosine_similarity")

        if optim is not None:
            transform, loss_history = train_transform(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                transform=transform,
                optim=optim,
                loss_module=loss_module,
                n_epochs=50,
                # wandb=wandb,
            )
        else:
            print(f"nothing trained for {transformation_name}")

        # accuracy = evaluate_accuracy(
        #     model, test_loader, transform, exact_match=False, print_results=True
        # )

        test_loss = calc_cos_sim_acc(test_loader, transform)

        print(f"{transformation_name}:")
        print(f"layer number: {layer}")
        print("test loss:", calc_cos_sim_acc(test_loader, transform))

        results["test"][transformation_name][layer] = test_loss

# %%
for transformation_name in results["test"]:
    for layer in results["test"][transformation_name]:
        print(
            f"Transformation {transformation_name}, Layer {layer}, Test Loss: {results['test'][transformation_name][layer]:.2f}%"
        )
        print()

import plotly.graph_objects as go

# Extract layer numbers and their corresponding accuracies for both transformations
layer_numbers = sorted(list(results["test"][list(results["test"].keys())[0]].keys()))
transformation_names = list(results["test"].keys())

# Prepare data for plotting
data = []
for transformation_name in transformation_names:
    test_loss = [results["test"][transformation_name][layer] for layer in layer_numbers]
    data.append(
        go.Scatter(
            x=layer_numbers,
            y=test_loss,
            mode="lines+markers",
            name=transformation_name,
        )
    )

# Create the plot
fig = go.Figure(data=data)
fig.update_layout(
    title="Test Loss by Layer Number",
    xaxis_title="Layer Number",
    yaxis_title="Test Loss (Cosine Similarity)",
    legend_title="Transformation Type",
)
fig.show()
