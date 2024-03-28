# %%
import json
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch as t
import torch.nn as nn
import transformer_lens as tl
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.data import get_dataset_path
from auto_embeds.embed_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    mark_translation,
    tokenize_word_pairs,
)
from auto_embeds.utils.misc import repo_path_to_abs_path

# %%
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

# %% model setup
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
# model = tl.HookedTransformer.from_pretrained_no_processing("gpt2-small")
# model = tl.HookedTransformer.from_pretrained_no_processing("gpt2-xl")
# model = tl.HookedTransformer.from_pretrained("bloom-560m")
# model = tl.HookedTransformer.from_pretrained("bloom-3b")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
cache_folder = repo_path_to_abs_path("datasets/activation_cache")

# %% kaikki french dictionary - learn W_E (embedding matrix) rotation

# file_path = (
#     f"{datasets_folder}/kaikki-french-dictionary-single-word-pairs-no-hyphen.json"
# )
# with open(file_path, "r") as file:
#     fr_en_pairs_file = json.load(file)

# # 38597 english-french pairs in total
# all_en_fr_pairs = [[pair["English"], pair["French"]] for pair in fr_en_pairs_file]
with open(
    get_dataset_path("muse_en_fr_filtered"),
    "r",
    encoding="utf-8",
) as file:
    fr_en_pairs_file = json.load(file)

all_en_fr_pairs = fr_en_pairs_file

# %%timeit -n 3 -r 1
random.seed(1)
random.shuffle(all_en_fr_pairs)
split_index = int(len(all_en_fr_pairs) * 0.97)
train_en_fr_pairs = all_en_fr_pairs[:split_index]
test_en_fr_pairs = all_en_fr_pairs[split_index:]

train_en_toks, train_fr_toks, _, _ = tokenize_word_pairs(model, train_en_fr_pairs)
test_en_toks, test_fr_toks, _, _ = tokenize_word_pairs(model, test_en_fr_pairs)

# train_en_embeds = (
#     model.embed.W_E[train_en_toks].detach().clone()
# )  # shape[batch, pos, d_model]
# test_en_embeds = (
#     model.embed.W_E[test_en_toks].detach().clone()
# )  # shape[batch, pos, d_model]
# train_fr_embeds = (
#     model.embed.W_E[train_fr_toks].detach().clone()
# )  # shape[batch, pos, d_model]
# test_fr_embeds = (
#     model.embed.W_E[test_fr_toks].detach().clone()
# )  # shape[batch, pos, d_model]

train_en_embeds = t.nn.functional.layer_norm(
    model.embed.W_E[train_en_toks].detach().clone(), [model.cfg.d_model]
)
train_fr_embeds = t.nn.functional.layer_norm(
    model.embed.W_E[train_fr_toks].detach().clone(), [model.cfg.d_model]
)
test_en_embeds = t.nn.functional.layer_norm(
    model.embed.W_E[test_en_toks].detach().clone(), [model.cfg.d_model]
)
test_fr_embeds = t.nn.functional.layer_norm(
    model.embed.W_E[test_fr_toks].detach().clone(), [model.cfg.d_model]
)

train_dataset = TensorDataset(train_en_embeds, train_fr_embeds)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

print(test_en_embeds.shape)
print(test_fr_embeds.shape)
test_dataset = TensorDataset(test_en_embeds, test_fr_embeds)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

X = train_en_embeds.detach().clone().squeeze()
Y = train_fr_embeds.detach().clone().squeeze()
C = t.matmul(X.T, Y)
U, _, V = t.svd(C)


class ManualMatMulModule(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = nn.Parameter(transform, requires_grad=False)

    def forward(self, x):
        return t.matmul(x, self.transform)


W = t.matmul(U, V.t())
transform = ManualMatMulModule(
    W
)  # this is the orthogonal matrix as the product of U and V

accuracy = evaluate_accuracy(
    model,
    test_loader,
    transform,
    exact_match=False,
    print_results=True,
)
print(f"Correct Percentage: {accuracy * 100:.2f}%")
print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))

# %%
translation_file = repo_path_to_abs_path("datasets/muse/4_azure_validation/en-fr.json")

mark_translation(
    model=model,
    transformation=transform,
    test_loader=test_loader,
    allowed_translations_path=translation_file,
    print_results=True,
)


# %% rotate then translate
def align_embeddings_with_translation(X, Y):
    # Step 1: Compute the means of X and Y
    X_mean = t.mean(X, dim=0, keepdim=True)
    Y_mean = t.mean(Y, dim=0, keepdim=True)

    # Step 2: Center X and Y by subtracting their respective means
    X_centered = X - X_mean
    Y_centered = Y - Y_mean

    # Step 3: Align the centered embeddings using SVD (as before)
    C = t.matmul(X_centered.T, Y_centered)
    U, S, V = t.svd(C)
    W = t.matmul(U, V.t())

    # Apply the rotation matrix W to X_centered to align it with Y
    X_rotated = t.matmul(X_centered, W)

    # Step 4: Compute the translation vector
    # Since X was centered, add the mean of Y to translate
    b = Y_mean - t.mean(X_rotated, dim=0, keepdim=True)

    return W, b


class ManualRotateTranslateModule(nn.Module):
    def __init__(self, rotation, translation):
        super().__init__()
        self.rotation = nn.Parameter(rotation, requires_grad=False)
        self.translation = nn.Parameter(translation, requires_grad=False)

    def forward(self, x):
        x_rotated = t.matmul(x, self.rotation)
        x_translated = t.add(x_rotated, self.translation)
        return x_translated


W, b = align_embeddings_with_translation(X, Y)
transform = ManualRotateTranslateModule(W, b)

accuracy = evaluate_accuracy(
    model,
    test_loader,
    transform,
    exact_match=False,
    print_results=True,
)
print(f"Correct Percentage: {accuracy * 100:.2f}%")
print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))

# %% analytical solution for the translation

T = t.mean(Y - X, dim=0)


class ManualMatAddModule(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = nn.Parameter(transform, requires_grad=False)

    def forward(self, x):
        return t.add(x, self.transform)


transform = ManualMatAddModule(T)

accuracy = evaluate_accuracy(
    model,
    test_loader,
    transform,
    exact_match=False,
    print_results=True,
)
print(f"Correct Percentage: {accuracy * 100:.2f}%")
print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))

# %%
