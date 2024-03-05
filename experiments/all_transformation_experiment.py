#%%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch as t
import transformer_lens as tl
import wandb

from auto_steer.data import create_data_loaders
from auto_steer.steering_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    initialize_loss,
    initialize_transform_and_optim,
    tokenize_texts,
    train_transform,
)
from auto_steer.utils.misc import repo_path_to_abs_path

#%%
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)
# %%
# %%
# %load_ext autoreload
# %load_ext line_profiler
# %autoreload 2
# %%prun -s cumulative
# %lprun -f get_most_similar_embeddings
# %% model setup
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
cache_folder = repo_path_to_abs_path("datasets/activation_cache")

# %% kaikki french dictionary - learn W_E (embedding matrix) rotation

file_path = (
    f"{datasets_folder}/kaikki-french-dictionary-single-word-pairs-no-hyphen.json"
)
with open(file_path, "r") as file:
    fr_en_pairs_file = json.load(file)

# 38597 english-french pairs in total
en_fr_pairs = [[pair["English"], pair["French"]] for pair in fr_en_pairs_file]

# %%
# %%timeit -n 3 -r 1
en_toks, en_attn_mask, fr_toks, fr_attn_mask = tokenize_texts(
    model,
    en_fr_pairs,
    padding_side="left",
    single_tokens_only=True,
    discard_if_same=True,
    min_length=3,
    capture_diff_case=True,
    capture_space=True,
    capture_no_space=True,
)
# %%
# en_embeds = model.embed.W_E[en_toks].detach().clone() # shape[batch, seq_len, d_model]
# fr_embeds = model.embed.W_E[fr_toks].detach().clone() # shape[batch, seq_len, d_model]
en_embeds = t.nn.functional.layer_norm(
    model.embed.W_E[en_toks].detach().clone(), [d_model]
)
fr_embeds = t.nn.functional.layer_norm(
    model.embed.W_E[fr_toks].detach().clone(), [d_model]
)
train_loader, test_loader = create_data_loaders(
    en_embeds,
    fr_embeds,
    batch_size=512,
    train_ratio=0.97,
)
# %%
run = wandb.init(
    project="single_token_tests",
)

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
            n_epochs=100,
            wandb=wandb,
        )
    else:
        print(f"nothing trained for {transformation_name}")
    print(f"{transformation_name}:")
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

# %%
