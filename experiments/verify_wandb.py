# %%
# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import torch as t
import transformer_lens as tl
import wandb
from IPython.core.getipython import get_ipython
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.data import get_dataset_path
from auto_embeds.embed_utils import (
    evaluate_accuracy,
    filter_word_pairs,
    initialize_loss,
    initialize_transform_and_optim,
    mark_translation,
    tokenize_word_pairs,
    train_transform,
)
from auto_embeds.utils.misc import repo_path_to_abs_path
from auto_embeds.verify import (
    calc_tgt_is_closest_embed,
    plot_cosine_similarity_trend,
    test_cosine_similarity_difference,
    verify_transform,
    verify_transform_table_from_dict,
)

ipython = get_ipython()
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)
try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("load_ext", "line_profiler")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass


# %% model setup
# model = tl.HookedTransformer.from_pretrained_no_processing("mistral-7b")
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
# model = tl.HookedTransformer.from_pretrained("bloom-560m")
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
model_caches_folder = repo_path_to_abs_path("datasets/model_caches")
token_caches_folder = repo_path_to_abs_path("datasets/token_caches")

# %% filtering
file_path = get_dataset_path("cc_cedict_zh_en_extracted")
with open(file_path, "r") as file:
    word_pairs = json.load(file)

all_word_pairs = filter_word_pairs(
    model,
    word_pairs,
    discard_if_same=True,
    min_length=2,
    # capture_diff_case=True,
    # capture_space=True,
    capture_no_space=True,
    # print_pairs=True,
    print_number=True,
    # max_token_id=100_000,
    # most_common_english=True,
    # most_common_french=True,
    # acceptable_overlap=0.8,
)

# file_path = get_dataset_path("wikdict_en_fr_extracted")
# with open(file_path, "r") as file:
#     word_pairs = json.load(file)

# all_word_pairs = filter_word_pairs(
#     model,
#     word_pairs,
#     discard_if_same=True,
#     min_length=5,
#     # capture_diff_case=True,
#     capture_space=True,
#     # capture_no_space=True,
#     # print_pairs=True,
#     print_number=True,
#     verbose_count=True,
#     # max_token_id=100_000,
#     # most_common_english=True,
#     # most_common_french=True,
#     # acceptable_overlap=0.8,
# )

# %%
# we can be more confident in our results if we randomly select a word, and calculate
# the cosine similarity between this word and all the rest and take the top 300 to be
# the test set

# Ensure model.tokenizer is not None and is callable to satisfy linter
if model.tokenizer is None or not callable(model.tokenizer):
    raise ValueError("model.tokenizer is not set or not callable")

random.seed(1)
random.shuffle(all_word_pairs)
word_pairs = all_word_pairs
random_word_pair_index = random.randint(0, len(word_pairs) - 1)
random_word_pair = word_pairs.pop(random_word_pair_index)
random_word_src = random_word_pair[0]
random_word_tgt = random_word_pair[1]
print(f"random word pair is {random_word_pair}")

# tokenize random word pair
random_word_src_tok = model.tokenizer(
    random_word_src, return_tensors="pt", add_special_tokens=False
).data["input_ids"]
random_word_tgt_tok = model.tokenizer(
    random_word_tgt, return_tensors="pt", add_special_tokens=False
).data["input_ids"]

# embed random word pair
random_word_src_embed = model.embed.W_E[random_word_src_tok].detach().clone().squeeze()
random_word_tgt_embed = model.embed.W_E[random_word_tgt_tok].detach().clone().squeeze()
random_word_src_embed = t.nn.functional.layer_norm(
    random_word_src_embed, [model.cfg.d_model]
)
random_word_tgt_embed = t.nn.functional.layer_norm(
    random_word_tgt_embed, [model.cfg.d_model]
)
# these should have shape[d_model]

# tokenize all the other word pairs
src_toks, tgt_toks, src_mask, tgt_mask = tokenize_word_pairs(model, word_pairs)
other_toks = t.cat((src_toks, tgt_toks), dim=0)

# embed all the other word pairs as well
src_embeds = model.embed.W_E[src_toks].detach().clone().squeeze()
tgt_embeds = model.embed.W_E[tgt_toks].detach().clone().squeeze()
src_embeds = t.nn.functional.layer_norm(src_embeds, [model.cfg.d_model])
tgt_embeds = t.nn.functional.layer_norm(tgt_embeds, [model.cfg.d_model])
other_embeds = t.cat((src_embeds, tgt_embeds), dim=0)


random_word_src_and_other_embeds_cos_sims = t.cosine_similarity(
    random_word_src_embed, other_embeds, dim=-1
)  # this should return shape[31712]
# random_word_src_embed has shape[1024]
# other_embeds has shape[31712, 1024]
random_word_tgt_and_other_embeds_cos_sims = t.cosine_similarity(
    random_word_tgt_embed, other_embeds, dim=-1
)  # this should return shape[31712]
# Rank the cosine similarities and get the indices of the top 200
src_and_other_top_200_cos_sims = t.topk(
    random_word_src_and_other_embeds_cos_sims, 200, largest=True
)
tgt_and_other_top_200_cos_sims = t.topk(
    random_word_tgt_and_other_embeds_cos_sims, 200, largest=True
)

# Calculate Euclidean distances
src_euc_dists = t.pairwise_distance(
    random_word_src_embed.unsqueeze(0), other_embeds, p=2
)
tgt_euc_dists = t.pairwise_distance(
    random_word_tgt_embed.unsqueeze(0), other_embeds, p=2
)
# Rank the Euclidean distances and get the indices of the top 200
src_top_200_euc_dists = t.topk(src_euc_dists, 200, largest=False)
tgt_top_200_euc_dists = t.topk(tgt_euc_dists, 200, largest=False)

# %%
# Assuming src_toks and tgt_toks are tensors containing the token IDs for source and
# target languages
assert src_toks.shape == tgt_toks.shape
assert other_toks.shape[0] == other_embeds.shape[0]

tgt_is_closest_embed = calc_tgt_is_closest_embed(
    model, src_toks, tgt_toks, other_toks, other_embeds
)
print(tgt_is_closest_embed["summary"])


# %%
# Create test_embeds tensor from top 200 indices
# other_embeds is of shape [batch, d_model] at this point
# and src_top_200_cos_sim_indices is of shape [batch] and we want to select all the
# the embeddings with the top 200 cos sims

# so we have our top 200 indices whose embeddings we want to use as test embeddings.
# however, this so so far is just for the source language. we need to get the
# corresponding target language embeddings as well.

# we index into src_embeds with these indices to get them, but we first need to
# define the indices that we want
src_cos_sims = t.cosine_similarity(random_word_src_embed, src_embeds, dim=-1)
# shape [batch]

src_top_200_cos_sims = t.topk(src_cos_sims, 200, largest=True)
test_indices = src_top_200_cos_sims.indices
# %%
src_embeds_with_top_200_cos_sims = src_embeds[test_indices]
# and it is actually the same indices to get the corresponding target language embeds
tgt_embeds_with_top_200_cos_sims = tgt_embeds[test_indices]

# these are our test embeddings
src_test_embeds = src_embeds_with_top_200_cos_sims.unsqueeze(1)
tgt_test_embeds = tgt_embeds_with_top_200_cos_sims.unsqueeze(1)
# these should now be [batch, pos, d_model]

print(f"source test embeds shape: {src_test_embeds.shape}")
print(f"target test embeds shape: {tgt_test_embeds.shape}")

# to get out train embeddings, we just need to remove the indices we used for the test
all_indices = t.arange(0, src_embeds.shape[0], device=src_embeds.device)
mask = t.ones(src_embeds.shape[0], dtype=t.bool, device=src_embeds.device)
mask[test_indices] = False

src_train_embeds = src_embeds[mask].unsqueeze(1)
tgt_train_embeds = tgt_embeds[mask].unsqueeze(1)
# these should now be [batch, pos, d_model]

print(f"source train embeds shape: {src_train_embeds.shape}")
print(f"target train embeds shape: {tgt_train_embeds.shape}")

train_dataset = TensorDataset(src_train_embeds, tgt_train_embeds)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(src_test_embeds, tgt_test_embeds)
test_loader = DataLoader(test_dataset, batch_size=256)

# %%
# translation_file = get_dataset_path("wikdict_en_fr_azure_validation")
translation_file = get_dataset_path("cc_cedict_zh_en_azure_validation")

transformation_names = [
    # "identity",
    # "translation",
    "linear_map",
    # "biased_linear_map",
    # "uncentered_linear_map",
    # "biased_uncentered_linear_map",
    "rotation",
    # "biased_rotation",
    "uncentered_rotation",
]

for transformation_name in transformation_names:

    print(f"{transformation_name}:")

    run = wandb.init(
        project="language-transformations",
        notes="now hopefully logging plotly plot",
        tags=[
            "test",
            # "all_transform",
            # "actual",
            # "en-fr",
            "zh-en",
        ],
    )

    config = wandb.config

    config.update(
        {
            "model_name": model.cfg.model_name,
            "transformation": transformation_name,
            "layernorm": True,
            "no_processing": True,
            "batch_size": train_loader.batch_size,
            "lr": 8e-5,
            "n_epochs": 100,
            "weight_decay": 1e-5,
        }
    )

    transform = None
    optim = None

    transform, optim = initialize_transform_and_optim(
        d_model,
        transformation=transformation_name,
        # optim_kwargs={"lr": 1e-4},
        optim_kwargs={"lr": config.lr, "weight_decay": config.weight_decay},
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
            n_epochs=config.n_epochs,
            plot_fig=False,
            azure_translations_path=translation_file,
            save_fig=True,
            wandb=wandb,
        )
    else:
        print(f"nothing trained for {transformation_name}")

    accuracy = evaluate_accuracy(
        model,
        test_loader,
        transform,
        exact_match=False,
        print_results=False,
        print_top_preds=False,
    )

    mark_translation_acc = mark_translation(
        model=model,
        transformation=transform,
        test_loader=test_loader,
        azure_translations_path=translation_file,
        print_results=False,
    )

    verify_results_dict = verify_transform(
        model=model,
        transformation=transform,
        test_loader=test_loader,
    )

    cos_sims_trend_plot = plot_cosine_similarity_trend(verify_results_dict)

    wandb.log = {
        "accuracy": accuracy,
        "mark_translation_acc": mark_translation_acc,
        "cos_sims_trend_plot": cos_sims_trend_plot,
    }

    wandb.finish()

# %%
verify_results_table = verify_transform_table_from_dict(verify_results_dict)

# %%
cos_sims_trend_plot

# %%


def test_cosine_similarity_difference(verify_results):
    cos_sims = verify_results["cos_sims"]
    first_25_cos_sims = cos_sims[:25].cpu()
    last_25_cos_sims = cos_sims[-25:].cpu()

    # Perform a two-sample t-test to check if there's a significant difference
    t_stat, p_value = stats.ttest_ind(first_25_cos_sims, last_25_cos_sims)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        print(
            "There is a significant difference between the first 25 and last 25 "
            "cosine similarity values."
        )
    else:
        print(
            "There is no significant difference between the first 25 and last 25 "
            "cosine similarity values."
        )


# Call the function with the verify_results_dict
test_cosine_similarity_difference(verify_results_dict)
