# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import torch as t
import transformer_lens as tl
from IPython.core.getipython import get_ipython
from rich.console import Console

# from rich.jupyter import print as richprint
from rich.layout import Layout
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.data import get_dataset_path
from auto_embeds.embed_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    filter_word_pairs,
    initialize_loss,
    initialize_transform_and_optim,
    mark_translation,
    train_transform,
)
from auto_embeds.utils.misc import repo_path_to_abs_path
from auto_embeds.verify import (
    calc_tgt_is_closest_embed,
    generate_top_word_pairs_table,
    plot_cosine_similarity_trend,
    prepare_verify_analysis,
    test_cos_sim_difference,
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
    # get_ipython().run_line_magic("load_ext", "rich")  # type: ignore
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
    print_pairs=True,
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

verify_analysis = prepare_verify_analysis(
    model=model, all_word_pairs=all_word_pairs, random_seed=1
)
# %%
# generate and display the top word pairs based on cosine similarity and euclidean
# distance for both source and target words.
# %%
table_cos_sim_src = generate_top_word_pairs_table(
    model,
    verify_analysis.src,
    sort_by="cos_sim",
    display_limit=30,
    exclude_identical=True,
)
table_euc_dist_src = generate_top_word_pairs_table(
    model,
    verify_analysis.src,
    sort_by="euc_dist",
    display_limit=30,
    exclude_identical=True,
)
table_cos_sim_tgt = generate_top_word_pairs_table(
    model,
    verify_analysis.tgt,
    sort_by="cos_sim",
    display_limit=30,
    exclude_identical=True,
)
table_euc_dist_tgt = generate_top_word_pairs_table(
    model,
    verify_analysis.tgt,
    sort_by="euc_dist",
    display_limit=30,
    exclude_identical=True,
)

# print the tables
# Create a layout with four columns for a 2x2 grid
console = Console()
layout = Layout()
# Adjusting the ratio to create more gap between the top and bottom grids
layout.split(Layout(name="top", ratio=1), Layout(name="bottom", ratio=1))
layout["top"].split_row(
    Layout(name="left_top", ratio=1), Layout(name="right_top", ratio=1)
)
layout["bottom"].split_row(
    Layout(name="left_bottom", ratio=1), Layout(name="right_bottom", ratio=1)
)

# Assign tables to each quadrant
layout["left_top"].update(table_cos_sim_src)
layout["right_top"].update(table_euc_dist_src)
layout["left_bottom"].update(table_cos_sim_tgt)
layout["right_bottom"].update(table_euc_dist_tgt)

# Print the layout to the console with minimal padding between columns
console.print(layout)

# %%
tgt_is_closest_embed = calc_tgt_is_closest_embed(
    model,
    all_word_pairs,
)
print(tgt_is_closest_embed["summary"])

for result in tgt_is_closest_embed["details"][:20]:
    print(result)

# %%
verify_learned = prepare_verify_analysis(
    model=model,
    all_word_pairs=all_word_pairs,
    random_seed=1,
    keep_other_pair=True,
)
assert verify_learned.src.other.toks.shape == verify_learned.tgt.other.toks.shape
assert (
    verify_learned.src.other.toks.shape[0] == verify_learned.tgt.other.embeds.shape[0]
)


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
src_embed = verify_learned.src.selected.embeds
src_embeds = verify_learned.src.other.embeds
tgt_embeds = verify_learned.tgt.other.embeds

src_cos_sims = t.cosine_similarity(src_embed, src_embeds, dim=-1)
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
# run = wandb.init(
#     project="single_token_tests",
# )

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
    # "rotation",
    # "biased_rotation",
    # "uncentered_rotation",
]

for transformation_name in transformation_names:

    transform = None
    optim = None

    transform, optim = initialize_transform_and_optim(
        d_model,
        transformation=transformation_name,
        # optim_kwargs={"lr": 1e-4},
        optim_kwargs={"lr": 8e-5, "weight_decay": 1e-5},
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
            plot_fig=False,
            azure_translations_path=translation_file,
            save_fig=True,
            # wandb=wandb,
        )
    else:
        print(f"nothing trained for {transformation_name}")

    print(f"{transformation_name}:")
    accuracy = evaluate_accuracy(
        model,
        test_loader,
        transform,
        exact_match=False,
        # print_results=True,
        print_top_preds=False,
    )
    print(f"{transformation_name}:")
    print(f"Correct Percentage: {accuracy * 100:.2f}%")
    print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))

    mark_translation_acc = mark_translation(
        model=model,
        transformation=transform,
        test_loader=test_loader,
        azure_translations_path=translation_file,
        # print_results=True,
    )

    print(f"Mark Translation Accuracy: {mark_translation_acc}")

# %%

verify_results_dict = verify_transform(
    model=model,
    transformation=transform,
    test_loader=test_loader,
)

# %%
verify_results_table = verify_transform_table_from_dict(verify_results_dict)

# %%
# verify_results_table

# %%
plot_cosine_similarity_trend(verify_results_dict)

# %%
test_cos_sim_difference(verify_result_dict)

# %%
# what is the effect of layernorm to cos sims and euclidean distance?
