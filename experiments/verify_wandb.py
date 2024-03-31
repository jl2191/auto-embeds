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
    train_transform,
)
from auto_embeds.utils.misc import repo_path_to_abs_path
from auto_embeds.verify import (
    calc_tgt_is_closest_embed,
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
# file_path = get_dataset_path("cc_cedict_zh_en_extracted")
# with open(file_path, "r") as file:
#     word_pairs = json.load(file)

# all_word_pairs = filter_word_pairs(
#     model,
#     word_pairs,
#     discard_if_same=True,
#     min_length=2,
#     # capture_diff_case=True,
#     # capture_space=True,
#     capture_no_space=True,
#     # print_pairs=True,
#     print_number=True,
#     # max_token_id=100_000,
#     # most_common_english=True,
#     # most_common_french=True,
#     # acceptable_overlap=0.8,
# )

file_path = get_dataset_path("wikdict_en_fr_extracted")
with open(file_path, "r") as file:
    word_pairs = json.load(file)

all_word_pairs = filter_word_pairs(
    model,
    word_pairs,
    discard_if_same=True,
    min_length=5,
    # capture_diff_case=True,
    capture_space=True,
    # capture_no_space=True,
    # print_pairs=True,
    print_number=True,
    verbose_count=True,
    # max_token_id=100_000,
    # most_common_english=True,
    # most_common_french=True,
    # acceptable_overlap=0.8,
)

# %%
# we can be more confident in our results if we randomly select a word, and calculate
# the cosine similarity between this word and all the rest and take the top 300 to be
# the test set

# Ensure model.tokenizer is not None and is callable to satisfy linter
if model.tokenizer is None or not callable(model.tokenizer):
    raise ValueError("model.tokenizer is not set or not callable")

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
translation_file = get_dataset_path("wikdict_en_fr_azure_validation")
# translation_file = get_dataset_path("cc_cedict_zh_en_azure_validation")

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

    print(f"{transformation_name}:")

    run = wandb.init(
        project="language-transformations",
        notes="all_rotations",
        tags=[
            "test",
            "all_transform",
            # "actual",
            "en-fr",
            # "zh-en",
        ],
    )

    config = wandb.config

    config.update(
        {
            "model_name": model.cfg.model_name,
            "transformation": transformation_name,
            "layernorm": True,
            "no_processing": True,
            "train_batch_size": len(train_loader),
            "test_batch_size": len(test_loader),
            "lr": 8e-5,
            "n_epochs": 50,
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

    test_cos_sim_diff = test_cos_sim_difference(verify_results_dict)

    wandb.log(
        {
            "mark_translation_acc": mark_translation_acc,
            "cos_sims_trend_plot": cos_sims_trend_plot,
            "test_cos_sim_diff": test_cos_sim_diff,
        }
    )

    wandb.finish()

# %%
# verify_results_table = verify_transform_table_from_dict(verify_results_dict)

# %%
# cos_sims_trend_plot

# %%
# verify_results_dict = verify_transform(
#     model=model,
#     transformation=transform,
#     test_loader=test_loader,
# )
# test_cos_sim_diff = test_cos_sim_difference(verify_results_dict)
# print(test_cos_sim_diff)

from auto_embeds.embed_utils import calc_canonical_angles

a_same = t.tensor([[1.0, 0.0], [0.0, 1.0]])
b_same = t.tensor([[1.0, 0.0], [0.0, 1.0]])

a_45 = t.tensor([[1.0, 0.0], [0.0, 1.0]])
b_45 = t.tensor(
    [
        [np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
        [-np.sqrt(2.0) / 2.0, np.sqrt(2.0) / 2.0],
    ],
    dtype=t.float32,
)
canonical_angles_same = calc_canonical_angles(a_same, b_same)
canonical_angles_45 = calc_canonical_angles(a_45, b_45)

print(canonical_angles_same)
print(canonical_angles_45)

verify_learned = prepare_verify_analysis(
    model=model,
    all_word_pairs=all_word_pairs,
    random_seed=1,
    keep_other_pair=False,
)
canonical_angles = calc_canonical_angles(
    verify_learned.src.other.embeds, verify_learned.tgt.other.embeds
)
all_embeds = t.cat((verify_learned.src.other.embeds, verify_learned.tgt.other.embeds))

verify_learned_ln = prepare_verify_analysis(
    model=model,
    all_word_pairs=all_word_pairs,
    random_seed=1,
    keep_other_pair=False,
    apply_ln=True,
)

all_embeds_ln = t.cat(
    (verify_learned_ln.src.other.embeds, verify_learned_ln.tgt.other.embeds)
)

canonical_angles_ln = calc_canonical_angles(
    verify_learned_ln.src.other.embeds, verify_learned_ln.tgt.other.embeds
)

print(canonical_angles)
print(canonical_angles_ln)

print(max(canonical_angles))
print(max(canonical_angles_ln))

# %%
import plotly.graph_objects as go

# Prepare data for histogram plotting with more bins
canonical_angles_hist = go.Histogram(
    x=canonical_angles.cpu().numpy(),
    name="Canonical Angles without LayerNorm",
    opacity=0.75,
    nbinsx=50,  # Increased number of bins
)

canonical_angles_ln_hist = go.Histogram(
    x=canonical_angles_ln.cpu().numpy(),
    name="Canonical Angles with LayerNorm",
    opacity=0.75,
    nbinsx=50,  # Increased number of bins
)

data = [canonical_angles_hist, canonical_angles_ln_hist]

# Create the histogram plot with more detailed binning
fig = go.Figure(data=data)
fig.update_layout(
    title="Distribution of Canonical Angles",
    xaxis_title="Canonical Angle",
    yaxis_title="Frequency",
    legend_title="Type",
    barmode="overlay",
)
fig.show()

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
