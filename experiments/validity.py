# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random

import numpy as np
import torch as t
import transformer_lens as tl
from IPython.core.getipython import get_ipython
from rich.console import Console
from rich.jupyter import print as richprint
from rich.layout import Layout
from rich.table import Table
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.data import get_dataset_path
from auto_embeds.embed_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    filter_word_pairs,
    get_most_similar_embeddings,
    initialize_loss,
    initialize_transform_and_optim,
    mark_correct,
    print_most_similar_embeddings_dict,
    tokenize_word_pairs,
    train_transform,
)
from auto_embeds.utils.misc import calculate_gradient_color, repo_path_to_abs_path

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
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
model_caches_folder = repo_path_to_abs_path("datasets/model_caches")
token_caches_folder = repo_path_to_abs_path("datasets/token_caches")

# %% filtering
# file_path = get_dataset_path("muse_zh_en_extracted_train")
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
#     print_pairs=True,
#     print_number=True,
#     # max_token_id=100_000,
#     most_common_english=True,
#     most_common_french=True,
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
    print_pairs=True,
    print_number=True,
    verbose_count=True,
    # max_token_id=100_000,
    most_common_english=True,
    most_common_french=True,
    # acceptable_overlap=0.8,
)

# %%
# we can be more confident in our results if we randomly select a word, and calculate
# the cosine similarity between this word and all the rest and take the top 300 to be
# the test set

# Ensure model.tokenizer is not None and is callable to satisfy linter
if model.tokenizer is None or not callable(model.tokenizer):
    raise ValueError("model.tokenizer is not set or not callable")

random.seed(2)
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
# these should have shape[d_model]

# tokenize all the other word pairs
src_toks, tgt_toks, src_mask, tgt_mask = tokenize_word_pairs(model, word_pairs)
other_toks = t.cat((src_toks, tgt_toks), dim=0)

# embed all the other word pairs as well
src_embeds = model.embed.W_E[src_toks].detach().clone().squeeze()
tgt_embeds = model.embed.W_E[tgt_toks].detach().clone().squeeze()
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


def generate_top_word_pairs_table(
    model,
    random_word,
    other_toks,
    top_indices,
    distances,
    cos_sims,
    sort_by="cos_sim",
    display_limit=50,
):
    if sort_by == "cos_sim":
        title_sort_by = "Cosine Similarity"
    elif sort_by == "euc_dist":
        title_sort_by = "Euclidean Distance"
    else:
        raise ValueError("Supported sort functions are 'cos_sim' and 'euc_dist'")

    table = Table(
        show_header=True,
        title=f"Closest tokens in {model.cfg.model_name} to "
        f"[plum3 on grey23]{random_word}[/plum3 on grey23] sorted by {title_sort_by}",
    )
    table.add_column("Rank")
    table.add_column("Token")
    table.add_column("Cos Sim")
    table.add_column("Euc Dist")

    # Determine the range of cosine similarities and euclidean distances for gradient
    # calculation

    cos_sim_values = [cos_sims[index].item() for index in top_indices]
    euc_dist_values = [distances[index].item() for index in top_indices]
    cos_sim_min, cos_sim_max = min(cos_sim_values), max(cos_sim_values)
    euc_dist_min, euc_dist_max = min(euc_dist_values), max(euc_dist_values)

    if sort_by == "cos_sim":
        sorted_indices = sorted(
            top_indices, key=lambda idx: cos_sims[idx].item(), reverse=True
        )
    elif sort_by == "euc_dist":
        sorted_indices = sorted(top_indices, key=lambda idx: distances[idx].item())
    else:
        raise ValueError("Supported sort functions are 'cos_sim' and 'euc_dist'")

    display_start = display_limit // 2
    for rank, index in enumerate(sorted_indices, start=1):
        if rank == display_start + 1:
            table.add_row("...", "...", "...", "...")
        elif rank > display_start and rank <= top_indices.shape[0] - display_start:
            continue
        else:
            token = other_toks[index]
            word = model.tokenizer.decode(token)
            cos_sim = cos_sims[index].item()
            euc_dist = distances[index].item()

            # Calculate gradient colors based on the value's magnitude
            cos_sim_color = calculate_gradient_color(cos_sim, cos_sim_min, cos_sim_max)
            euc_dist_color = calculate_gradient_color(
                euc_dist, euc_dist_min, euc_dist_max, reverse=True
            )

            word_styled = f"[plum3 on grey23]{word}[/plum3 on grey23]"
            cos_sim_styled = f"[{cos_sim_color}]{cos_sim:.4f}[/{cos_sim_color}]"
            euc_dist_styled = f"[{euc_dist_color}]{euc_dist:.4f}[/{euc_dist_color}]"
            table.add_row(str(rank), word_styled, cos_sim_styled, euc_dist_styled)

    return table


table_cos_sim_src = generate_top_word_pairs_table(
    model,
    random_word_src,
    other_toks,
    tgt_and_other_top_200_cos_sims.indices,
    src_euc_dists,
    random_word_src_and_other_embeds_cos_sims,
    sort_by="cos_sim",
    display_limit=30,
)
table_euc_dist_src = generate_top_word_pairs_table(
    model,
    random_word_src,
    other_toks,
    src_top_200_euc_dists.indices,
    src_euc_dists,
    random_word_src_and_other_embeds_cos_sims,
    sort_by="euc_dist",
    display_limit=30,
)
table_cos_sim_tgt = generate_top_word_pairs_table(
    model,
    random_word_tgt,
    other_toks,
    tgt_and_other_top_200_cos_sims.indices,
    tgt_euc_dists,
    random_word_tgt_and_other_embeds_cos_sims,
    sort_by="cos_sim",
    display_limit=30,
)
table_euc_dist_tgt = generate_top_word_pairs_table(
    model,
    random_word_tgt,
    other_toks,
    tgt_top_200_euc_dists.indices,
    tgt_euc_dists,
    random_word_tgt_and_other_embeds_cos_sims,
    sort_by="euc_dist",
    display_limit=30,
)

# print the tables
# Create a layout with four columns for a 2x2 grid
console = Console()
layout = Layout()
# Adjusting the ratio to create more gap between the top and bottom grids
layout.split(Layout(name="top", ratio=7), Layout(name="bottom", ratio=10))
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
layout["right_bottom"].update(table_euc_dist_tgt)  # Corrected variable name

# Print the layout to the console with minimal padding between columns
console.print(layout)


# %%
def test_hypothesis(model, src_toks, tgt_toks, other_toks, other_embeds):
    other_toks = other_toks.to(device)
    src_toks = src_toks.to(device)
    tgt_toks = tgt_toks.to(device)
    correct_count_top_5 = 0
    correct_count_top_1 = 0
    total_count = len(src_toks)
    print(src_toks.shape)

    results = []

    for i, (src_tok, correct_tgt_tok) in enumerate(zip(src_toks, tgt_toks)):
        # Embed the source token
        src_embed = model.embed.W_E[src_tok].detach().clone().squeeze(0)
        # shape [d_model]

        # Exclude the current source token from other_toks and other_embeds to avoid
        # self-matching
        valid_indices = (other_toks != src_tok).squeeze(-1)
        # the mask valid_indices should be of shape [batch]

        valid_other_toks = other_toks[valid_indices]
        valid_other_embeds = other_embeds[valid_indices]

        # The cosine similarities between the source token and all the other tokens
        cos_sims = t.cosine_similarity(src_embed, valid_other_embeds, dim=-1)

        # Get the indices of the top 5 target tokens with the highest cosine similarity
        # that is valid (i.e. not the same as the source language token)
        top_5_indices = t.topk(cos_sims, 5).indices

        # Check if the correct target token is among the top 5
        is_correct_in_top_5 = correct_tgt_tok in valid_other_toks[top_5_indices]
        is_correct_in_top_1 = correct_tgt_tok == valid_other_toks[top_5_indices[0]]

        if is_correct_in_top_5:
            correct_count_top_5 += 1

        if is_correct_in_top_1:
            correct_count_top_1 += 1
            status = "Correct ✅ "
        else:
            status = "Incorrect ❌"

        src_tok_str = model.tokenizer.decode(src_tok)
        correct_tgt_tok_str = model.tokenizer.decode(correct_tgt_tok)
        top_5_tokens = [
            model.tokenizer.decode(valid_other_toks[index], skip_special_tokens=True)
            for index in top_5_indices
        ]
        cos_sim_values = [cos_sims[index].item() for index in top_5_indices]
        top_5_details = "\n".join(
            f"  {rank}. {token} (Cosine Similarity: {cos_sim:.4f})"
            for rank, (token, cos_sim) in enumerate(
                zip(top_5_tokens, cos_sim_values), start=1
            )
        )
        result = (
            f"{i+1} {status}\n"
            f"Source Token: '{src_tok_str}'\n"
            f"Target Token: '{correct_tgt_tok_str}'\n"
            f"Top 5 tokens with highest cosine similarity:\n"
            f"{top_5_details}\n\n"
        )
        results.append(result)

    # Calculate the percentage where the hypothesis holds true for top 5 and top 1
    percentage_correct_top_1 = (correct_count_top_1 / total_count) * 100
    percentage_correct_top_5 = (correct_count_top_5 / total_count) * 100
    results.append(
        f"Percentage where the hypothesis is true (correct translation is top 1):"
        f"{percentage_correct_top_1:.2f}%\n"
        f"Percentage where the hypothesis is true (correct translation in top 5):"
        f"{percentage_correct_top_5:.2f}%"
    )

    return results


# Assuming src_toks and tgt_toks are tensors containing the token IDs for source and
# target languages
assert src_toks.shape == tgt_toks.shape
assert other_toks.shape[0] == other_embeds.shape[0]

results = test_hypothesis(model, src_toks, tgt_toks, other_toks, other_embeds)
for result in results:
    print(result)

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
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# %%
# run = wandb.init(
#     project="single_token_tests",
# )
# %%

translation_file = get_dataset_path("wikdict_en_fr_azure_validation")

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
        optim_kwargs={"lr": 1e-4, "weight_decay": 1e-5},
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
            plot_fig=True,
            # save_fig=True,
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
        print_results=True,
        print_top_preds=False,
    )
    print(f"{transformation_name}:")
    print(f"Correct Percentage: {accuracy * 100:.2f}%")
    print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))

    test_acc_2 = mark_correct(
        model=model,
        transformation=transform,
        test_loader=test_loader,
        allowed_translations_path=translation_file,
        print_results=True,
    )

    print(f"Test Accuracy 2.0: {test_acc_2}")
