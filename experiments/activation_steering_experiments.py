# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch as t
import transformer_lens as tl

import wandb
from auto_steer.data import create_data_loaders
from auto_steer.steering_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    initialize_transform_and_optim,
    tokenize_texts,
    train_transform,
)
from auto_steer.utils.misc import repo_path_to_abs_path

t.manual_seed(42)
# %%
# %load_ext autoreload
# %autoreload 2
# %%
# %load_ext line_profiler
# %%timeit -n 3 -r 1
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

# %% -----------------------------------------------------------------------------------
# %% kaikki french dictionary - learn W_E (embedding matrix) rotation
file_path = (
    f"{datasets_folder}/kaikki-french-dictionary-single-word-pairs-no-hyphen.json"
)
with open(file_path, "r") as file:
    fr_en_pairs_file = json.load(file)

# 38597 english-french pairs in total
en_fr_pairs = [[pair["English"], pair["French"]] for pair in fr_en_pairs_file]

# %%
# en_toks, en_attn_mask = tokenize_texts(model, en_strs_list)
# fr_toks, fr_attn_mask = tokenize_texts(model, fr_strs_list)

# en (fr_toks, fr_attn_mask)] = tokenize_texts(
#     model, en_strs_list, fr_strs_list, padding_side="left", pad_to_same_length=True
# )

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

# single_tokens_only, discard_if_same, min_length=3, capture_space:
# 1656 tokens pairs

# single_tokens_only, discard_if_same, min_length=3, capture_diff_case, capture_space:
# 3502 token pairs
# bloom-560m, batch size 512, 200 epochs, performance 65%

# single_tokens_only, discard_if_same, min_length=3, capture_diff_case, capture_space,
# capture_no_space:
# 7772 token pairs
# bloom-560m, batch size 512, 200 epochs, performance 65%


# %%
en_embeds = model.embed.W_E[en_toks].detach().clone()  # shape[batch, seq_len, d_model]
fr_embeds = model.embed.W_E[fr_toks].detach().clone()  # shape[batch, seq_len, d_model]

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
filename_base = "bloom-560m-kaikki"

transform, optim = initialize_transform_and_optim(
    d_model,
    transformation="linear_map",
)
if optim is not None:
    transform, loss_history = train_transform(
        model,
        train_loader,
        transform,
        optim,
        25,
        device,
    )
else:
    print("Optimizer is None, cannot proceed with training.")
# %%
accuracy = evaluate_accuracy(
    model,
    test_loader,
    transform,
    exact_match=False,
    print_results=True,
)
print(f"Correct Percentage: {accuracy * 100:.2f}%")
# %%
print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))

# %% -----------------------------------------------------------------------------------
# # %% gather activations
# en_acts, fr_acts = run_and_gather_acts(model, train_loader, layers=[0, 1])
# # %% save activations
# save_acts(cache_folder, filename_base, en_acts, fr_acts)
# # %% load activations
# en_resids = t.load(f"{cache_folder}/bloom-560m-kaikki-en-layers-[0, 1].pt")
# fr_resids = t.load(f"{cache_folder}/bloom-560m-kaikki-fr-layers-[0, 1].pt")
# # %%
# en_resids = {layer: t.cat(en_resids[layer], dim=0) for layer in en_resids}
# fr_resids = {layer: t.cat(fr_resids[layer], dim=0) for layer in fr_resids}

# layer_idx = 1
# gen_length = 20

# # %% train en-fr rotation
# file_path = (
#     f"{datasets_folder}/kaikki-french-dictionary-single-word-pairs-no-hyphen.json"
# )
# with open(file_path, "r") as file:
#     fr_en_pairs = json.load(file)

# en_file = f"{datasets_folder}/europarl/europarl-v7.fr-en.en"
# fr_file = f"{datasets_folder}/europarl/europarl-v7.fr-en.fr"
# test_en_strs = load_test_strings(en_file, skip_lines=1000)
# test_fr_strs = load_test_strings(fr_file, skip_lines=1000)

# en_strs_list, fr_strs_list = zip(
#     *[(pair["English"], pair["French"]) for pair in fr_en_pairs]
# )

# generator = t.Generator().manual_seed(42)


# def split_dataset(dataset, prop=0.1, print_sizes=False):
#     original_size = len(dataset)
#     split_size = int(original_size * prop)
#     remaining_size = original_size - split_size
#     dataset, _ = random_split(
#         dataset, [split_size, remaining_size], generator=generator
#     )
#     if print_sizes:
#         print(f"Original size: {original_size}, Final size: {split_size}")
#     return dataset


# test_loader, train_loader = create_data_loaders(
#     split_dataset(en_resids[layer_idx], prop=0.1, print_sizes=True),
#     split_dataset(fr_resids[layer_idx], prop=0.1, print_sizes=True),
#     train_ratio=0.99,
#     batch_size=128,
#     match_dims=True,
# )
# # %%
# initial_rotation, optim = initialize_transform_and_optim(
#     d_model, transformation="rotation", device=device
# )

# learned_rotation = train_transform(
#     model, train_loader, test_loader, initial_rotation, optim, 1, device
# )

# # %%
# # print("Test Accuracy:", calc_cos_sim_acc(test_loader, learned_rotation))

# # fr_to_en_mean_vec = mean_vec(en_resids[layer_idx], fr_resids[layer_idx])

# # perform_translation_tests(
# #     model, test_en_strs, test_fr_strs, layer_idx, gen_length, learned_rotation
# # )
