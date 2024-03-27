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
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.embed_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    filter_word_pairs,
    initialize_loss,
    initialize_transform_and_optim,
    mark_correct,
    tokenize_word_pairs,
    train_transform,
)
from auto_embeds.utils.misc import repo_path_to_abs_path

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
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
model_caches_folder = repo_path_to_abs_path("datasets/model_caches")
token_caches_folder = repo_path_to_abs_path("datasets/token_caches")

# %% -----------------------------------------------------------------------------------
# file_path = f"{datasets_folder}/muse/3_filtered/en-fr.json"
# file_path = f"{datasets_folder}/wikdict/3_filtered/eng-fra.json"
file_path = f"{datasets_folder}/wikdict/2_extracted/eng-fra.json"
with open(file_path, "r") as file:
    word_pairs = json.load(file)

random.seed(1)
random.shuffle(word_pairs)
split_index = int(len(word_pairs) * 0.97)

train_en_fr_pairs = word_pairs[:split_index]
test_en_fr_pairs = word_pairs[split_index:]

# split_index = int(len(word_pairs) * 0.8)

# test_split_start = int(len(word_pairs) * 0.4)
# test_split_end = int(len(word_pairs) * 0.5)

# test_en_fr_pairs = word_pairs[test_split_start:test_split_end]
# train_en_fr_pairs = [word_pair for word_pair in word_pairs if word_pair not in test_en_fr_pairs]


# %% filtering
all_word_pairs = filter_word_pairs(
    model,
    word_pairs,
    discard_if_same=True,
    # min_length=6,
    # capture_diff_case=True,
    capture_space=True,
    # capture_no_space=True,
    print_pairs=True,
    print_number=True,
    # max_token_id=100_000,
    most_common_english=True,
    most_common_french=True,
    # acceptable_overlap=0.8,
)

random.seed(1)
random.shuffle(all_word_pairs)
split_index = int(len(all_word_pairs) * 0.97)
train_en_fr_pairs = all_word_pairs[:split_index]
test_en_fr_pairs = all_word_pairs[split_index:]
# %%
train_en_toks, train_fr_toks, train_en_mask, train_fr_mask = tokenize_word_pairs(
    model, train_en_fr_pairs
)
test_en_toks, test_fr_toks, test_en_mask, test_fr_mask = tokenize_word_pairs(
    model, test_en_fr_pairs
)

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

print(train_en_embeds.shape)
train_dataset = TensorDataset(train_en_embeds, train_fr_embeds)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(test_en_embeds, test_fr_embeds)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
# %%
# run = wandb.init(
#     project="single_token_tests",
# )
# %%

translation_file = repo_path_to_abs_path(
    # "datasets/muse/4_azure_validation/en-fr.json"
    "datasets/wikdict/4_azure_validation/eng-fra.json"
    # "datasets/wikdict-azure-en-fr.json"
)

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
            plot_fig=False,
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
        print_results=True,
        print_top_preds=False,
    )
    print(f"{transformation_name}:")
    print(f"Correct Percentage: {accuracy * 100:.2f}%")
    print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))

    mark_correct(
        model=model,
        transformation=transform,
        test_loader=test_loader,
        acceptable_translations_path=translation_file,
        print_results=True,
    )

# %%
translation_file = repo_path_to_abs_path(
    # "datasets/muse/4_azure_validation/en-fr.json"
    "datasets/wikdict/4_azure_validation/eng-fra.json"
    # "datasets/wikdict-azure-en-fr.json"
)
# Load acceptable translations from JSON file
with open(translation_file, "r") as file:
    acceptable_translations = json.load(file)

# Convert list of acceptable translations to a more accessible format
translations_list = []
for item in acceptable_translations:
    source = item["normalizedSource"]
    top_translation = next(
        (
            trans["normalizedTarget"]
            for trans in item["translations"]
            if trans["normalizedTarget"] is not None
        ),
        None,
    )
    if top_translation:
        translations_list.append([source, top_translation])

print(len(translations_list))

wikdict_azure_save_path = repo_path_to_abs_path("datasets/wikdict-azure-en-fr.json")
with open(wikdict_azure_save_path, "w") as f:
    json.dump(translations_list, f)
