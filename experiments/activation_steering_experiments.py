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
from torch.utils.data import DataLoader, TensorDataset, random_split

# from auto_embeds.data import create_data_loaders
from auto_embeds.embed_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    filter_word_pairs,
    initialize_loss,
    initialize_transform_and_optim,
    run_and_gather_acts,
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
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
model_caches_folder = repo_path_to_abs_path("datasets/model_caches")
token_caches_folder = repo_path_to_abs_path("datasets/token_caches")

model.tokenizer.padding_side = "left"  # type: ignore
# %% -----------------------------------------------------------------------------------

file_path = f"{datasets_folder}/wikdict/2_extracted/eng-fra.json"
with open(file_path, "r") as file:
    wikdict_pairs = json.load(file)

word_pairs = wikdict_pairs
# %%
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
    # capture_no_space=True,
    print_pairs=True,
    print_number=True,
    max_token_id=100_000,
    # most_common_english=True,
    # most_common_french=True,
)

test_word_pairs = filter_word_pairs(
    model,
    test_en_fr_pairs,
    discard_if_same=True,
    min_length=4,
    # capture_diff_case=True,
    capture_space=True,
    # capture_no_space=True,
    # print_pairs=True,
    print_number=True,
    max_token_id=100_000,
    # most_common_english=True,
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
        optim_kwargs={"lr": 2e-4},
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
# %% gather the residuals when europarl dataset is fed into model

layers = [0, 1, 12, 18, 22, 23]

en_europarl_file = f"{datasets_folder}/europarl-v7.fr-en.en"
fr_europarl_file = f"{datasets_folder}/europarl-v7.fr-en.fr"

with open(en_europarl_file, "r") as f:
    en_euro_strings = [
        f.readline()[:-1] + " " + f.readline()[:-1] for _ in range(1000)
    ][1:]
with open(fr_europarl_file, "r") as f:
    fr_euro_strings = [
        f.readline()[:-1] + " " + f.readline()[:-1] for _ in range(1000)
    ][1:]

# padded_sequences = model.tokenizer(
#     tuple(en_euro_strings),
#     tuple(fr_euro_strings),
#     padding="longest",
#     return_tensors="pt",
# )  # type: ignore
# en_tokenized = model.tokenizer(en_euro_strings, padding=True, return_tensors="pt")
# fr_tokenized = model.tokenizer(fr_euro_strings, padding=True, return_tensors="pt")
# en_toks, en_mask = en_tokenized["input_ids"], en_tokenized["attention_mask"]
# fr_toks, fr_mask = fr_tokenized["input_ids"], fr_tokenized["attention_mask"]

combined_texts = en_euro_strings + fr_euro_strings

tokenized = model.tokenizer(
    combined_texts, padding="longest", return_tensors="pt"
)  # type: ignore

num_pairs = tokenized.input_ids.shape[0]
assert num_pairs % 2 == 0
word_each = num_pairs // 2
tokens = tokenized.data["input_ids"]
attn_masks = tokenized.data["attention_mask"]
en_tokens, fr_tokens = tokens[:word_each], tokens[word_each:]
en_mask, fr_mask = attn_masks[:word_each], attn_masks[word_each:]

train_dataset = TensorDataset(en_tokens, fr_tokens, en_mask, fr_mask)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

en_resids, fr_resids = run_and_gather_acts(model, train_loader, layers=layers)
# %% save activations
t.save(
    en_resids, f"{model_caches_folder}/bloom-560m-wikdict-en-resids-layers-{layers}.pt"
)
t.save(
    fr_resids, f"{model_caches_folder}/bloom-560m-wikdict-fr-resids-layers-{layers}.pt"
)
# %% load activations
en_resids = t.load(
    f"{model_caches_folder}/bloom-560m-wikdict-en-resids-layers-{layers}.pt"
)
fr_resids = t.load(
    f"{model_caches_folder}/bloom-560m-wikdict-fr-resids-layers-{layers}.pt"
)

# %%
# en_resids = {layer: t.cat(en_resids[layer], dim=0) for layer in en_resids}
# fr_resids = {layer: t.cat(fr_resids[layer], dim=0) for layer in fr_resids}

# %%
layer = 23
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
    full_resids_dataset = TensorDataset(en_resids[layer], fr_resids[layer])
    train_dataset, test_dataset = random_split(full_resids_dataset, [0.95, 0.05])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
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
            n_epochs=100,
            # wandb=wandb,
        )
    else:
        print(f"nothing trained for {transformation_name}")

    accuracy = evaluate_accuracy(
        model, test_loader, transform, exact_match=False, print_results=True
    )
    print(f"Correct Percentage: {accuracy * 100:.2f}%")

    test_loss = calc_cos_sim_acc(test_loader, transform)

    print(f"{transformation_name}:")
    print(f"layer number: {layer}")
    print("test loss:", calc_cos_sim_acc(test_loader, transform))

# %% -----------------------------------------------------------------------------------
# %%
# perform_steering_tests(
#     model,
#     en_euro_strings,
#     fr_euro_strings,
#     layer_idx=0,
#     gen_length=20,
#     transformation=transform,
#     positions_to_steer="all",
#     num_tests=3,
# )
# %%
