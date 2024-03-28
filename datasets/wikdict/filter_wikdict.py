# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random

import numpy as np
import torch as t
import transformer_lens as tl

from auto_embeds.embed_utils import filter_word_pairs
from auto_embeds.utils.misc import repo_path_to_abs_path

np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

# %% model setup
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
model_caches_folder = repo_path_to_abs_path("datasets/model_caches")
token_caches_folder = repo_path_to_abs_path("datasets/token_caches")

# %% filtering
file_path = f"{datasets_folder}/wikdict/2_extracted/eng-fra.json"
with open(file_path, "r") as file:
    word_pairs = json.load(file)

all_word_pairs = filter_word_pairs(
    model,
    word_pairs,
    discard_if_same=True,
    min_length=3,
    # capture_diff_case=True,
    capture_space=True,
    # capture_no_space=True,
    print_pairs=True,
    print_number=True,
    verbose_count=True,
    # max_token_id=100_000,
    # most_common_english=True,
    # most_common_french=True,
    # acceptable_english_overlap=0.8,
    # acceptable_french_overlap=0.8,
)

# %% saving
# filtered_save_path = repo_path_to_abs_path("datasets/wikdict/3_filtered/eng-fra.json")
# with open(filtered_save_path, "w") as f:
#     json.dump(all_word_pairs, f)

# %% code for quick training if required (commented out by default)

from auto_embeds.embed_utils import tokenize_word_pairs  # noqa: I001
from torch.utils.data import DataLoader, TensorDataset
from auto_embeds.embed_utils import mark_translation
import torch as t
from auto_embeds.data import get_dataset_path
from auto_embeds.embed_utils import (
    initialize_transform_and_optim,
    train_transform,
    initialize_loss,
    evaluate_accuracy,
    calc_cos_sim_acc,
)

random.seed(1)
random.shuffle(all_word_pairs)
split_index = int(len(all_word_pairs) * 0.97)

train_word_pairs = all_word_pairs[:split_index]
test_word_pairs = all_word_pairs[split_index:]

# # extracting the middle 3% as test data
# middle_start_index = int(len(all_word_pairs) * 0.485)
# middle_end_index = int(len(all_word_pairs) * 0.515)

# train_word_pairs = (
#     all_word_pairs[:middle_start_index] + all_word_pairs[middle_end_index:]
# )
# test_word_pairs = all_word_pairs[middle_start_index:middle_end_index]


# translation_file = repo_path_to_abs_path(
#     "datasets/wikdict/4_azure_validation/eng-fra.json"
# )

train_en_toks, train_fr_toks, train_en_mask, train_fr_mask = tokenize_word_pairs(
    model, train_word_pairs
)
test_en_toks, test_fr_toks, test_en_mask, test_fr_mask = tokenize_word_pairs(
    model, test_word_pairs
)

train_en_embeds = model.embed.W_E[train_en_toks].detach().clone()
test_en_embeds = model.embed.W_E[test_en_toks].detach().clone()
train_fr_embeds = model.embed.W_E[train_fr_toks].detach().clone()
test_fr_embeds = model.embed.W_E[test_fr_toks].detach().clone()
# all are of shape[batch, pos, d_model]

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
        # optim_kwargs={"lr": 6e-5},
        optim_kwargs={"lr": 1e-4, "weight_decay": 2e-5},
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

    print(f"{transformation_name}:")
    accuracy = evaluate_accuracy(
        model,
        test_loader,
        transform,
        exact_match=False,
        print_results=True,
        print_top_preds=True,
    )
    print(f"{transformation_name}:")
    print(f"Correct Percentage: {accuracy * 100:.2f}%")
    print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))


# %%
mark = mark_translation(
    model=model,
    transformation=transform,
    test_loader=test_loader,
    azure_translations_path=translation_file,
    print_results=True,
)

print(mark)

# %%

import torch


def pairwise_cosine_similarity(single_embedding, embedding_set):
    """
    Computes the pairwise cosine similarity between a single embedding and a set of many other embeddings.

    Args:
    - single_embedding (torch.Tensor): A single embedding tensor of shape (embedding_dim,).
    - embedding_set (torch.Tensor): A tensor containing a set of embeddings of shape (num_embeddings, embedding_dim).

    Returns:
    - torch.Tensor: A tensor containing the cosine similarity scores between the single embedding and each embedding in the set.
    """
    # Normalize the single embedding and the embedding set along the embedding dimension
    single_embedding_norm = single_embedding / single_embedding.norm(
        dim=0, keepdim=True
    )
    embedding_set_norm = embedding_set / embedding_set.norm(dim=1, keepdim=True)

    # Compute the cosine similarity
    cosine_similarities = torch.mm(
        embedding_set_norm, single_embedding_norm.unsqueeze(1)
    )

    return cosine_similarities.squeeze()
