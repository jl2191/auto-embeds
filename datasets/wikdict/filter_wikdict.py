# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch as t
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import transformer_lens as tl

from auto_embeds.embed_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    initialize_transform_and_optim,
    train_transform,
    filter_word_pairs,
    tokenize_word_pairs,
    initialize_loss,
)
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
    # print_pairs=True,
    print_number=True,
    # max_token_id=200_000,
    # most_common_english=True,
    # most_common_french=True,
)

filtered_save_path = repo_path_to_abs_path("datasets/wikdict/3_filtered/eng-fra.json")
with open(filtered_save_path, 'w') as f:
    json.dump(all_word_pairs, f)

# %%
for word_pair in all_word_pairs:
    print(word_pair)
# train_en_toks, train_fr_toks, train_en_mask, train_fr_mask = tokenize_word_pairs(
#     model, train_word_pairs
# )
# test_en_toks, test_fr_toks, test_en_mask, test_fr_mask = tokenize_word_pairs(
#     model, test_word_pairs
# )
# %%
# t.save(
#     {
#         "en_toks": train_en_toks,
#         "fr_toks": train_fr_toks,
#         "en_mask": train_en_mask,
#         "fr_mask": train_fr_mask,
#     },
#     f"{token_caches_folder}/wikdict-train-en-fr-tokens.pt",
# )

# t.save(
#     {
#         "en_toks": test_en_toks,
#         "fr_toks": test_fr_toks,
#         "en_mask": test_en_mask,
#         "fr_mask": test_fr_mask,
#     },
#     f"{token_caches_folder}/wikdict-test-en-fr-tokens.pt",
# )
