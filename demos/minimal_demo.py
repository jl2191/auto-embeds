# %%
import json
import random
from typing import Tuple

import tabulate
import torch as t
from fancy_einsum import einsum
from jaxtyping import Float
from torch import Tensor
from transformers import AutoTokenizer

from auto_embeds.data import filter_word_pairs, get_cached_weights, get_dataset_path
from auto_embeds.embed_utils import initialize_embed_and_unembed

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model_weights = get_cached_weights("bloom-560m", False)
embed_module, unembed_module = initialize_embed_and_unembed(
    tokenizer=tokenizer, model_weights=model_weights, device="cuda"
)
WE = model_weights["W_E"]
WU = model_weights["W_U"]
b_U = model_weights["b_U"]

d_model = model_weights["W_E"].shape[1]
n_toks = model_weights["W_E"].shape[0]

# some processing to get our dataset. we're using the CC CEDICT dataset
# which is a dataset of Chinese to English translations extracted from the CC-CEDICT
# dictionary. filter_word_pairs is rather long and so has been abstracted away.

dataset_name = "cc_cedict_zh_en_extracted"
file_path = get_dataset_path(dataset_name)
with open(file_path, "r", encoding="utf-8") as file:
    word_pairs = json.load(file)

all_word_pairs = filter_word_pairs(
    tokenizer=tokenizer,
    word_pairs=word_pairs,
    discard_if_same=True,
    min_length=2,
    print_number=True,
    space_configurations=[{"en": "no_space", "fr": "space"}],
    verbose_count=True,
)

all_word_pairs_copy = all_word_pairs.copy()
random.seed(0)
random.shuffle(all_word_pairs_copy)
selected_index = random.randint(0, len(all_word_pairs_copy) - 1)
selected_pair = all_word_pairs_copy.pop(selected_index)
src_word, tgt_word = selected_pair

# for our selected src and tgt word
## tokenize
src_tok = (
    tokenizer(src_word, return_tensors="pt", add_special_tokens=False)
    .data["input_ids"]
    .to("cuda")
)
tgt_tok = (
    tokenizer(tgt_word, return_tensors="pt", add_special_tokens=False)
    .data["input_ids"]
    .to("cuda")
)
print(src_tok)
print(tgt_tok)

src_tok_str = tokenizer.batch_decode(src_tok)
tgt_tok_str = tokenizer.batch_decode(tgt_tok)

print(src_tok_str)
print(tgt_tok_str)


def embed(token_ids, apply_ln=True):
    embeds = WE[token_ids, :]
    if apply_ln:
        embeds = t.nn.functional.layer_norm(embeds, embeds.shape[1:])
    return embeds


def unembed(embeds):
    """Unembeds embeddings to get the token IDs by indexing with W_U and adding b_U"""
    return (
        einsum(
            "batch pos d_model, d_model vocab -> batch pos vocab",
            embeds,
            WU,
        )
        + b_U
    )


def embeds_to_str(embeds):
    return tokenizer.batch_decode(unembed(embeds).argmax(dim=-1))


# %%
## embed
src_embed = embed(src_tok).detach().clone().squeeze()
tgt_embed = embed(tgt_tok).detach().clone().squeeze()
src_other_words = [word_pair[0] for word_pair in all_word_pairs_copy]
tgt_other_words = [word_pair[1] for word_pair in all_word_pairs_copy]
# as we have removed the selected pair, the length should be one less
assert len(src_other_words) == len(all_word_pairs) - 1
assert len(tgt_other_words) == len(all_word_pairs) - 1
# tokenize
src_other_toks = (
    tokenizer(src_other_words, return_tensors="pt", add_special_tokens=False)
    .data["input_ids"]
    .to("cuda")
)
tgt_other_toks = (
    tokenizer(tgt_other_words, return_tensors="pt", add_special_tokens=False)
    .data["input_ids"]
    .to("cuda")
)
## embed
src_other_embeds = embed_module(src_other_toks).detach().clone().squeeze(1)
tgt_other_embeds = embed_module(tgt_other_toks).detach().clone().squeeze(1)
# both should have shape [batch, d_model]
# calculate cosine similarities and euclidean distances
src_cos_sims = t.cosine_similarity(src_embed, src_other_embeds, dim=-1)
tgt_cos_sims = t.cosine_similarity(tgt_embed, tgt_other_embeds, dim=-1)
src_euc_dists = t.pairwise_distance(src_embed.unsqueeze(0), src_other_embeds, p=2)
tgt_euc_dists = t.pairwise_distance(tgt_embed.unsqueeze(0), tgt_other_embeds, p=2)
random_embed = src_embed
random_embed_w_src_embeds_cos_sims = t.cosine_similarity(
    random_embed, src_other_embeds, dim=-1
)
random_embed_w_tgt_embeds_cos_sims = t.cosine_similarity(
    random_embed, tgt_other_embeds, dim=-1
)
random_embed_w_all_embeds_cos_sims = t.cat(
    (random_embed_w_src_embeds_cos_sims, random_embed_w_tgt_embeds_cos_sims)
)
# get the indices of the top k cos sims for both src and tgt
_, indices = t.topk(
    random_embed_w_all_embeds_cos_sims,
    random_embed_w_all_embeds_cos_sims.shape[0],
)
# turn it into a list to get the correct indices for tgt embeds as they are
# offset by the src embeds
indices = indices.tolist()
indices = [
    index - src_other_embeds.shape[0] if index >= src_other_embeds.shape[0] else index
    for index in indices
]
test_indices = t.tensor(indices)
test_indices = t.unique_consecutive(test_indices)[:100]

src_other_embeds = src_other_embeds.unsqueeze(1)
src_embed = src_embed.unsqueeze(1)
tgt_other_embeds = tgt_other_embeds.unsqueeze(1)
tgt_embed = tgt_embed.unsqueeze(1)


# %%
# now the fun part, trying out the various combinations of layernorms and transforms


def calculate_translation(
    src_embed: Float[Tensor, "batch pos d_model"],
    tgt_embed: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "pos d_model"]:
    """Calculates translation vector for source to target language embeddings."""
    X = src_embed.detach().clone().squeeze()
    Y = tgt_embed.detach().clone().squeeze()
    T = t.mean(Y - X, dim=0)
    return T


def calculate_procrustes_torch(
    src_embed: Float[Tensor, "batch pos d_model"],
    tgt_embed: Float[Tensor, "batch pos d_model"],
) -> Tuple[Float[Tensor, "d_model d_model"], Float[Tensor, ""]]:
    A = src_embed.detach().clone().squeeze()
    B = tgt_embed.detach().clone().squeeze()
    u, w, vt = t.linalg.svd(t.matmul(B.T, A).T)
    R = u @ vt
    scale = w.sum()
    return R, scale


def calculate_linear_map(
    src_embed: Float[Tensor, "batch pos d_model"],
    tgt_embed: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "d_model d_model"]:
    """Calculate the best linear map matrix for source to target language embeddings."""
    A = src_embed.detach().clone().squeeze()
    B = tgt_embed.detach().clone().squeeze()
    result = t.linalg.lstsq(A, B)
    X = result.solution
    return X


# %%
# transform, scale = calculate_procrustes_torch(src_other_embeds, tgt_other_embeds)
transform = calculate_linear_map(src_other_embeds, tgt_other_embeds)
predicted_tgt_embed = einsum(
    "d_model_row d_model_col, batch pos d_model_row -> batch pos d_model_col",
    transform,
    src_other_embeds,
)

# %% print results in a table
source_token = tokenizer.batch_decode(unembed(src_other_embeds).argmax(dim=-1))
target_token = tokenizer.batch_decode(unembed(tgt_other_embeds).argmax(dim=-1))
predicted_target_token = tokenizer.batch_decode(
    unembed(predicted_tgt_embed).argmax(dim=-1)
)
# add a new column with green tick emoji if the predicted token matches original
# target token
correct = [
    "✅" if pred == target else "❌"
    for pred, target in zip(predicted_target_token, target_token)
]

# calculate percentage of correct predictions
percentage_correct = sum(1 for match in correct if match == "✅") / len(correct) * 100

# calculate average cosine similarity
average_cosine_similarity = (
    t.nn.functional.cosine_similarity(predicted_tgt_embed, tgt_other_embeds, dim=-1)
    .mean()
    .item()
)

# calculate average MSE loss
average_mse_loss = t.nn.functional.mse_loss(
    predicted_tgt_embed, tgt_other_embeds
).item()

table_data = list(zip(source_token, target_token, predicted_target_token, correct))
table_headers = [
    "Original Source Token",
    "Original Target Token",
    "Predicted Token",
    "Match Status",
]
print(tabulate.tabulate(table_data, headers=table_headers, tablefmt="grid"))
print(f"Percentage Correct: {percentage_correct:.2f}%")
print(f"Average Cosine Similarity: {average_cosine_similarity:.4f}")
print(f"Average MSE Loss: {average_mse_loss:.4f}")
print(
    """
if all has gone well, you should see a percentage correct figure of around 55%. one
thing to note is that in some sense this may be underestimating how well the
transformation performs as a lot of the translations can be thought of as a good
translation but not the exact one that we were looking for. our results notebooks
(they are on their way) will demonstrate our way of getting around this, along with
showing off the other various features of the library!
"""
)
