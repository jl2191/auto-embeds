# %%
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
from fancy_einsum import einsum
from rich.console import Console
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from auto_embeds.data import get_cached_weights

console = Console()

# weights = get_cached_weights("gpt2-small", processing=False)
weights = get_cached_weights("bloom-560m", processing=False)
# weights = get_cached_weights("bloom-3b", processing=False)

tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
    "bigscience/bloom-560m"
)  # type: ignore

original_we = weights["W_E"]
original_d_vocab = original_we.shape[0]
original_tokens = tokenizer.batch_decode(list(range(original_d_vocab)))

# %%
SHOW_PLOT_RANGE = (2, 99)


def should_show_plot(plot_index):
    start, end = SHOW_PLOT_RANGE
    return start <= plot_index <= end


plot_index = 0

# %%
we = original_we
u, s, v = t.pca_lowrank(we, q=2)
pca_result = t.matmul(we, v[:, :2])
pca_df = pd.DataFrame(pca_result.cpu(), columns=["PC1", "PC2"])
# %%
pca_df["token"] = original_tokens
sample_indices = np.random.choice(pca_df.index, size=50000, replace=False)
pca_df_sample = pca_df.loc[sample_indices]

show_text_on_hover = True
# show_text_on_hover = False

if show_text_on_hover:
    fig = px.scatter(
        pca_df_sample,
        x="PC1",
        y="PC2",
        hover_name="token",
        title="PCA of GPT2 Embeddings",
    )
else:
    fig = px.scatter(
        pca_df_sample,
        x="PC1",
        y="PC2",
        text="token",
        title="PCA of GPT2 Embeddings",
    )
if should_show_plot(plot_index):
    fig.show(responsive=True)
plot_index += 1

# %%
we = original_we[sample_indices]
# compute pairwise cosine similarities
we_normed = we / t.norm(we, dim=1, keepdim=True)
cos_sims = t.matmul(we_normed, we_normed.T)
cos_sims_avg = t.mean(cos_sims, dim=-1)

# %%
# plot histogram of pairwise cosine similarities
fig_hist = px.histogram(
    cos_sims_avg.cpu(),
    nbins=500,
    title="Histogram of Pairwise Cosine Similarities for Embeddings",
    labels={"value": "Cosine Similarity"},
)

if should_show_plot(plot_index):
    fig_hist.show(responsive=True)
plot_index += 1


# %%
def is_chinese_or_latin(token):
    try:
        contains_latin = any("LATIN" in unicodedata.name(char) for char in token)
        contains_chinese = any(
            "CJK UNIFIED IDEOGRAPH" in unicodedata.name(char) for char in token
        )

        if contains_latin:
            return token.isascii()
        if contains_chinese:
            return all(
                "CJK UNIFIED IDEOGRAPH" in unicodedata.name(char) or char.isspace()
                for char in token
            )

        return False
    except ValueError:
        return False


filtered_indices = [
    i for i, token in enumerate(original_tokens) if is_chinese_or_latin(token)
]
subset_size = 80000
if len(filtered_indices) > subset_size:
    filtered_indices = np.random.choice(
        filtered_indices, size=subset_size, replace=False
    )
we = original_we[filtered_indices]
we_normed = we / t.norm(we, dim=1, keepdim=True)
new_to_original_index = {
    new_index: original_index
    for new_index, original_index in enumerate(filtered_indices)
}


def new_to_original_indices(new_idx):
    """
    Given a new index, return the corresponding original index.
    """
    return new_to_original_index.get(new_idx, None)


tokens = [original_tokens[i] for i in filtered_indices]


# %%
# we = t.nn.functional.layer_norm(we, we.shape)


def print_top_k_similar_embeddings(embedding, k=10, print_results=True):
    """
    Prints the top k embeddings closest to the given embedding in terms of cosine
    similarity.
    """
    # Normalize the given embedding
    embedding_normed = embedding / t.norm(embedding, dim=-1, keepdim=True)

    # Compute cosine similarities with all other embeddings
    cos_sims = t.matmul(embedding_normed, we_normed.T)

    # Get the top k indices with the highest cosine similarities
    top_k_indices = t.topk(cos_sims, k=k, dim=-1).indices.squeeze()

    if print_results:
        # Decode the tokens corresponding to the top k indices
        top_k_tokens = [tokens[i] for i in top_k_indices.tolist()]

        # Print the top k tokens and their cosine similarities
        for idx in range(len(top_k_indices)):
            print(
                f"Token: {top_k_tokens[idx]}, Cosine Similarity: "
                f"{cos_sims[0, top_k_indices[idx]].item()}"
            )
    return top_k_indices.tolist()


# Example usage
for i in range(31_000, 31_100):
    random_embedding = we[i].unsqueeze(0)
    top_k_indices = print_top_k_similar_embeddings(random_embedding)
    print(top_k_indices)
    print()


# %%


def find_closest_embeddings(embedding, we, k=10000):
    embedding_normed = embedding / t.norm(embedding, dim=1, keepdim=True)
    cos_sims = t.matmul(embedding_normed, we_normed.T)
    top_k_indices = t.topk(cos_sims, k=k, dim=-1).indices.squeeze()
    return we[top_k_indices]


def calculate_rotation(A, B):
    u, w, vt = t.linalg.svd(t.matmul(B.T, A).T)
    R = u @ vt
    return R


def calculate_loss(A, B, R):
    # print(R.shape)
    # print(A.shape)
    A_rotated = R @ A.T
    A_rotated = einsum("ij,ij->ji", R, A)
    A_rotated = A_rotated.T
    loss = t.nn.functional.cosine_similarity(A_rotated, B, dim=-1).mean().item()
    return loss


def find_best_rotation_pairs(we, num_pairs=500, k=5000):
    d_vocab = we.shape[0]
    losses = []
    for _ in range(num_pairs):
        idx_A, idx_B = np.random.choice(d_vocab, size=2, replace=False)
        A = we[idx_A].unsqueeze(0)  # shape [1,
        B = we[idx_B].unsqueeze(0)
        closest_A = find_closest_embeddings(A, we, k)
        closest_B = find_closest_embeddings(B, we, k)
        R = calculate_rotation(closest_A, closest_B)
        loss = calculate_loss(closest_A, closest_B, R)
        losses.append((idx_A, idx_B, loss))
    losses.sort(key=lambda x: x[2])
    return losses


# lp = LineProfiler()
# lp.add_function(find_closest_embeddings)
# lp.add_function(calculate_rotation)
# lp.add_function(calculate_loss)
# lp.add_function(find_best_rotation_pairs)
# best_pairs = lp(find_best_rotation_pairs)(we)

with t.no_grad():
    best_pairs = find_best_rotation_pairs(we)

for idx_A, idx_B, loss in best_pairs[:40]:
    token_A = tokens[idx_A]
    token_B = tokens[idx_B]
    print(f"Pair (A: {idx_A} - '{token_A}', B: {idx_B} - '{token_B}') - Loss: {loss}")

# lp.print_stats()
