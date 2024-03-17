# %%
import json
import os
import random

from jaxtyping import Float

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch as t
import torch.testing as tt
import transformer_lens as tl
from IPython.core.getipython import get_ipython
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, random_split

import wandb
from auto_steer.steering_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    initialize_transform_and_optim,
    train_transform,
    filter_word_pairs,
    tokenize_word_pairs,
    initialize_loss,
)
from auto_steer.utils.misc import repo_path_to_abs_path

# %%
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)
try:
    get_ipython().run_line_magic("load_ext", "autoreload")  #type: ignore
    get_ipython().run_line_magic("load_ext", "line_profiler")  #type: ignore
    get_ipython().run_line_magic("autoreload", "2")   #type: ignore
except:
    pass

# Model setup
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")

# Data preparation (example for loading and processing word pairs)
file_path = f"{datasets_folder}/wikdict/2_extracted/eng-fra.json"
with open(file_path, "r") as file:
    word_pairs = json.load(file)
random.seed(1)
random.shuffle(word_pairs)
split_index = int(len(word_pairs) * 0.95)
train_word_pairs = filter_word_pairs(model, word_pairs[:split_index], discard_if_same=True, min_length=3, capture_diff_case=True, capture_space=True, capture_no_space=True)
test_word_pairs = filter_word_pairs(model, word_pairs[split_index:], discard_if_same=True, min_length=3, capture_diff_case=True, capture_space=True, capture_no_space=True)

# Tokenization and DataLoader preparation
train_en_toks, train_fr_toks, _, _ = tokenize_word_pairs(model, train_word_pairs)
test_en_toks, test_fr_toks, _, _ = tokenize_word_pairs(model, test_word_pairs)

train_en_embeds = model.embed.W_E[train_en_toks].detach().clone()  # shape[batch, seq_len, d_model]
train_fr_embeds = model.embed.W_E[train_fr_toks].detach().clone()  # shape[batch, seq_len, d_model]
test_en_embeds = model.embed.W_E[test_en_toks].detach().clone()  # shape[batch, seq_len, d_model]
test_fr_embeds = model.embed.W_E[test_fr_toks].detach().clone()  # shape[batch, seq_len, d_model]

total_train_dataset = TensorDataset(train_en_embeds, train_fr_embeds)
total_test_dataset = TensorDataset(test_en_embeds, test_fr_embeds)
train_loader = DataLoader(total_train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(total_test_dataset, batch_size=512, shuffle=False)

# Transformation and Optimization Initialization
transformation_names = [
    # "identity",
    # "translation",
    # "linear_map",
    # "offset_linear_map",
    # "uncentered_linear_map",
    "rotation",
    # "offset_rotation",
    # "uncentered_rotation"
]

transform, optim = initialize_transform_and_optim(
    d_model,
    transformation="offset_linear_map",
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
        device=device,
    )
else:
    print("Optimizer not initialized, skipping training.")

accuracy = evaluate_accuracy(
    model,
    test_loader,
    transform,
    exact_match=False,
    print_results=True,
)
print(f"Correct Percentage: {accuracy * 100:.2f}%")
print("Test Accuracy:", calc_cos_sim_acc(test_loader, transform))

# %%
for name, param in transform.named_parameters():
    print(f"{name}: {param.data}")
    print(f"{name}: {param.data.shape}")
# %%
transform_result = transform(t.ones(1024).to(device))
    print(f"{num1.item()} - {num2.item()} = {num1.item() - num2.item()}")
# tt.assert_close(transform_result, transform_manual_result)

# %%
for num1, num2 in zip(transform_result, transform_manual_result):
    print(f"{num1.item()} - {num2.item()} = {num1.item() - {num2.item()}}")

# %%
transform.bias.requires_grad

# %%
# en_embeds has shape [batch=4270, seq_len=1, d_model=1024]
en_center = en_embeds.squeeze().mean(0)  # en_center has shape [d_model=1024]
en_center.shape

# en_embeds has shape [batch=4270, seq_len=1, d_model=1024]
fr_center = fr_embeds.squeeze().mean(0)  # fr_center has shape [d_model=1024]
fr_center.shape

# model.embed.W_E has shape [d_vocab=250880, d_model=1024]
all_center = model.embed.W_E.mean(0)  # all_center has shape [d_model=1024]

# %%
"""
i would predict that as english made up 31.3% of the training data, the center of the
english embeddings would be close to the center of the entire embedding matrix. french
makes up 13.5% of the training data. we will look at the l1 and l2 norm for this.
"""

l1_norm_en_center = t.norm(en_center, p=1)
l1_norm_fr_center = t.norm(fr_center, p=1)
l1_norm_all_center = t.norm(all_center, p=1)

print(f"L1 Norm of English Center: {l1_norm_en_center.item()}")
print(f"L1 Norm of French Center: {l1_norm_fr_center.item()}")
print(f"L1 Norm of All Center: {l1_norm_all_center.item()}")
print(0)
print(
    f"L1 Norm of English Center - All Center: {l1_norm_en_center.item() - l1_norm_all_center.item()}"
)
print(
    f"L1 Norm of French Center - All Center: {l1_norm_fr_center.item() - l1_norm_all_center.item()}"
)

# %%
l2_norm_en_center = t.norm(en_center, p=2)
l2_norm_fr_center = t.norm(fr_center, p=2)
l2_norm_all_center = t.norm(all_center, p=2)

print()
print(f"L2 Norm of English Center: {l2_norm_en_center.item()}")
print(f"L2 Norm of French Center: {l2_norm_fr_center.item()}")
print(f"L2 Norm of All Center: {l2_norm_all_center.item()}")
print()
print(
    f"L2 Norm of English Center - All Center: {l2_norm_en_center.item() - l2_norm_all_center.item()}"
)
print(
    f"L2 Norm of French Center - All Center: {l2_norm_fr_center.item() - l2_norm_all_center.item()}"
)
# %%
"""
huh, the l1 and l2 norms of the english and french centers are very close to the l1 and
l2 norms of the entire center. i would have expected the l1 and l2 norms of the english
and french centers to be much smaller than the l1 and l2 norms of the entire center.
what is going on?

ahh okay, i think i would expect the the l1 and l2 norm of
(en_center-all_center) < (fr_center - all_center), this is different from what i have
calculated
"""
en_center_minus_all_center = en_center - all_center
fr_center_minus_all_center = fr_center - all_center
print(f"l1 norm of (en_center-all_center): {t.norm(en_center_minus_all_center, p=1)}")
print(f"l1 norm of (fr_center-all_center): {t.norm(fr_center_minus_all_center, p=1)}")


print(f"l2 norm of (en_center-all_center): {t.norm(en_center_minus_all_center, p=2)}")
print(f"l2 norm of (fr_center-all_center): {t.norm(fr_center_minus_all_center, p=2)}")

"""
they don't?? weird. and so in terms of embeddings, the french tokens are closer to the
center of the model's vocab that english tokens? i would have expected the other way
round due to there being just many more english tokens.
"""

# %%
"""
surely we would expect the average cosine similarity between the english word embeddings
and the french word embeddings to be bigger than between all the english word embeddings
and the french word embeddings?
"""


def calculate_avg_cosine_similarity(embeds):
    # Normalize the embeddings to unit vectors
    norm_embeds = t.nn.functional.normalize(embeds.squeeze(), p=2, dim=1)
    # Compute cosine similarity using matrix multiplication
    cos_sim_matrix = t.mm(norm_embeds, norm_embeds.transpose(0, 1))
    # Zero out the diagonal (self-similarity) and extract upper triangle values
    cos_sim_values = cos_sim_matrix.masked_select(
        t.triu(t.ones_like(cos_sim_matrix), diagonal=1) > 0
    )
    # Calculate the average cosine similarity
    avg_cos_sim = cos_sim_values.mean().item()
    return avg_cos_sim


avg_cos_sim_en_embed = calculate_avg_cosine_similarity(en_embeds)
print(f"Average Cosine Similarity among English Embeddings: {avg_cos_sim_en_embed}")

avg_cos_sim_fr_embed = calculate_avg_cosine_similarity(fr_embeds)
print(f"Average Cosine Similarity among French Embeddings: {avg_cos_sim_fr_embed}")

avg_cos_sim_en_embed_and_fr_embed = calculate_avg_cosine_similarity(
    t.cat((en_embeds, fr_embeds), dim=0)  # shape [batch=8540, seq=1, d_model=1024]
)
print(
    f"Average Cosine Similarity between English and French Embeddings: {avg_cos_sim_en_embed_and_fr_embed}"
)

# %%
"""
hang on, that can't be correct, let me try with a cos_sim function that i have used in
the past
"""


def all_pairs_cos_sim(tensor: Float[Tensor, "seq d_model"]) -> Float[Tensor, "seq seq"]:
    """
    Takes in the activations from a given layer and returns the cosine similarity matrix
    representing all the cosine similarity of all the pairs of vectors
    """
    # Normalize each vector in the tensor
    normalized_tensor = tensor / tensor.norm(dim=1)[:, None]

    # Compute cosine similarities matrix by transposition
    cos_sim_matrix = t.mm(normalized_tensor, normalized_tensor.transpose(0, 1))

    return cos_sim_matrix


def avg_all_pairs_cos_sim(
    tensor: Float[Tensor, "batch seq d_model"]
) -> Float[Tensor, "batch seq"]:
    """
    Calculates the average cosine similarity for all pairs of vectors in a tensor
    (excluding self-similarity)
    """
    # Compute the cosine similarity matrix for all pairs of vectors in the tensor
    cos_sim_matrix = all_pairs_cos_sim(tensor)

    # Exclude self-similarity by setting the diagonal elements to zero
    cos_sim_matrix.fill_diagonal_(0)

    # Calculate the number of non-diagonal elements in the matrix
    num_non_diag_elements = cos_sim_matrix.numel() - cos_sim_matrix.shape[0]

    # Calculate the average cosine similarity by summing all elements and dividing by
    # the number of non-diagonal elements
    avg_cos_sim = cos_sim_matrix.sum() / num_non_diag_elements

    return avg_cos_sim


avg_cos_sim_en_embed = avg_all_pairs_cos_sim(en_embeds.squeeze())
print(f"Average Cosine Similarity among English Embeddings: {avg_cos_sim_en_embed}")

avg_cos_sim_fr_embed = avg_all_pairs_cos_sim(fr_embeds.squeeze())
print(f"Average Cosine Similarity among French Embeddings: {avg_cos_sim_fr_embed}")

avg_cos_sim_en_embed_and_fr_embed = avg_all_pairs_cos_sim(
    t.cat((en_embeds, fr_embeds), dim=0).squeeze()  # shape [batch=8540, d_model=1024]
)
print(
    f"Average Cosine Similarity between English and French Embeddings: {avg_cos_sim_en_embed_and_fr_embed}"
)

# %%
"""
gawd damn, gives the same result. it's roughly the same distance between the two? surely
if we pair up english and french embedding, then average that, the result is going to be
lower than both among English embeddings and among French embeddings?
"""
en_fr_pair_avg_cos_sim = t.nn.functional.cosine_similarity(
    en_embeds.squeeze(), fr_embeds.squeeze(), dim=-1
).mean(0)
print(en_fr_pair_avg_cos_sim)

"""
okay good lmao, i'll just assume that it was due to pairing weirdness.
"""

# %%
"""
let me remind myself of the model architecture again
"""
for name, param in model.named_parameters():
    print(name, param.shape)

# %%
print(model)

# %%
"""
hmm yes, there is that layer norm, what if we do what we have done after layernorm?
"""
processing_model = tl.HookedTransformer.from_pretrained("bloom-560m")
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")

# %%
t.allclose(processing_model.embed.W_E, model.embed.W_E)
"""
okay and so in transformer lens because of layernorm folding, we center all the weights
write to the residual stream such as W_E. as such, this assert is not equal.
"""

# %%
import plotly.express as px
from sklearn.manifold import TSNE
from umap import UMAP

# Combine English and French embeddings for visualization
combined_embeddings = t.cat((en_embeds, fr_embeds), dim=0).squeeze().cpu().numpy()
combined_labels = ["English"] * en_embeds.size(0) + ["French"] * fr_embeds.size(0)

# Perform TSNE
tsne_embeddings = TSNE(n_components=2).fit_transform(combined_embeddings)

# Perform UMAP
umap_embeddings = UMAP(n_components=2).fit_transform(combined_embeddings)

# Plot TSNE
fig_tsne = px.scatter(
    x=tsne_embeddings[:, 0], y=tsne_embeddings[:, 1], color=combined_labels
)
fig_tsne.update_layout(title="TSNE of English and French Embeddings")
fig_tsne.show()

# Plot UMAP
fig_umap = px.scatter(
    x=umap_embeddings[:, 0], y=umap_embeddings[:, 1], color=combined_labels
)
fig_umap.update_layout(title="UMAP of English and French Embeddings")
fig_umap.show()

# %%
import plotly.express as px
from sklearn.manifold import TSNE
from umap import UMAP

# Combine English and French embeddings for visualization
combined_embeddings = t.cat((en_embeds, fr_embeds), dim=0).squeeze().cpu().numpy()
combined_labels = ["English"] * en_embeds.size(0) + ["French"] * fr_embeds.size(0)

# Perform TSNE in 3D
tsne_embeddings = TSNE(n_components=3).fit_transform(combined_embeddings)

# Perform UMAP in 3D
umap_embeddings = UMAP(n_components=3).fit_transform(combined_embeddings)

# Plot TSNE in 3D
fig_tsne = px.scatter_3d(
    x=tsne_embeddings[:, 0],
    y=tsne_embeddings[:, 1],
    z=tsne_embeddings[:, 2],
    color=combined_labels,
)
fig_tsne.update_layout(title="3D TSNE of English and French Embeddings")
fig_tsne.show()

# Plot UMAP in 3D
fig_umap = px.scatter_3d(
    x=umap_embeddings[:, 0],
    y=umap_embeddings[:, 1],
    z=umap_embeddings[:, 2],
    color=combined_labels,
)
fig_umap.update_layout(title="3D UMAP of English and French Embeddings")
fig_umap.show()

# %%
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from umap import UMAP

# Access all embeddings from the model
all_embeddings = model.embed.W_E.detach().clone().cpu().numpy()

# Create a DataFrame to hold embeddings and their labels
embeddings_df = pd.DataFrame(all_embeddings)

# Label English and French embeddings based on en_toks and fr_toks indices
embeddings_df["Language"] = ["Other"] * embeddings_df.shape[0]  # Default to 'Other'
for idx in en_toks.squeeze().tolist():
    embeddings_df.at[idx, "Language"] = "English"
for idx in fr_toks.squeeze().tolist():
    embeddings_df.at[idx, "Language"] = "French"

# Perform TSNE in 3D on all embeddings
tsne_embeddings = TSNE(n_components=3, random_state=42).fit_transform(
    embeddings_df.iloc[:, :-1]
)
embeddings_df[["TSNE_1", "TSNE_2", "TSNE_3"]] = tsne_embeddings

# Perform UMAP in 3D on all embeddings
umap_embeddings = UMAP(n_components=3, random_state=42).fit_transform(
    embeddings_df.iloc[:, :-4]
)
embeddings_df[["UMAP_1", "UMAP_2", "UMAP_3"]] = umap_embeddings


# Define a function to plot embeddings in 3D
def plot_3d_embeddings(df, embedding_cols, title):
    fig = px.scatter_3d(
        df,
        x=embedding_cols[0],
        y=embedding_cols[1],
        z=embedding_cols[2],
        color="Language",
    )
    fig.update_layout(title=title)
    fig.show()


# Plot TSNE in 3D for all embeddings
plot_3d_embeddings(
    embeddings_df, ["TSNE_1", "TSNE_2", "TSNE_3"], "3D TSNE of All Model Embeddings"
)

# Plot UMAP in 3D for all embeddings
plot_3d_embeddings(
    embeddings_df, ["UMAP_1", "UMAP_2", "UMAP_3"], "3D UMAP of All Model Embeddings"
)
