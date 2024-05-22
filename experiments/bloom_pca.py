# %%
import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
from rich.console import Console
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from auto_embeds.data import get_cached_weights

console = Console()

# gpt2_small = HookedTransformer.from_pretrained("gpt2-small")
bloom_560m = get_cached_weights("bloom-560m", processing=False)
we = bloom_560m["W_E"]
d_vocab = we.shape[0]

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    "bigscience/bloom-560m"
)  # type: ignore


# %%
u, s, v = t.pca_lowrank(we, q=2)
pca_result = t.matmul(we, v[:, :2])
pca_df = pd.DataFrame(pca_result.cpu(), columns=["PC1", "PC2"])
tokens = tokenizer.batch_decode(list(range(d_vocab)))
# %%
pca_df["token"] = tokens
sample_indices = np.random.choice(pca_df.index, size=100000, replace=False)
pca_df_sample = pca_df.loc[sample_indices]

show_text_on_hover = True
# show_text_on_hover = False

if show_text_on_hover:
    fig = px.scatter(
        pca_df_sample,
        x="PC1",
        y="PC2",
        hover_name="token",
        title="PCA of bloom_560m Embeddings",
    )
else:
    fig = px.scatter(
        pca_df_sample,
        x="PC1",
        y="PC2",
        text="token",
        title="PCA of bloom_560m Embeddings",
    )

SHOW_PLOT_RANGE = (0, 99)


def should_show_plot(plot_index):
    start, end = SHOW_PLOT_RANGE
    return start <= plot_index <= end


plot_index = 1

if should_show_plot(plot_index):
    fig.show(responsive=True)
plot_index += 1

# %%
sample_indices = np.random.choice(pca_df.index, size=1000, replace=False)
pca_df_sample = pca_df.loc[sample_indices]
fig = px.scatter(
    pca_df_sample,
    x=pca_df_sample.index,
    y="PC2",
    hover_name="token",
    title="Second Principal Component of bloom_560m Embeddings",
    labels={"index": "Token ID"},
    trendline="ols",
).update_traces(line=dict(color="red"), selector=dict(mode="lines"))

if should_show_plot(plot_index):
    fig.show(responsive=True)
plot_index += 1

# %%
import re

with open(
    "/workspace/auto-embeds/datasets/english-words-list/dwyl-english-words.txt", "r"
) as file:
    words = set(re.sub(r"[^\w]", " ", file.read()).split())


def is_word(word):
    return word.lower() in words


def filter_english_words(df, column):
    return df[df[column].apply(is_word)]


def filter_chinese_words(df, column):
    return df[
        df[column].apply(lambda x: any("\u4e00" <= char <= "\u9fff" for char in x))
    ]


def print_side_by_side_table(title1, examples1, title2, examples2):
    print(f"{title1:<40} {title2}")
    print(
        f"{'token ID':>10} {'token':>10} {'PC2':>10}    {'token ID':>10} {'token':>10} {'PC2':>10}"
    )
    for (idx1, row1), (idx2, row2) in zip(examples1.iterrows(), examples2.iterrows()):
        print(
            f"{str(idx1):>10} {str(row1['token']):>10} {str(row1['PC2']):>10}    "
            f"{str(idx2):>10} {str(row2['token']):>10} {str(row2['PC2']):>10}"
        )


# filtered_pca_df = filter_english_words(pca_df, "token")
filtered_pca_df = filter_chinese_words(pca_df, "token")

top_pc2_tokens = filtered_pca_df.nlargest(300, "PC2")
bottom_pc2_tokens = filtered_pca_df.nsmallest(300, "PC2")

print_side_by_side_table(
    "top 300 tokens by pc2",
    top_pc2_tokens,
    "bottom 300 tokens by pc2",
    bottom_pc2_tokens,
)

# ##
we_tensor = t.tensor(we)
norms = t.norm(we_tensor, dim=1)
fig = px.histogram(
    norms.cpu(),
    nbins=200,
    title="Distribution of Norms of bloom_560m Embeddings",
    labels={"value": "Norm", "count": "Frequency"},
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
# reuse the sample_indices from above
sample_indices = np.random.choice(pca_df.index, size=10000, replace=False)
sampled_norms = norms[sample_indices].cpu()
sampled_tokens = [tokens[i] for i in sample_indices]

fig = px.scatter(
    x=sample_indices,
    y=sampled_norms,
    labels={"x": "token id", "y": "norm"},
    title="token id vs norm",
    trendline="ols",
    hover_name=sampled_tokens,
).update_traces(line=dict(color="red"), selector=dict(mode="lines"))

if should_show_plot(plot_index):
    fig.show(responsive=True)
plot_index += 1

# %%
# perform PCA
pca = PCA(n_components=100).fit(we)

# create a scree plot
explained_variance = pca.explained_variance_ratio_
fig = px.scatter(
    x=range(1, len(explained_variance) + 1),
    y=explained_variance,
    labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
    title="Scree Plot",
    mode="lines+markers",
)

if should_show_plot(plot_index):
    fig.show(responsive=True)
plot_index += 1

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Sample 1000 embeddings
sample_indices = np.random.choice(we.shape[0], size=10000, replace=False)
sampled_we = we[sample_indices].cpu().numpy()

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(sampled_we)
labels = kmeans.labels_

# Create a scatter plot of the first two principal components with cluster labels
pca_2d = PCA(n_components=2).fit_transform(sampled_we)
hover_text = [tokens[i] for i in sample_indices]
fig = px.scatter(
    x=pca_2d[:, 0],
    y=pca_2d[:, 1],
    color=labels,
    labels={"x": "PCA Component 1", "y": "PCA Component 2", "color": "Cluster"},
    title="KMeans Clustering of Sampled Embeddings",
    hover_name=hover_text,
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
# Elbow Method for Optimal Clusters
inertias = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(sampled_we)
    inertias.append(kmeans.inertia_)
fig = px.line(
    x=range(1, 20),
    y=inertias,
    labels={"x": "Number of Clusters", "y": "Inertia"},
    title="Elbow Method for Optimal Clusters",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
# Silhouette Score for Optimal Clusters
silhouette_scores = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(sampled_we)
    score = silhouette_score(sampled_we, kmeans.labels_)
    silhouette_scores.append(score)
fig = px.line(
    x=range(2, 15),
    y=silhouette_scores,
    labels={"x": "Number of Clusters", "y": "Silhouette Score"},
    title="Silhouette Score for Optimal Clusters",
)

if should_show_plot(plot_index):
    fig.show(config={"responsive": True})
plot_index += 1

# %%
# the second PCA component in chinese seems to represent something like an official /
# political tone. i wonder what this distribution looks like when we apply various types
# of layernorm. to test this, let's take a sample of 10000 tokens and apply various types
# of layernorm to the second PCA component and see how the distribution changes.


from hanziconv import HanziConv


def filter_chinese_words(df, column):
    return df[
        df[column].apply(lambda x: any("\u4e00" <= char <= "\u9fff" for char in x))
    ]


def print_side_by_side_table(title1, examples1, title2, examples2):
    print(f"{title1:<40} {title2}")
    print(
        f"{'token ID':>10} {'token':>10} {'PC2':>10}    {'token ID':>10} {'token':>10} {'PC2':>10}"
    )
    for (idx1, row1), (idx2, row2) in zip(examples1.iterrows(), examples2.iterrows()):
        print(
            f"{str(idx1):>10} {str(row1['token']):>10} {str(row1['PC2']):>10}    "
            f"{str(idx2):>10} {str(row2['token']):>10} {str(row2['PC2']):>10}"
        )


filtered_pca_df = filter_chinese_words(pca_df, "token").sample(n=10000, random_state=0)
filtered_pca_df["is_traditional"] = filtered_pca_df["token"].apply(
    lambda x: HanziConv.toSimplified(x) != x
)

top_pc2_tokens = filtered_pca_df.nlargest(300, "PC2")
bottom_pc2_tokens = filtered_pca_df.nsmallest(300, "PC2")

print_side_by_side_table(
    "top 300 tokens by pc2",
    top_pc2_tokens,
    "bottom 300 tokens by pc2",
    bottom_pc2_tokens,
)

fig = px.scatter(
    filtered_pca_df,
    x="PC2",
    y=filtered_pca_df.index,
    color="is_traditional",
    hover_name="token",
    title="Distribution of Chinese Words along PC2",
    labels={"x": "Index", "y": "PC2"},
)
fig.show(config={"responsive": True})
# %%
import roma
import torch

# Assuming PC2 is a rotation matrix
PC2 = v[:, 1]
print(PC2.shape)
identity_matrix = torch.eye(PC2.size(0))

# 1. Check if PC2 is a valid rotation matrix
is_rotation = roma.is_rotation_matrix(PC2)
print(f"Is PC2 a rotation matrix? {is_rotation}")

# 2. Verify if PC2 is an orthonormal matrix
is_orthonormal = roma.is_orthonormal_matrix(PC2)
print(f"Is PC2 an orthonormal matrix? {is_orthonormal}")

# 3. Measure the angular distance from the identity matrix
distance = roma.rotmat_geodesic_distance(PC2, identity_matrix)
print(f"Geodesic distance from identity matrix: {distance}")
