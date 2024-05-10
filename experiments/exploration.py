# %%
import itertools

import plotly.graph_objects as go
import torch as t
import transformer_lens as tl
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from auto_embeds.data import get_cached_weights
from auto_embeds.embed_utils import initialize_embed_and_unembed

# %%
print(
    """
    given a random embedding, when you do the logits, what sort of distribution do you
    get?
    """
)

# %%
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
    "bigscience/bloom-560m"
)  # type: ignore
model_weights = get_cached_weights("bigscience/bloom-560m", False)

# %%
random_token_str = " hello"
random_token_id = tokenizer.encode(random_token_str, return_tensors="pt")
logger.info(f"random_token_str: {random_token_str}")
logger.info(f"random_token_id: {random_token_id}")

# %%
embed_module, unembed_module = initialize_embed_and_unembed(
    tokenizer=tokenizer,
    model_weights=model_weights,
    embed_weight="model_weights",
    embed_ln=True,
    embed_ln_weights="model_weights",
    unembed_weight="model_weights",
    unembed_ln=True,
    unembed_ln_weights="default_weights",
)

# %%
with t.no_grad():
    random_token_embedding = embed_module(random_token_id)
logger.info(f"random_token_embedding: {random_token_embedding}")
logger.info(f"random_token_embedding.shape: {random_token_embedding.shape}")

# %%
with t.no_grad():
    random_token_unembedded = unembed_module(random_token_embedding)
logger.info(f"random_token_unembedded: {random_token_unembedded}")
logger.info(f"random_token_unembedded.shape: {random_token_unembedded.shape}")

# %%
embedding_values = random_token_unembedded.squeeze().tolist()
token_ids_list = list(range(len(embedding_values)))
token_ids_tensor = t.tensor(token_ids_list)
token_labels = tokenizer.batch_decode(token_ids_tensor)
# %%
# plot the embedding values
fig = go.Figure(
    data=go.Scatter(
        x=token_ids_list,
        y=embedding_values,
        # mode='markers',
        hoverinfo="text",
        text=token_labels,
    )
)
fig.update_layout(
    title="Token Embedding Logits",
    xaxis_title="Token ID",
    yaxis_title="Embedding Value",
)
fig.show()

# %%
# plot the embedding values after softmax
softmaxed_embedding_values = random_token_unembedded.squeeze().softmax(dim=0).tolist()
fig = go.Figure(
    data=go.Scatter(
        x=token_ids_list,
        y=softmaxed_embedding_values,
        hoverinfo="text",
        text=token_labels,
    )
)
fig.update_layout(
    title="Token Embedding Values after Softmax",
    xaxis_title="Token ID",
    yaxis_title="Softmax Embedding Value",
)
fig.show()

print(
    """using default_weights for the unembed_ln_weights instead of model_weights means
    that we now have little bumps on things close to '_hello' like 'Hello'
    """
)

# %%
# Iterate through combinations of initialize_embed_and_unembed for plotting
embed_config_possibilities = {
    "embed_weight": ["model_weights"],
    "embed_ln_weights": ["no_ln", "model_weights", "default_weights"],
    "unembed_weight": ["model_weights"],
    "unembed_ln_weights": ["no_ln", "model_weights", "default_weights"],
}
config_list = list(itertools.product(*embed_config_possibilities.values()))

# %%
logits_figures = []
softmax_figures = []

for (
    embed_weight,
    embed_ln_weights,
    unembed_weight,
    unembed_ln_weights,
) in config_list:
    if embed_ln_weights == "no_ln":
        embed_ln = False
    else:
        embed_ln = True

    if unembed_ln_weights == "no_ln":
        unembed_ln = False
    else:
        unembed_ln = True

    embed_module, unembed_module = initialize_embed_and_unembed(
        tokenizer=tokenizer,
        model_weights=model_weights,
        embed_weight=embed_weight,
        embed_ln=embed_ln,
        embed_ln_weights=embed_ln_weights,
        unembed_weight=unembed_weight,
        unembed_ln=unembed_ln,
        unembed_ln_weights=unembed_ln_weights,
    )

    with t.no_grad():
        random_token_embedding = embed_module(random_token_id).unsqueeze(0)
    with t.no_grad():
        random_token_unembedded = unembed_module(random_token_embedding)
    embedding_values = random_token_unembedded.squeeze().tolist()
    token_ids_list = list(range(len(embedding_values)))
    token_labels = tokenizer.batch_decode(token_ids_tensor)
    softmaxed_values = random_token_unembedded.squeeze().softmax(dim=0).tolist()

    # Prepare the configured embedding values plot
    logits_fig = go.Figure(
        data=go.Scatter(
            x=token_ids_list,
            y=embedding_values,
            hoverinfo="text",
            text=token_labels,
        )
    )
    logits_fig.update_layout(
        title=f"Token Embedding Logits with {embed_ln_weights} and {unembed_ln_weights}",
        xaxis_title="Token ID",
        yaxis_title="Embedding Value",
    )
    logits_figures.append(logits_fig)

    # Prepare the softmaxed embedding values plot
    softmax_fig = go.Figure(
        data=go.Scatter(
            x=token_ids_list,
            y=softmaxed_values,
            hoverinfo="text",
            text=token_labels,
        )
    )
    softmax_fig.update_layout(
        title=f"Token Embedding Values after Softmax with {embed_ln_weights} and {unembed_ln_weights}",
        xaxis_title="Token ID",
        yaxis_title="Softmax Embedding Value",
    )
    softmax_figures.append(softmax_fig)

# %%
for fig in logits_figures:
    fig.show()

for fig in softmax_figures:
    fig.show()

# %%
# Calculate L1 norm of the embedding weights
l1_norms = t.norm(embed_module.W_E, p=1, dim=1).tolist()

l1_norm_fig = go.Figure(
    data=go.Scatter(
        x=list(range(len(l1_norms))),
        y=l1_norms,
        mode="markers",
        hoverinfo="text",
        text=token_labels,
    )
)
l1_norm_fig.update_layout(
    title="L1 Norm of Bloom Embedding Weights",
    xaxis_title="Token ID",
    yaxis_title="L1 Norm",
)
l1_norm_fig.show()

# %%
# Calculate L2 norm of the embedding weights
l2_norms = t.norm(embed_module.W_E, p=2, dim=1).tolist()

l2_norm_fig = go.Figure(
    data=go.Scatter(
        x=list(range(len(l2_norms))),
        y=l2_norms,
        mode="markers",
        hoverinfo="text",
        text=token_labels,
    )
)
l2_norm_fig.update_layout(
    title="L2 Norm of Bloom Embedding Weights",
    xaxis_title="Token ID",
    yaxis_title="L2 Norm",
)
l2_norm_fig.show()

# %%
gpt2 = tl.HookedTransformer.from_pretrained_no_processing("gpt2-small")

# %%
W_E = gpt2.W_E.detach().clone()
del gpt2

# %%
W_E.shape

# %%
# Calculate L1 norm of the embedding weights
l1_norms = t.norm(W_E, p=1, dim=1).tolist()

l1_norm_fig = go.Figure(
    data=go.Scatter(
        x=list(range(len(l1_norms))),
        y=l1_norms,
        mode="markers",
        hoverinfo="text",
        text=token_labels,
    )
)
l1_norm_fig.update_layout(
    title="L1 Norm of GPT2-Small Embedding Weights",
    xaxis_title="Token ID",
    yaxis_title="L1 Norm",
)
l1_norm_fig.show()

# %%
# Calculate L2 norm of the embedding weights
l2_norms = t.norm(W_E, p=2, dim=1).tolist()

l2_norm_fig = go.Figure(
    data=go.Scatter(
        x=list(range(len(l2_norms))),
        y=l2_norms,
        mode="markers",
        hoverinfo="text",
        text=token_labels,
    )
)
l2_norm_fig.update_layout(
    title="L2 Norm of GPT2-Small Embedding Weights",
    xaxis_title="Token ID",
    yaxis_title="L2 Norm",
)
l2_norm_fig.show()
