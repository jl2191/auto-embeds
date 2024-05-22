# %%
import torch as t
import transformer_lens as tl
from transformers import AutoTokenizer

# %%
gpt2 = tl.HookedTransformer.from_pretrained_no_processing("gpt2-small")

# %%
bloom = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")

# %%
gpt2_W_E = gpt2.W_E.clone().detach()
bloom_W_E = bloom.W_E.clone().detach()

print(gpt2_W_E.shape)
print(bloom_W_E.shape)

# %%
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
bloom_tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

# %%
del gpt2
del bloom
# %%
bloom_d_vocab = bloom_W_E.shape[0]
gpt2_d_vocab = gpt2_W_E.shape[0]
# %%
bloom_vocab_toks = [[i] for i in range(bloom_d_vocab)]
bloom_strings = bloom_tokenizer.batch_decode(bloom_vocab_toks)
# %%
gpt_vocab_toks = [[i] for i in range(gpt2_d_vocab)]
gpt2_strings = gpt2_tokenizer.batch_decode(gpt_vocab_toks)

# %%
# Convert lists to sets
bloom_set = set(bloom_strings)
gpt2_set = set(gpt2_strings)

# %%
print(len(bloom_set))
print(len(gpt2_set))
# Find common strings
common_vocab = bloom_set.intersection(gpt2_set)

# %%
common_strings = list(common_vocab)
print(len(common_strings))

# %%
print(*common_strings, sep="\n")

# %%
bloom_common_vocab_token_ids = [
    bloom_tokenizer.encode(token) for token in common_strings
]
gpt2_common_vocab_tokens_ids = [
    gpt2_tokenizer.encode(token) for token in common_strings
]

# %%
not_single_token = [token for token in bloom_common_vocab_token_ids if len(token) > 1]
print(not_single_token)
not_single_token_gpt2 = [
    token for token in gpt2_common_vocab_tokens_ids if len(token) > 1
]
print(not_single_token_gpt2)

# %%
print(bloom_tokenizer.batch_decode(not_single_token))
print(gpt2_tokenizer.batch_decode(not_single_token_gpt2))

# %%
bloom_str_tokens = [token for token in bloom_common_vocab_token_ids if len(token) == 1]
gpt2_str_tokens = [token for token in gpt2_common_vocab_tokens_ids if len(token) == 1]

# %%
print(gpt2_tokenizer.batch_decode(gpt2_str_tokens))
print(bloom_tokenizer.batch_decode(bloom_str_tokens))

# %%
bloom_str_tokens_flat = [item for sublist in bloom_str_tokens for item in sublist]
gpt2_str_tokens_flat = [item for sublist in gpt2_str_tokens for item in sublist]

print(bloom_str_tokens_flat)
print(gpt2_str_tokens_flat)

# %%
import plotly.express as px

# %%
# Plotting the correlation between bloom_str_tokens_flat and gpt2_str_tokens_flat using Plotly
fig = px.scatter(
    x=bloom_str_tokens_flat,
    y=gpt2_str_tokens_flat,
    labels={"x": "Bloom Token IDs", "y": "GPT-2 Token IDs"},
    title="Correlation between Bloom and GPT-2 Token IDs",
    trendline="ols",
    trendline_color_override="red",
)
fig.show()

# %%
# import statsmodels.api as sm

# Adding a constant to the bloom_str_tokens_flat for the intercept in the regression model
# X = sm.add_constant(bloom_str_tokens_flat)
# y = gpt2_str_tokens_flat

# # Fit the linear regression model
# model = sm.OLS(y, X).fit()

# # Retrieve the gradient (slope) and intercept from the model
# gradient, intercept = model.params
# print(f"Gradient (Slope): {gradient}, Intercept: {intercept}")


# %%
assert gpt2_tokenizer.batch_decode(
    gpt2_str_tokens_flat
) == bloom_tokenizer.batch_decode(bloom_str_tokens)

# %%
bloom_tokens = t.tensor(bloom_str_tokens_flat, device="cuda")
gpt2_tokens = t.tensor(gpt2_str_tokens_flat, device="cuda")
# %%
gpt2_common_embeds = gpt2_W_E[gpt2_tokens]
bloom_common_embeds = bloom_W_E[bloom_tokens]

# %%
import plotly.graph_objects as go
from fancy_einsum import einsum
from torch.utils.data import DataLoader, TensorDataset, random_split

from auto_embeds.utils.custom_tqdm import tqdm

# %%
transform = t.nn.utils.parametrizations.orthogonal(
    t.nn.Linear(768, 1024, bias=False, device="cuda")
)

dataset = TensorDataset(gpt2_common_embeds, bloom_common_embeds)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Utilizing DataLoader for batch processing
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print("Training dataset size:", len(train_dataset))
print("Testing dataset size:", len(test_dataset))

# %%
# Training the transformation model
criterion = t.nn.CosineSimilarity()
# criterion = t.nn.MSELoss()
optimizer = t.optim.Adam(transform.parameters(), lr=0.001)

# Ensure the model and data are on the same device to prevent runtime errors
device = t.device("cuda")  # Use "cuda" for NVIDIA GPUs or "cpu" for CPU

# Initialize lists to store loss values for plotting
train_losses = []
test_losses = []

# Training loop with tqdm for progress visualization
# for epoch in tqdm(range(100), desc="Epochs"):  # Number of epochs
for epoch in (epoch_pbar := tqdm(range(100))):
    epoch_losses = []
    for gpt2_embeds, bloom_embeds in train_loader:
        optimizer.zero_grad()
        # Move data to the appropriate device
        gpt2_embeds = gpt2_embeds.to(device)
        bloom_embeds = bloom_embeds.to(device)

        transformed_embeds = transform(gpt2_embeds)
        loss = -criterion(transformed_embeds, bloom_embeds).mean()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    epoch_loss = sum(epoch_losses) / len(epoch_losses)
    train_losses.append(epoch_loss)
    epoch_pbar.set_description(f"train loss: {epoch_loss:.3f}")

    # Evaluate on test data
    with t.no_grad():
        test_epoch_losses = []
        for gpt2_embeds, bloom_embeds in test_loader:
            gpt2_embeds = gpt2_embeds
            bloom_embeds = bloom_embeds
            transformed_embeds = transform(gpt2_embeds)
            loss = -criterion(transformed_embeds, bloom_embeds).mean()
            test_epoch_losses.append(loss.item())
        test_epoch_loss = sum(test_epoch_losses) / len(test_epoch_losses)
        test_losses.append(test_epoch_loss)

# Plotting the training and testing losses
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(1, 101)), y=train_losses, mode="lines+markers", name="Train Loss"
    )
)
fig.add_trace(
    go.Scatter(
        x=list(range(1, 101)), y=test_losses, mode="lines+markers", name="Test Loss"
    )
)
fig.update_layout(
    title="Train vs Test Loss per Epoch",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    legend_title="Loss Type",
)
fig.show()

# %%
# Transform GPT2 embeddings and unembed to check output strings
transformed_test_embeds = []
unembedded_strings = []
original_unembedded_strings = []
transform.eval()

with t.no_grad():
    for gpt2_embeds, _ in test_loader:
        gpt2_embeds = gpt2_embeds.to(device)
        transformed = transform(gpt2_embeds)
        transformed_test_embeds.append(transformed)
        unembedded = einsum(
            "batch d_model, d_vocab d_model -> batch d_vocab",
            transformed,
            bloom_W_E,
        )
        logits = unembedded.argmax(dim=-1)
        # most_similar_embeddings_dict = get_most_similar_embeddings(
        #     bloom_tokenizer, transformed
        # )
        # print_most_similar_embeddings_dict(most_similar_embeddings_dict)

        unembedded_strings.extend([bloom_tokenizer.decode(s) for s in logits.tolist()])

        # Compute original embeddings without transformation for comparison
        original_unembedded = einsum(
            "batch d_model, d_vocab d_model -> batch d_vocab",
            gpt2_embeds,
            gpt2_W_E,
        )
        original_logits = original_unembedded.argmax(dim=-1)
        original_unembedded_strings.extend(
            [gpt2_tokenizer.decode(s) for s in original_logits.tolist()]
        )

# %%
# Optionally, print or log the unembedded strings to check the outputs
for string, original_string in zip(unembedded_strings, original_unembedded_strings):
    print(f"Transformed: {string} | Original: {original_string}")

# Calculate the percentage of correct transformations
total_transformed = len(unembedded_strings)
correct_transformations = sum(
    1
    for transformed, original in zip(unembedded_strings, original_unembedded_strings)
    if transformed == original
)
percentage_correct = (correct_transformations / total_transformed) * 100
print(f"Percentage of Correct Transformations: {percentage_correct:.2f}%")
