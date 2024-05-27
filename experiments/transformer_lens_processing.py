# %%

import einops
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import torch as t
import transformer_lens as tl
from fancy_einsum import einsum

model_name = "bloom-560m"
no_processing = tl.HookedTransformer.from_pretrained_no_processing(model_name)
processing = tl.HookedTransformer.from_pretrained(model_name)

# %%
"""
what matrices are actually different?
"""
matrices = {
    "no_processing.W_E": no_processing.W_E.detach(),
    "no_processing.W_U": no_processing.W_U.detach().T,
    "processing.W_E": processing.W_E.detach(),
    "processing.W_U": processing.W_U.detach().T,
}

# Calculate difference
matrix_names = list(matrices.keys())

# Calculate cosine similarity
cosine_similarity_matrix = t.zeros((len(matrices), len(matrices)))
euclidean_distance_matrix = t.zeros((len(matrices), len(matrices)))
for i, mat1 in enumerate(matrix_names):
    for j, mat2 in enumerate(matrix_names):
        if i <= j:
            cosine_similarity_matrix[i, j] = t.tensor(float("nan"))
            euclidean_distance_matrix[i, j] = t.tensor(float("nan"))
        else:
            cosine_similarity_matrix[i, j] = t.nn.functional.cosine_similarity(
                matrices[mat1], matrices[mat2], dim=-1
            ).mean()
            # Calculate Euclidean distance
            euclidean_distance_matrix[i, j] = t.norm(
                matrices[mat1] - matrices[mat2], p=2
            )

# Move matrix to CPU for plotting
cosine_similarity_matrix = cosine_similarity_matrix.cpu()
euclidean_distance_matrix = euclidean_distance_matrix.cpu()

# Plot cosine similarity matrix
sns.heatmap(
    cosine_similarity_matrix,
    annot=True,
    xticklabels=matrix_names,
    yticklabels=matrix_names,
    cmap="coolwarm",
)
plt.title("Matrix Cosine Similarity")
plt.show()

# Plot Euclidean distance matrix
sns.heatmap(
    euclidean_distance_matrix,
    annot=True,
    xticklabels=matrix_names,
    yticklabels=matrix_names,
    cmap="viridis",
)
plt.title("Matrix Euclidean Distance")
plt.show()

# %%
"""
plotting the distribution of l2 norms for the W_E vectors to visualise what the
difference weight processing does.
"""

# Calculate L2 norms for vectors before and after processing for plotting and print the
# values
l2_norms_no_processing = t.norm(matrices["no_processing.W_E"], dim=-1).cpu()
print("L2 norms without processing:", l2_norms_no_processing)
l2_norms_processing = t.norm(matrices["processing.W_E"], dim=-1).cpu()
print("L2 norms with processing:", l2_norms_processing)

# Create a figure
fig = go.Figure()

# Add histograms for L2 norms of vectors before and after processing
fig.add_trace(
    go.Histogram(x=l2_norms_no_processing, name="No Processing", opacity=0.75)
)
fig.add_trace(go.Histogram(x=l2_norms_processing, name="Processing", opacity=0.75))

# Update layout for better visualization of L2 norms distribution
fig.update_layout(
    title_text="Distribution of L2 Norms of W_E Vectors",  # title of plot
    xaxis_title_text="L2 Norm Value",  # xaxis label
    yaxis_title_text="Count",  # yaxis label
    bargap=0.2,  # gap between bars of adjacent location coordinates
    bargroupgap=0.1,  # gap between bars of the same location coordinate
    barmode="overlay",  # bars are drawn on top of one another
)

# Show the figure
# fig.show()

"""
this i think shows that the W_E vectors for processing on average have a smaller
magnitude.
"""

# %%
"""
verify transformation from no processing to processing using centering of W_E weights
Extract the original and processed W_E matrices
"""

original_W_E = matrices["no_processing.W_E"].cpu()
processed_W_E = matrices["processing.W_E"].cpu()

# Calculate the mean of the original W_E matrix along the last dimension
original_W_E_mean = original_W_E.mean(-1, keepdim=True)

# Simulate the processing by subtracting the mean from the original W_E matrix
simulated_processed_W_E = original_W_E - original_W_E_mean

# Compare the simulated processed W_E with the actual processed W_E
comparison = t.allclose(processed_W_E, simulated_processed_W_E, atol=1e-6)

# Print the result of the comparison
print(f"Transformation verification result: {comparison}")

# %%
"""
test embedding and unembedding equivalence for both no_processing and processing models
"""
models = [("no_processing", no_processing), ("processing", processing)]
test_tokens = [1000, 1001]  # Test tokens with a batch size of 2

for model_name, model in models:
    # Embed tokens using W_E
    test_embeds = model.embed.W_E[test_tokens].unsqueeze(1)
    # shape [batch, pos, d_model]

    # Method 1: Compute logits / unembed using einsum
    logits_method1 = einops.einsum(
        test_embeds,
        model.embed.W_E,
        "batch pos d_model, vocab d_model -> batch pos vocab",
    )

    # Method 2: Directly use model's unembed method
    logits_method2 = model.unembed(test_embeds)

    # Compare logits from both methods to ensure equivalence
    if t.allclose(logits_method1, logits_method2, atol=1e-6):
        print(f"Logits comparison for {model_name}: True")
    else:
        # Calculate the maximum absolute difference
        max_diff = t.max(t.abs(logits_method1 - logits_method2)).item()
        # Calculate the percentage of mismatched elements
        total_elements = logits_method1.numel()
        mismatched_elements = t.ne(logits_method1, logits_method2).sum().item()
        mismatch_percentage = (mismatched_elements / total_elements) * 100
        print(
            f"Logits comparison for {model_name}: False, max difference: {max_diff}, "
            f"Mismatched elements: {mismatched_elements} / {total_elements} "
            f"({mismatch_percentage:.1f}%)"
        )

# %%
"""
okay so it seems that as a result of processing, as we are folding the value biases,
Check if all elements in no_processing.b_U are zeros and print the result
"""
b_U_zero_check_no_processing = t.all(no_processing.b_U == 0)
print(
    f"b_U for no processing contains only zero elements: {b_U_zero_check_no_processing}"
)

# Check if all elements in processing.b_U are zeros and print the result
b_U_zero_check_processing = t.all(processing.b_U == 0)
print(f"b_U for processing contains only zero elements: {b_U_zero_check_processing}")

# %%
"""
test embedding and unembedding equivalence for both no_processing and processing models
"""
models = [("no_processing", no_processing), ("processing", processing)]
test_tokens = [1000, 1001]  # Test tokens with a batch size of 2

for model_name, model in models:
    # Embed tokens using W_E
    test_embeds = model.embed.W_E[test_tokens].unsqueeze(1)
    # shape [batch, pos, d_model]

    # Method 1: Compute logits / unembed using einsum
    logits_method1 = einops.einsum(
        test_embeds,
        model.embed.W_E,
        "batch pos d_model, vocab d_model -> batch pos vocab",
    )

    # Method 2: Directly use model's unembed method
    logits_method2 = model.unembed(test_embeds)

    # Compare logits from both methods to ensure equivalence
    if t.allclose(logits_method1, logits_method2, atol=1e-6):
        print(f"Logits comparison for {model_name}: True")
    else:
        # Calculate the maximum absolute difference
        max_diff = t.max(t.abs(logits_method1 - logits_method2)).item()
        # Calculate the percentage of mismatched elements
        total_elements = logits_method1.numel()
        mismatched_elements = t.ne(logits_method1, logits_method2).sum().item()
        mismatch_percentage = (mismatched_elements / total_elements) * 100
        print(
            f"Logits comparison for {model_name}: False, max difference: {max_diff}, "
            f"Mismatched elements: {mismatched_elements} / {total_elements} "
            f"({mismatch_percentage:.1f}%)"
        )


"""
so it seems like processing.W_E is the result of subtracting away the mean of
no_processing.W_E
"""

# %%
"""
how is processing.W_U actually different from no_processing W_U?
in processed, it seems
"""

# Check if the state_dict keys of processing and no_processing are the same
processing_keys = set(processing.state_dict().keys())
no_processing_keys = set(no_processing.state_dict().keys())
# Compare the keys
keys_match = processing_keys == no_processing_keys
print(f"State_dict keys match between processing and no_processing: {keys_match}")
# If keys do not match, print the differences
if not keys_match:
    only_in_processing = processing_keys - no_processing_keys
    only_in_no_processing = no_processing_keys - processing_keys
    print("Keys only in processing:")
    for key in only_in_processing:
        print(key)
    print("Keys only in no_processing:")
    for key in only_in_no_processing:
        print(key)

# %%
"""
is embed.ln.w and embed.ln.b different between processings?
"""
# Extract layer norm weights and biases from both models
ln_weights_processing = processing.state_dict()["embed.ln.w"].data
ln_bias_processing = processing.state_dict()["embed.ln.b"].data
ln_weights_no_processing = no_processing.state_dict()["embed.ln.w"].data
ln_bias_no_processing = no_processing.state_dict()["embed.ln.b"].data

# Check if LN weights and biases are the same between processing and no_processing
weights_match = t.allclose(ln_weights_processing, ln_weights_no_processing, atol=1e-6)
biases_match = t.allclose(ln_bias_processing, ln_bias_no_processing, atol=1e-6)

# Print the comparison results
print(f"Layer norm weights match: {weights_match}")
print(f"Layer norm biases match: {biases_match}")

# If weights or biases do not match, calculate and print the maximum absolute difference
if not weights_match:
    max_diff_weights = t.max(
        t.abs(ln_weights_processing - ln_weights_no_processing)
    ).item()
    print(f"Max difference in layer norm weights: {max_diff_weights}")
if not biases_match:
    max_diff_biases = t.max(t.abs(ln_bias_processing - ln_bias_no_processing)).item()
    print(f"Max difference in layer norm biases: {max_diff_biases}")

"""
no apparently
"""

# %%
"""
checking if embed() always the same as doing embed.W_E @ input?
"""
# Generate a random batch of token IDs with shape (batch, pos)
# Assuming 'vocab_size' is the vocabulary size
token_ids = [1000, 1001]  # Test tokens with a batch size of 2

# Loop to compare embed() method vs manual embedding lookup and layernorm for token IDs
for model, name in [(processing, "processing"), (no_processing, "no_processing")]:
    # Using the model's embed method
    embedded_method = model.embed(token_ids)
    # Manual embedding lookup
    manual_embed = model.embed.W_E[token_ids]

    # Nested loop to compare layernormed manual embedding with embed() method
    for layernorm in [True, False]:
        if layernorm:
            # Apply layernorm to manual embedding
            manual_embed_layernormed = t.nn.functional.layer_norm(
                manual_embed, normalized_shape=[manual_embed.size(-1)]
            )
            manual_embed_layernormed = (
                manual_embed_layernormed * model.embed.ln.w.data + model.embed.ln.b.data
            )
            comparison_embed = manual_embed_layernormed
            comparison_name = f"{name} layernormed"
        else:
            comparison_embed = manual_embed
            comparison_name = name

        # Check if the results are close
        if t.allclose(embedded_method, comparison_embed):
            print(f"Embedding comparison for {comparison_name}: True")
        else:
            # Calculate the maximum absolute difference
            max_diff = t.max(t.abs(embedded_method - comparison_embed)).item()
            # Calculate the percentage of mismatched elements
            total_elements = embedded_method.numel()
            mismatched_elements = t.ne(embedded_method, comparison_embed).sum().item()
            mismatch_percentage = (mismatched_elements / total_elements) * 100
            print(
                f"Embedding comparison for {comparison_name}: False, "
                f"max difference: {max_diff}, "
                f"Mismatched elements: {mismatched_elements} / {total_elements} "
                f"({mismatch_percentage:.1f}%)"
            )

# %%
"""
check if the embed and unembed methods produce equivalent results between processing and
no_processing models
"""
# Generate a random batch of token IDs
token_ids = [1000, 1001]

# Embed the tokens using both models' embed method
embedded_tokens_processing = processing.embed(token_ids)
embedded_tokens_no_processing = no_processing.embed(token_ids)
if t.allclose(embedded_tokens_processing, embedded_tokens_no_processing):
    print("Embedding results match between processing and no_processing models.")
else:
    # Calculate the maximum absolute difference
    max_diff_embed = t.max(
        t.abs(embedded_tokens_processing - embedded_tokens_no_processing)
    ).item()
    # Calculate the percentage of mismatched elements
    total_elements_embed = embedded_tokens_processing.numel()
    mismatched_elements_embed = (
        t.ne(embedded_tokens_processing, embedded_tokens_no_processing).sum().item()
    )
    mismatch_percentage_embed = (mismatched_elements_embed / total_elements_embed) * 100
    print(
        f"Embedding results do not match: max difference: {max_diff_embed}, "
        f"Mismatched elements: {mismatched_elements_embed} / {total_elements_embed} "
        f"({mismatch_percentage_embed:.1f}%)"
    )

# Unembed the tokens using both models' unembed method
unembedded_tokens_processing = processing.unembed(
    embedded_tokens_processing.unsqueeze(0)
)
unembedded_tokens_no_processing = no_processing.unembed(
    embedded_tokens_processing.unsqueeze(0)
)
# Check if the unembedded tokens are equivalent for processing and no_processing models
if t.allclose(unembedded_tokens_processing, unembedded_tokens_no_processing):
    print("Unembedding results match between processing and no_processing models.")
else:
    # Calculate the maximum absolute difference
    max_diff_unembed = t.max(
        t.abs(unembedded_tokens_processing - unembedded_tokens_no_processing)
    ).item()
    # Calculate the percentage of mismatched elements
    total_elements_unembed = unembedded_tokens_processing.numel()
    mismatched_elements_unembed = (
        t.ne(unembedded_tokens_processing, unembedded_tokens_no_processing).sum().item()
    )
    mismatch_percentage_unembed = (
        mismatched_elements_unembed / total_elements_unembed
    ) * 100
    print(
        f"Unembedding results do not match: max difference: {max_diff_unembed}, "
        f"Mismatched elems: {mismatched_elements_unembed} / {total_elements_unembed} "
        f"({mismatch_percentage_unembed:.1f}%)"
    )

# %%
"""
seems like unembedding methods for the two different models produce different results.
i think this is because we have centered weights writing to the residual stream and
so the logits are different. however, i think the softmaxes would be the same
"""

# Check if the unembedded tokens are equivalent for processing and no_processing models
if t.allclose(
    unembedded_tokens_processing.softmax(dim=-1),
    unembedded_tokens_no_processing.softmax(dim=-1),
):
    print("Unembedding results match between processing and no_processing models.")
else:
    # Calculate the maximum absolute difference
    max_diff_unembed = t.max(
        t.abs(
            unembedded_tokens_processing.softmax(dim=-1)
            - unembedded_tokens_no_processing.softmax(dim=-1)
        )
    ).item()
    # Calculate the percentage of mismatched elements
    total_elements_unembed = unembedded_tokens_processing.numel()
    mismatched_elements_unembed = (
        t.ne(
            unembedded_tokens_processing.softmax(dim=-1),
            unembedded_tokens_no_processing.softmax(dim=-1),
        )
        .sum()
        .item()
    )
    mismatch_percentage_unembed = (
        mismatched_elements_unembed / total_elements_unembed
    ) * 100
    print(
        f"Unembedding results do not match: max difference: {max_diff_unembed}, "
        f"Mismatched elements: {mismatched_elements_unembed}/{total_elements_unembed} "
        f"({mismatch_percentage_unembed:.1f}%)"
    )

"""
okay, the difference now is a lot less!
still unsure about why there is still a difference though... like the greatest relative
difference is quite big. these i think should really be the same!
"""

# %%
t.testing.assert_close(
    unembedded_tokens_processing.softmax(dim=-1),
    unembedded_tokens_no_processing.softmax(dim=-1),
)
"""
huh, is this close enough?
"""

# %%
"""
for the embedding, does model.embed do:
 index into W_E
 layernorm (using ln.w and ln.b?)

"""
token_ids = [1000, 1001, 10030, 1]
a = no_processing.embed.W_E[token_ids]
a = t.nn.functional.layer_norm(
    a, [a.shape[-1]], no_processing.embed.ln.w, no_processing.embed.ln.b
)

b = no_processing.embed(token_ids)

t.testing.assert_close(a, b)

"""
oh wow yay, that actually worked!
"""

# %%
"""
and the same works for the model with processing?
"""
a = processing.embed.W_E[token_ids]
a = t.nn.functional.layer_norm(
    a, [a.shape[-1]], processing.embed.ln.w, processing.embed.ln.b
)

b = processing.embed(token_ids)

t.testing.assert_close(a, b)

"""
okay, good stuff.
"""

# %%
"""
now for the unembedding, my guess is that for processing, model.unembed does:
"""
embed = processing.embed(token_ids).unsqueeze(1)

a = processing.unembed(embed)

b = (
    einsum(
        "batch pos d_model, d_model vocab -> batch pos vocab",
        embed,
        processing.unembed.W_U,
    )
    + processing.unembed.b_U
)

t.testing.assert_close(a, b)
"""
it does!
"""

# %%
"""
my guess for no_processing, model.embed only does the multiplication with W_U and
addition with b_U. as such, for both of these to get the same result, we need to
layernorm no_processing first
"""
a = embed
a = a @ no_processing.unembed.W_U + no_processing.unembed.b_U
a = a.softmax(-1)

b = embed
b = no_processing.unembed(b)
b = b.softmax(-1)

t.testing.assert_close(a, b)
"""
this passes!
"""

# %%
# %%
"""
okay now how can i get no_processing.unembed to be the same as processing.unembed?
"""
token_ids = [100, 101, 102]
embed = no_processing.embed(token_ids).unsqueeze(1)
a = embed.clone().detach()
a = t.nn.functional.layer_norm(
    a, [a.shape[-1]], no_processing.ln_final.w, no_processing.ln_final.b
)
a = a @ (
    no_processing.unembed.W_U - no_processing.unembed.W_U.mean(-1, keepdim=True)
) + (no_processing.unembed.b_U - no_processing.unembed.b_U.mean(-1, keepdim=True))
a = a.softmax(-1)

b = embed.clone().detach()
b = t.nn.functional.layer_norm(b, [b.shape[-1]])
b = (
    einsum(
        "batch pos d_model, d_model vocab -> batch pos vocab",
        b,
        processing.unembed.W_U,
    )
    + processing.unembed.b_U
)
b = b.softmax(-1)
t.testing.assert_close(a, b)
"""
woop woop
"""

# %%
"""
and just to check that we can use ln_final.w and ln_final.b outside of
t.nn.function.layernorm.
"""
a = embed.clone().detach()
a = t.nn.functional.layer_norm(
    a, [a.shape[-1]], no_processing.ln_final.w, no_processing.ln_final.b
)
a = a @ no_processing.unembed.W_U + no_processing.unembed.b_U

b = embed.clone().detach()
b = t.nn.functional.layer_norm(b, [b.shape[-1]])
print(b.shape)
b = (
    einsum(
        "batch pos d_model, d_model -> batch pos d_model",
        b,
        no_processing.ln_final.w,
    )
    + no_processing.ln_final.b
)

b = b @ no_processing.unembed.W_U + no_processing.unembed.b_U

t.testing.assert_close(a, b)
