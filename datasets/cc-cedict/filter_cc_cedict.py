# %%
import json
import random

import torch as t
import transformer_lens as tl
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.data import filter_word_pairs, tokenize_word_pairs
from auto_embeds.embed_utils import (
    initialize_loss,
    initialize_transform_and_optim,
    train_transform,
)
from auto_embeds.metrics import calc_cos_sim_acc, evaluate_accuracy
from auto_embeds.utils.misc import repo_path_to_abs_path

model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")

device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out

# %%
with open(
    repo_path_to_abs_path("datasets/cc-cedict/cc-cedict-zh-en-parsed.json"),
    "r",
    encoding="utf-8",
) as file:
    word_pairs = json.load(file)
print(f"Loaded {len(word_pairs)} entries from the dictionary.")
# for word_pair in word_pairs[:300]:
#     print(word_pair)


# %%
word_pairs = filter_word_pairs(
    model,
    word_pairs,
    discard_if_same=True,
    min_length=2,
    # capture_diff_case=True,
    # capture_space=True,
    capture_no_space=True,
    print_pairs=True,
    print_number=True,
    verbose_count=True,
    # max_token_id=200_000,
    # most_common_english=True,
    # most_common_french=True,
    # acceptable_english_overlap=0.9,
    # acceptable_french_overlap=0.8,
)
filtered_save_path = repo_path_to_abs_path(
    "datasets/cc-cedict/cc-cedict-zh-en-filtered.json"
)
with open(filtered_save_path, "w") as f:
    json.dump(word_pairs, f, ensure_ascii=False, indent=4)

random.seed(1)
# all_en_fr_pairs.sort(key=lambda pair: pair[0])
random.shuffle(word_pairs)
split_index = int(len(word_pairs) * 0.95)
train_en_fr_pairs = word_pairs[:split_index]
test_en_fr_pairs = word_pairs[split_index:]

train_en_toks, train_fr_toks, _, _ = tokenize_word_pairs(model, train_en_fr_pairs)
test_en_toks, test_fr_toks, _, _ = tokenize_word_pairs(model, test_en_fr_pairs)

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

print(train_en_embeds.shape)
print(test_en_embeds.shape)
train_dataset = TensorDataset(train_en_embeds, train_fr_embeds)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(test_en_embeds, test_fr_embeds)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

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
        # optim_kwargs={"lr": 1e-4, "weight_decay": 1e-5},
        # optim_kwargs={"lr": 1e-4}
        optim_kwargs={"lr": 2e-4},
        # optim_kwargs={"lr": 1e-4, "weight_decay": 1e-5}
        # optim_kwargs={"lr": 5e-4}
        # optim_kwargs={"lr": 2e-4},
        # optim_kwargs={"lr": 8e-5}
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
            # neptune=neptune,
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

    if transformation_name == "rotation":
        with t.no_grad():
            t.testing.assert_close(
                transform(t.eye(d_model, device=device)).det().abs(), 1
            )

# %%
