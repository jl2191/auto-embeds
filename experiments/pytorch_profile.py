# %%
import os

from auto_embeds.metrics import calc_cos_sim_acc, evaluate_accuracy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


import numpy as np
import torch as t
import transformer_lens as tl
from IPython.core.getipython import get_ipython
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.embed_utils import (
    initialize_loss,
    initialize_transform_and_optim,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.misc import repo_path_to_abs_path

np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)
try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("load_ext", "line_profiler")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass


t.backends.cuda.matmul.allow_tf32 = True


# %% model setup
# model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
model_caches_folder = repo_path_to_abs_path("datasets/model_caches")
token_caches_folder = repo_path_to_abs_path("datasets/token_caches")

# %% -----------------------------------------------------------------------------------
# file_path = f"{datasets_folder}/wikdict/2_extracted/eng-fra.json"
# with open(file_path, "r") as file:
#     word_pairs = json.load(file)
# random.seed(1)
# random.shuffle(word_pairs)
# split_index = int(len(word_pairs) * 0.95)
# train_en_fr_pairs = word_pairs[:split_index]
# test_en_fr_pairs = word_pairs[split_index:]

# train_word_pairs = filter_word_pairs(
#     model,
#     train_en_fr_pairs,
#     discard_if_same=True,
#     min_length=4,
#     # capture_diff_case=True,
#     capture_space=True,
#     capture_no_space=True,
#     print_pairs=True,
#     print_number=True,
#     max_token_id=100_000,
#     # most_common_english=True,
#     # most_common_french=True,
# )

# test_word_pairs = filter_word_pairs(
#     model,
#     test_en_fr_pairs,
#     discard_if_same=True,
#     min_length=4,
#     # capture_diff_case=True,
#     capture_space=True,
#     capture_no_space=True,
#     # print_pairs=True,
#     print_number=True,
#     max_token_id=100_000,
#     # most_common_english=True,
#     # most_common_french=True,
# )

# train_en_toks, train_fr_toks, train_en_mask, train_fr_mask = tokenize_word_pairs(
#     model, train_word_pairs
# )
# test_en_toks, test_fr_toks, test_en_mask, test_fr_mask = tokenize_word_pairs(
#     model, test_word_pairs
# )
# # %%
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
# %%
train_data = t.load(f"{token_caches_folder}/wikdict-train-en-fr-tokens.pt")
test_data = t.load(f"{token_caches_folder}/wikdict-test-en-fr-tokens.pt")

train_en_toks = train_data["en_toks"]
train_fr_toks = train_data["fr_toks"]
train_en_mask = train_data["en_mask"]
train_fr_mask = train_data["fr_mask"]

test_en_toks = test_data["en_toks"]
test_fr_toks = test_data["fr_toks"]
test_en_mask = test_data["en_mask"]
test_fr_mask = test_data["fr_mask"]

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

train_dataset = TensorDataset(train_en_embeds, train_fr_embeds)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = TensorDataset(test_en_embeds, test_fr_embeds)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
# %%
# run = wandb.init(
#     project="single_token_tests",
# )
# %%

transformation_name = "rotation"
transform, optim = initialize_transform_and_optim(
    d_model,
    transformation=transformation_name,
    # optim_kwargs={"lr": 2e-4},
    optim_kwargs={"lr": 2e-4},
)
transform = transform
loss_module = initialize_loss("cosine_similarity")

n_epochs = 15
loss_history = {"train_loss": [], "test_loss": []}

with t.profiler.profile(
    schedule=t.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
    on_trace_ready=t.profiler.tensorboard_trace_handler("./log/rotation"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    # with profile(activities=[
    #     ProfilerActivity.CPU,
    #     ProfilerActivity.CUDA
    # ]) as prof:
    for epoch in (epoch_pbar := tqdm(range(n_epochs))):
        for batch_idx, (en_embed, fr_embed) in enumerate(train_loader):
            optim.zero_grad()  # type: ignore
            pred = transform(en_embed)
            train_loss = loss_module(pred.squeeze(), fr_embed.squeeze())
            info_dict = {
                "train_loss": train_loss.item(),
                "batch": batch_idx,
                "epoch": epoch,
            }
            loss_history["train_loss"].append(info_dict)
            train_loss.backward()
            optim.step()  # type: ignore
            prof.step()
            epoch_pbar.set_description(f"train loss: {train_loss.item():.3f}")
# prof.export_chrome_trace("trace.json")

print(f"{transformation_name}:")
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

# %%
print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cuda_time_total", row_limit=5
    )
)
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
