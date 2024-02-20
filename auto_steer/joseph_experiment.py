# %%
import torch as t
import transformer_lens as tl

from auto_steer.steering_utils import (
    calc_cos_sim_acc,
    create_data_loaders,
    generate_embeddings,
    generate_tokens,
    initialize_transform_and_optim,
    load_test_strings,
    mean_vec,
    perform_translation_tests,
    train_and_evaluate_transform,
)
from auto_steer.utils.misc import (
    repo_path_to_abs_path,
)

# %% model setup
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
model = tl.HookedTransformer.from_pretrained_no_processing("bloom-3b")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
cache_folder = repo_path_to_abs_path("datasets/activation_cache")

#%% ------------------------------------------------------------------------------------
# joseph experiment - learn rotation
en_toks, fr_toks = generate_tokens(model, n_toks, device)
en_embeds, fr_embeds = generate_embeddings(model, en_toks, fr_toks)
train_loader, test_loader = create_data_loaders(
    en_embeds, fr_embeds, train_ratio=0.99, batch_size=512
)
initial_rotation, optim = initialize_transform_and_optim(
    d_model, transformation="rotation", lr=0.0002, device=device
)
learned_rotation = train_and_evaluate_transform(
    model, train_loader, test_loader, initial_rotation, optim, 50, device
)
print("Test Accuracy:", calc_cos_sim_acc(test_loader, learned_rotation))

# %% joseph experiment - generate fr en bloom embed data for europarl dataset

# en_file = f"{datasets_folder}/europarl/europarl-v7.fr-en.en"
# fr_file = f"{datasets_folder}/europarl/europarl-v7.fr-en.fr"

# batch_size = 2
# layers = [20, 25, 27, 29]

# # Read the first 5000 lines of the files (excluding the first line)
# with open(en_file, "r") as f:
#     en_strs_list = [f.readline()[:-1] + " " + f.readline()[:-1] for _ in range(5001)][
#         1:
#     ]
# with open(fr_file, "r") as f:
#     fr_strs_list = [f.readline()[:-1] + " " + f.readline()[:-1] for _ in range(5001)][
#         1:
#     ]

# en_toks, en_attn_mask = tokenize_texts(model, en_strs_list)
# fr_toks, fr_attn_mask = tokenize_texts(model, fr_strs_list)

# train_loader = create_data_loaders(
#     en_toks, fr_toks, batch_size=2, en_attn_mask=en_attn_mask,
# fr_attn_mask=fr_attn_mask
# )
# en_acts, fr_acts = run_and_gather_acts(model, train_loader, layers)

# cache_folder = repo_path_to_abs_path("datasets/activation_cache")
# filename_base = "bloom-3b-2"
# save_acts(datasets_folder, filename_base, en_acts, fr_acts)

# %% joseph experiment - load fr en embed data

en_file = f"{datasets_folder}/europarl/europarl-v7.fr-en.en"
fr_file = f"{datasets_folder}/europarl/europarl-v7.fr-en.fr"

train_en_resids = t.load(
    f"{cache_folder}/europarl_v7_fr_en_double_prompt_all_toks-"
    f"{model.cfg.model_name}-lyrs_[20, 25, 27, 29]-en.pt"
)
train_fr_resids = t.load(
    f"{cache_folder}/europarl_v7_fr_en_double_prompt_all_toks-"
    f"{model.cfg.model_name}-lyrs_[20, 25, 27, 29]-fr.pt"
)

# %%% joseph experiment - train fr en embedding rotation

layer_idx = 20
gen_length = 20

fr_to_en_data_loader = create_data_loaders(
    train_en_resids[layer_idx],
    train_fr_resids[layer_idx],
    batch_size=512,
    match_dims=True,
)
fr_to_en_mean_vec = mean_vec(train_en_resids[layer_idx], train_fr_resids[layer_idx])

test_en_strs = load_test_strings(en_file, skip_lines=1000)
test_fr_strs = load_test_strings(fr_file, skip_lines=1000)

perform_translation_tests(
    model, test_en_strs, test_fr_strs, layer_idx, gen_length, learned_rotation
)
