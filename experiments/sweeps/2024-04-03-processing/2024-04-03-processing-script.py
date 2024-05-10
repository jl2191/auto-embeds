# %%
import itertools
import json
import os

import neptune
import numpy as np
import torch as t
import transformer_lens as tl

from auto_embeds.data import filter_word_pairs, get_dataset_path
from auto_embeds.embed_utils import (
    initialize_loss,
    initialize_transform_and_optim,
    train_transform,
)
from auto_embeds.metrics import (
    evaluate_accuracy,
    mark_translation,
)
from auto_embeds.verify import (
    plot_cosine_similarity_trend,
    prepare_verify_analysis,
    prepare_verify_datasets,
    test_cos_sim_difference,
    verify_transform,
)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Seed for reproducibility
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)
try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("load_ext", "line_profiler")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass

# Configuration for experiments
config = {
    "neptune": {
        "notes": "instead of just selecting top k based on source word pair, will now \
            will now select an entire given word pair based on whether it is source or \
            target language that is closest.",
        "tags": [
            "2024-04-02",
            "actual",
            # "test",
            # "very test",
            # "new_top_k_algorithm",
            # "processing",
        ],
    },
    "models": [
        "bloom-560m",
        "bloom-3b",
        # "bloom-7b",
    ],
    "processings": [
        True,
        False,
    ],
    "layernorms": [
        # True,
        False,
    ],
    "datasets": [
        {
            "name": "wikdict_en_fr_extracted",
            "min_length": 5,
            "capture_space": True,
            "capture_no_space": False,
            "mark_accuracy_path": "wikdict_en_fr_azure_validation",
        },
        {
            "name": "muse_en_fr_extracted",
            "min_length": 5,
            "capture_space": True,
            "capture_no_space": False,
            "mark_accuracy_path": "muse_en_fr_azure_validation",
        },
        {
            "name": "cc_cedict_zh_en_extracted",
            "min_length": 2,
            "capture_space": False,
            "capture_no_space": True,
            "mark_accuracy_path": "cc_cedict_zh_en_azure_validation",
        },
        {
            "name": "muse_zh_en_extracted_train",
            "min_length": 2,
            "capture_space": False,
            "capture_no_space": True,
            "mark_accuracy_path": "muse_zh_en_azure_validation",
        },
    ],
    "transformations": [
        # "identity",
        # "translation",
        "linear_map",
        # "biased_linear_map",
        # "uncentered_linear_map",
        # "biased_uncentered_linear_map",
        "rotation",
        # "biased_rotation",
        # "uncentered_rotation",
        # "rotation_translation",
    ],
    "seeds": [1, 2, 3],
    "n_epochs": [100],
    "lr": [8e-5],
    "weight_decay": [
        0,
        #  2e-5,
    ],
    "train_batch_sizes": [128],
    "test_batch_sizes": [256],
    "top_k": [200],
    "top_k_selection_methods": [
        "src_and_src",
        # "tgt_and_tgt",
        # "top_src",
        # "top_tgt",
    ],
}

total_runs = 1
for value in config.values():
    if isinstance(value, list):
        total_runs *= len(value)

print(f"Total runs to be executed: {total_runs}")

# %%
# Main experiment loop
last_loaded_model = None
last_loaded_word_pairs = None
model = None
for (
    model_name,
    processing,
    layernorm,
    dataset_config,
    transformation,
    n_epoch,
    lr,
    weight_decay,
    train_batch_size,
    test_batch_size,
    top_k,
    top_k_selection_method,
    seed,
) in itertools.product(
    config["models"],
    config["processings"],
    config["layernorms"],
    config["datasets"],
    config["transformations"],
    config["n_epochs"],
    config["lr"],
    config["weight_decay"],
    config["train_batch_sizes"],
    config["test_batch_sizes"],
    config["top_k"],
    config["top_k_selection_methods"],
    config["seeds"],
):
    # Model setup
    model_needs_loading = (
        last_loaded_model is None
        or last_loaded_model["model_name"] != model_name
        or last_loaded_model["processing"] != processing
    )
    if model_needs_loading:
        if processing:
            model = tl.HookedTransformer.from_pretrained(model_name)
        else:
            model = tl.HookedTransformer.from_pretrained_no_processing(model_name)
        last_loaded_model = {
            "model_name": model_name,
            "processing": processing,
            "model": model,
        }

    assert model is not None, "The model has not been loaded successfully."

    d_model = model.cfg.d_model
    n_toks = model.cfg.d_vocab_out

    # Dataset filtering
    dataset_name = dataset_config["name"]
    file_path = get_dataset_path(dataset_name)
    with open(file_path, "r") as file:
        word_pairs = json.load(file)

    word_pairs_needs_loading = (
        last_loaded_word_pairs is None
        or last_loaded_word_pairs["dataset_config"] != dataset_config
    )
    if word_pairs_needs_loading:
        all_word_pairs = filter_word_pairs(
            model,
            word_pairs,
            discard_if_same=True,
            min_length=dataset_config["min_length"],
            capture_space=dataset_config["capture_space"],
            capture_no_space=dataset_config["capture_no_space"],
            print_number=True,
            verbose_count=True,
        )
        last_loaded_word_pairs = {
            "dataset_config": dataset_config,
            "filtered_word_pairs": all_word_pairs,
        }
        all_word_pairs = last_loaded_word_pairs["filtered_word_pairs"]

    # Prepare datasets
    verify_learning = prepare_verify_analysis(
        model=model,
        all_word_pairs=all_word_pairs,
        seed=seed,
        keep_other_pair=True,
    )
    print(seed)
    print(verify_learning.src.selected.words)

    train_loader, test_loader = prepare_verify_datasets(
        verify_learning=verify_learning,
        batch_sizes=(train_batch_size, test_batch_size),
        top_k=top_k,
        top_k_selection_method=top_k_selection_method,
    )
    for item in test_loader:
        en_embed, fr_embed = item
        en_probs = model.unembed(en_embed).squeeze(0).softmax(-1).argmax(dim=-1)
        fr_probs = model.unembed(fr_embed).squeeze(0).softmax(-1).argmax(dim=-1)
        en_str = model.to_string(en_probs)
        fr_str = model.to_string(fr_probs)
        print("English Embedding String Representation:", en_str)
        print("French Embedding String Representation:", fr_str)
        break

    azure_translations_path = get_dataset_path(dataset_config["mark_accuracy_path"])

    run_config = {
        "model_name": model_name,
        "processing": processing,
        "layernorm": layernorm,
        "dataset_name": dataset_name,
        **dataset_config,
        "transformation": transformation,
        "seed": seed,
        "n_epoch": n_epoch,
        "lr": lr,
        "weight_decay": weight_decay,
        "train_batch_size": train_batch_size,
        "test_batch_size": test_batch_size,
        "top_k": top_k,
    }

    # neptune setup
    run = neptune.init(
        project="language-transformations",
        config=run_config,
        notes=config["neptune"]["notes"],
        tags=config["neptune"]["tags"],
    )

    # Initialize transformation and optimizer
    transform, optim = initialize_transform_and_optim(
        d_model,
        transformation=transformation,
        optim_kwargs={"lr": lr, "weight_decay": weight_decay},
    )
    loss_module = initialize_loss("cosine_similarity")

    # Train transformation
    if optim is not None:
        transform, loss_history = train_transform(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            transform=transform,
            optim=optim,
            loss_module=loss_module,
            n_epochs=n_epoch,
            plot_fig=False,
            neptune=neptune,
            azure_translations_path=azure_translations_path,
        )

    # Evaluate and log results
    accuracy = evaluate_accuracy(
        model,
        test_loader,
        transform,
        exact_match=False,
        print_results=False,
        print_top_preds=False,
    )

    mark_translation_acc = mark_translation(
        model=model,
        transformation=transform,
        test_loader=test_loader,
        azure_translations_path=azure_translations_path,
        print_results=False,
    )

    verify_results_dict = verify_transform(
        model=model,
        transformation=transform,
        test_loader=test_loader,
    )

    cos_sims_trend_plot = plot_cosine_similarity_trend(verify_results_dict)

    # cos_sims_trend_plot.show(config={"responsive": True, "autosize": True})

    test_cos_sim_diff = test_cos_sim_difference(verify_results_dict)

    neptune.log(
        {
            "mark_translation_acc": mark_translation_acc,
            "cos_sims_trend_plot": cos_sims_trend_plot,
            "test_cos_sim_diff": test_cos_sim_diff,
            # "model.cfg": model.cfg,
        }
    )

    neptune.finish()
