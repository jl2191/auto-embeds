# %%
import datetime
import itertools
import json
import os

import numpy as np
import torch as t
import transformer_lens as tl
import wandb

from auto_embeds.data import filter_word_pairs, get_dataset_path
from auto_embeds.embed_utils import (
    initialize_embed_and_unembed,
    initialize_loss,
    initialize_transform_and_optim,
    train_transform,
)
from auto_embeds.metrics import (
    evaluate_accuracy,
    mark_translation,
)
from auto_embeds.utils.cache import auto_embeds_cache
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.misc import get_experiment_worker_config, is_notebook
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
os.environ["AUTOEMBEDS_CACHING"] = "true"

# Seed for reproducibility
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

# Configuration for overall experiments
experiment_config = {
    "wandb": {
        "notes": "blank",
        "tags": [
            f"{datetime.datetime.now():%Y-%m-%d}",
            f"{datetime.datetime.now():%Y-%m-%d} analytical solutions",
            "run group 1",
            # "actual",
            # "test",
        ],
    },
    "models": [
        "bloom-560m",
        # "bloom-3b",
        # "bloom-7b",
    ],
    "processings": [
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
        # {
        #     "name": "random_word_pairs",
        #     "min_length": 2,
        #     "capture_space": True,
        #     "capture_no_space": False,
        # },
        # {
        #     "name": "singular_plural_pairs",
        #     "min_length": 2,
        #     "capture_space": True,
        #     "capture_no_space": False,
        # },
        # {
        #     "name": "muse_en_fr_extracted",
        #     "min_length": 5,
        #     "capture_space": True,
        #     "capture_no_space": False,
        #     "mark_accuracy_path": "muse_en_fr_azure_validation",
        # },
        # {
        #     "name": "cc_cedict_zh_en_extracted",
        #     "min_length": 2,
        #     "capture_space": False,
        #     "capture_no_space": True,
        #     "mark_accuracy_path": "cc_cedict_zh_en_azure_validation",
        # },
        # {
        #     "name": "muse_zh_en_extracted_train",
        #     "min_length": 2,
        #     "capture_space": False,
        #     "capture_no_space": True,
        #     "mark_accuracy_path": "muse_zh_en_azure_validation",
        # },
    ],
    "transformations": [
        # "identity",
        # "translation",
        "linear_map",
        # "biased_linear_map",
        # "uncentered_linear_map",
        # "biased_uncentered_linear_map",
        # "rotation",
        # "biased_rotation",
        # "uncentered_rotation",
        # "analytical_rotation",
        # "analytical_translation",
    ],
    "train_batch_sizes": [128],
    "test_batch_sizes": [256],
    "top_k": [200],
    "top_k_selection_methods": [
        # "src_and_src",
        # "tgt_and_tgt",
        "top_src",
        # "top_tgt",
    ],
    "seeds": [1],
    "embed_apply_ln": [True, False],
    "embed_apply_ln_weights": [True],
    "transform_apply_ln": [True],
    "unembed_apply_ln": [True, False],
    "unembed_apply_ln_weights": [True],
    "n_epochs": [100],
    "weight_decay": [
        0,
        # 2e-5,
    ],
    "lr": [8e-5],
}

total_runs = 1
for value in experiment_config.values():
    if isinstance(value, list):
        total_runs *= len(value)

print(f"Total experiment runs calculated: {total_runs}")


# %%
config_dict = get_experiment_worker_config(
    experiment_config=experiment_config,
    split_parameter="datasets",
    n_splits=1,
    worker_id=0,
)

wandb_config = config_dict.pop("wandb")
config_values = [
    config_dict[entry] if entry != "datasets" else config_dict[entry]
    for entry in config_dict
]
config_list = list(itertools.product(*config_values))

for (
    model_name,
    processing,
    dataset_config,
    transformation,
    train_batch_size,
    test_batch_size,
    top_k,
    top_k_selection_method,
    seed,
    embed_apply_ln,
    embed_apply_ln_weights,
    transform_apply_ln,
    unembed_apply_ln,
    unembed_apply_ln_weights,
    n_epoch,
    weight_decay,
    lr,
) in tqdm(config_list, total=len(config_list)):

    run_config = {
        "model_name": model_name,
        "processing": processing,
        "dataset": dataset_config,
        "transformation": transformation,
        "train_batch_size": train_batch_size,
        "test_batch_size": test_batch_size,
        "top_k": top_k,
        "top_k_selection_method": top_k_selection_method,
        "seed": seed,
        "embed_apply_ln": embed_apply_ln,
        "embed_apply_ln_weights": embed_apply_ln_weights,
        "transform_apply_ln": transform_apply_ln,
        "unembed_apply_ln": unembed_apply_ln,
        "unembed_apply_ln_weights": unembed_apply_ln_weights,
        "n_epoch": n_epoch,
        "weight_decay": weight_decay,
        "lr": lr,
    }

    # WandB run init
    run = wandb.init(
        project="language-transformations",
        config=run_config,
        notes=wandb_config["notes"],
        tags=wandb_config["tags"],
    )

    @auto_embeds_cache
    def load_model(processing, model_name):
        if processing:
            model = tl.HookedTransformer.from_pretrained(model_name)
        else:
            model = tl.HookedTransformer.from_pretrained_no_processing(model_name)
        return model

    model = load_model(processing, model_name)

    assert model is not None, "The model has not been loaded successfully."

    # Initialize embed and unembed modules
    embed_module, unembed_module = initialize_embed_and_unembed(
        model=model,
        use_model_weights=True,
        apply_embed_ln=embed_apply_ln,
        apply_ln_final=unembed_apply_ln,
    )

    # print(embed_module.W_E)

    d_model = model.cfg.d_model
    n_toks = model.cfg.d_vocab_out

    # Dataset filtering
    dataset_name = dataset_config["name"]
    file_path = get_dataset_path(dataset_name)
    with open(file_path, "r", encoding="utf-8") as file:
        word_pairs = json.load(file)

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

    # Prepare datasets
    verify_learning = prepare_verify_analysis(
        model=model,
        embed_module=embed_module,
        all_word_pairs=all_word_pairs,
        seed=seed,
        keep_other_pair=True,
    )

    train_loader, test_loader = prepare_verify_datasets(
        verify_learning=verify_learning,
        batch_sizes=(train_batch_size, test_batch_size),
        top_k=top_k,
        top_k_selection_method=top_k_selection_method,
    )

    if "mark_accuracy_path" in dataset_config:
        azure_translations_path = get_dataset_path(dataset_config["mark_accuracy_path"])
    else:
        azure_translations_path = None

    # Initialize transformation and optimizer
    transform, optim = initialize_transform_and_optim(
        d_model,
        transformation=transformation,
        transform_kwargs={
            "apply_ln": transform_apply_ln,
        },
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
            unembed_module=unembed_module,
            loss_module=loss_module,
            n_epochs=n_epoch,
            plot_fig=False,
            wandb=wandb,
            azure_translations_path=azure_translations_path,
        )

    # Evaluate and log results
    test_accuracy = evaluate_accuracy(
        model,
        test_loader,
        transform,
        unembed_module=unembed_module,
        exact_match=False,
        print_results=False,
        print_top_preds=False,
    )

    if azure_translations_path is None:
        mark_translation_acc = None
    else:
        mark_translation_acc = mark_translation(
            model=model,
            transformation=transform,
            unembed_module=unembed_module,
            test_loader=test_loader,
            azure_translations_path=azure_translations_path,
            print_results=False,
        )

    verify_results_dict = verify_transform(
        model=model,
        transformation=transform,
        test_loader=test_loader,
        unembed_module=unembed_module,
    )

    cos_sims_trend_plot = plot_cosine_similarity_trend(verify_results_dict)

    # cos_sims_trend_plot.show(config={"responsive": True, "autosize": True})

    test_cos_sim_diff = test_cos_sim_difference(verify_results_dict)

    wandb.log(
        {
            "test_accuracy": test_accuracy,
            "mark_translation_acc": mark_translation_acc,
            "cos_sims_trend_plot": cos_sims_trend_plot,
            "test_cos_sim_diff": test_cos_sim_diff,
        }
    )

    wandb.finish()
