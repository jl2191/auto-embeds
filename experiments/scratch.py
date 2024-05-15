# %%
import argparse
import datetime
import itertools
import json
import os

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["AUTOEMBEDS_CACHING"] = "true"

import neptune
import numpy as np
import plotly.io as pio
import torch as t
from neptune.types import File
from neptune.utils import stringify_unsupported
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from auto_embeds.analytical import initialize_manual_transform
from auto_embeds.data import filter_word_pairs, get_cached_weights, get_dataset_path
from auto_embeds.embed_utils import (
    calculate_test_loss,
    initialize_embed_and_unembed,
    initialize_loss,
    initialize_transform_and_optim,
    train_transform,
)
from auto_embeds.metrics import (
    evaluate_accuracy,
    mark_translation,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.logging import logger
from auto_embeds.utils.misc import get_experiment_worker_config, is_notebook
from auto_embeds.verify import (
    plot_cosine_similarity_trend,
    prepare_verify_analysis,
    prepare_verify_datasets,
    test_cos_sim_difference,
    verify_transform,
)

# Seed for reproducibility
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

# Configuration for overall experiments
experiment_config = {
    "neptune": {
        "tags": [
            f"{datetime.datetime.now():%Y-%m-%d}",
            f"{datetime.datetime.now():%Y-%m-%d} rotation trials",
            "experiment 2",
            "run group 2",
        ],
    },
    "description": ["none"],
    "models": [
        "bigscience/bloom-560m",
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
            "space_configurations": [{"en": "space", "fr": "space"}],
            "mark_accuracy_path": "wikdict_en_fr_azure_validation",
        },
        # {
        #     "name": "random_word_pairs",
        #     "min_length": 2,
        #     "space_configurations": [{"en": "space", "fr": "space"}],
        # },
        # {
        #     "name": "singular_plural_pairs",
        #     "min_length": 2,
        #     "space_configurations": [{"en": "space", "fr": "space"}],
        # },
        # {
        #     "name": "muse_en_fr_extracted",
        #     "min_length": 5,
        #     "space_configurations": [{"en": "space", "fr": "space"}],
        #     "mark_accuracy_path": "muse_en_fr_azure_validation",
        # },
        {
            "name": "cc_cedict_zh_en_extracted",
            "min_length": 2,
            "space_configurations": [{"en": "no_space", "fr": "space"}],
            "mark_accuracy_path": "cc_cedict_zh_en_azure_validation",
        },
        # {
        #     "name": "muse_zh_en_extracted_train",
        #     "min_length": 2,
        #     "space_configurations": [{"en": "no_space", "fr": "space"}],
        #     "mark_accuracy_path": "muse_zh_en_azure_validation",
        # },
    ],
    "transformations": [
        # "identity",
        # "translation",
        # "rotation",
        # "biased_rotation",
        # "uncentered_rotation",
        "linear_map",
        # "biased_linear_map",
        # "uncentered_linear_map",
        # "biased_uncentered_linear_map",
        # "analytical_linear_map",
        # "analytical_translation",
        # "analytical_rotation_scipy",
        # "analytical_rotation_roma",
        # "analytical_rotation_torch",
        # "analytical_rotation_scipy_scale",
        # "analytical_rotation_roma_scale",
        # "analytical_rotation_torch_scale",
        # "roma_analytical",
        # "roma_scale_analytical",
        # "torch_analytical",
        # "torch_scale_analytical",
        # "kabsch_analytical",
        # "kabsch_analytical_new",
        # "kabsch_analytical_no_scale",
    ],
    "train_batch_sizes": [128],
    "test_batch_sizes": [256],
    "top_k": [200],
    "top_k_selection_methods": [
        # "src_and_src",
        # "tgt_and_tgt",
        # "top_src",
        "top_tgt",
    ],
    "seeds": [1],
    # "loss_functions": ["cosine_similarity", "mse_loss"],
    "loss_functions": ["cosine_similarity"],
    "embed_weight": ["model_weights"],
    "embed_ln_weights": ["no_ln", "default_weights", "model_weights"],
    # "embed_ln_weights": ["no_ln", "model_weights"],
    # "embed_ln_weights": ["default_weights"],
    "unembed_weight": ["model_weights"],
    "unembed_ln_weights": ["no_ln", "default_weights", "model_weights"],
    # "unembed_ln_weights": ["no_ln", "model_weights"],
    # "unembed_ln_weights": ["default_weights"],
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


def run_experiment(config_dict):
    # Extracting 'neptune' configuration and generating all combinations of configurations
    # as a list of lists
    neptune_config = config_dict.pop("neptune")
    config_values = [
        config_dict[entry] if entry != "datasets" else config_dict[entry]
        for entry in config_dict
    ]
    config_list = list(itertools.product(*config_values))

    # To prevent unnecessary reloading
    last_model_config = None
    last_dataset_config = None
    model = None
    model_weights = None

    for (
        description,
        model_name,
        processing,
        dataset_config,
        transformation,
        train_batch_size,
        test_batch_size,
        top_k,
        top_k_selection_method,
        seed,
        loss_function,
        embed_weight,
        embed_ln_weights,
        unembed_weight,
        unembed_ln_weights,
        n_epoch,
        weight_decay,
        lr,
    ) in tqdm(config_list, total=len(config_list)):

        embed_ln = True if embed_ln_weights != "no_ln" else False
        unembed_ln = True if unembed_ln_weights != "no_ln" else False

        run_config = {
            "description": description,
            "model_name": model_name,
            "processing": processing,
            "dataset": dataset_config,
            "transformation": transformation,
            "train_batch_size": train_batch_size,
            "test_batch_size": test_batch_size,
            "top_k": top_k,
            "top_k_selection_method": top_k_selection_method,
            "seed": seed,
            "loss_function": loss_function,
            "embed_weight": embed_weight,
            "embed_ln": embed_ln,
            "embed_ln_weights": embed_ln_weights,
            "unembed_weight": unembed_weight,
            "unembed_ln": unembed_ln,
            "unembed_ln_weights": unembed_ln_weights,
            "n_epoch": n_epoch,
            "weight_decay": weight_decay,
            "lr": lr,
        }

        logger.info(f"Running experiment with config: {run_config}")

        # neptune run init
        run = neptune.init_run(
            project="mars/language-transformations",
            tags=neptune_config["tags"],
        )
        run["config"] = stringify_unsupported(run_config)

        # Tokenizer setup
        tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            model_name
        )  # type: ignore

        # Model weights setup
        current_model_config = (model_name, processing)
        if current_model_config != last_model_config:
            model_weights = get_cached_weights(model_name, processing)
            last_model_config = current_model_config

        # Initialize embed and unembed modules
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

        d_model = model_weights["W_E"].shape[1]
        n_toks = model_weights["W_E"].shape[0]

        # Dataset filtering
        dataset_name = dataset_config["name"]
        file_path = get_dataset_path(dataset_name)
        with open(file_path, "r", encoding="utf-8") as file:
            word_pairs = json.load(file)

        current_dataset_config = dataset_config
        if current_dataset_config != last_dataset_config:
            all_word_pairs = filter_word_pairs(
                tokenizer=tokenizer,
                word_pairs=word_pairs,
                discard_if_same=True,
                min_length=dataset_config["min_length"],
                space_configurations=dataset_config["space_configurations"],
                print_number=True,
                verbose_count=True,
            )
            last_dataset_config = current_dataset_config

        # Prepare datasets
        verify_learning = prepare_verify_analysis(
            tokenizer=tokenizer,
            embed_module=embed_module,
            all_word_pairs=all_word_pairs,
            seed=seed,
        )

        train_loader, test_loader = prepare_verify_datasets(
            verify_learning=verify_learning,
            batch_sizes=(train_batch_size, test_batch_size),
            top_k=top_k,
            top_k_selection_method=top_k_selection_method,
        )

        if "mark_accuracy_path" in dataset_config:
            azure_translations_path = get_dataset_path(
                dataset_config["mark_accuracy_path"]
            )
        else:
            azure_translations_path = None

        # Initialize transformation and optimizer
        if "analytical" in transformation:
            transform, expected_metrics = initialize_manual_transform(
                transform_name=transformation,
                train_loader=train_loader,
            )
            optim = None
        else:
            transform, optim = initialize_transform_and_optim(
                d_model,
                transformation=transformation,
                optim_kwargs={"lr": lr, "weight_decay": weight_decay},
            )
            expected_metrics = None

        loss_module = initialize_loss(loss_function)

        if optim is not None:
            transform, loss_history = train_transform(
                tokenizer=tokenizer,
                train_loader=train_loader,
                test_loader=test_loader,
                transform=transform,
                optim=optim,
                unembed_module=unembed_module,
                loss_module=loss_module,
                n_epochs=n_epoch,
                plot_fig=False,
                neptune_run=run,
                azure_translations_path=azure_translations_path,
            )

        # Evaluate and log results
        cosine_similarity_test_loss = calculate_test_loss(
            test_loader=test_loader,
            transform=transform,
            loss_module=initialize_loss("cosine_similarity"),
        )

        mse_test_loss = calculate_test_loss(
            test_loader=test_loader,
            transform=transform,
            loss_module=initialize_loss("mse_loss"),
        )

        test_accuracy = evaluate_accuracy(
            tokenizer=tokenizer,
            test_loader=test_loader,
            transformation=transform,
            unembed_module=unembed_module,
            exact_match=False,
            print_results=False,
            print_top_preds=False,
        )

        if azure_translations_path is None:
            mark_translation_acc = None
        else:
            mark_translation_acc = mark_translation(
                tokenizer=tokenizer,
                transformation=transform,
                unembed_module=unembed_module,
                test_loader=test_loader,
                azure_translations_path=azure_translations_path,
                print_results=False,
            )

        verify_results_dict = verify_transform(
            tokenizer=tokenizer,
            transformation=transform,
            test_loader=test_loader,
            unembed_module=unembed_module,
        )

        # calculating and logging metrics
        cos_sims_trend_plot = plot_cosine_similarity_trend(verify_results_dict)
        verify_results_json = json.dumps(
            {
                key: value.tolist() if isinstance(value, t.Tensor) else value
                for key, value in verify_results_dict.items()
            }
        )
        test_cos_sim_diff = json.dumps(
            {
                k: bool(v) if isinstance(v, np.bool_) else v
                for k, v in test_cos_sim_difference(verify_results_dict).items()
            }
        )

        run["results"] = {
            "expected_metrics": expected_metrics,
            "test_accuracy": test_accuracy,
            "mark_translation_acc": mark_translation_acc,
            "cos_sims_trend_plot": cos_sims_trend_plot,
            "cosine_similarity_test_loss": cosine_similarity_test_loss,
            "mse_test_loss": mse_test_loss,
        }
        run["results/json/verify_results"].upload(
            File.from_content(verify_results_json)
        )
        run["results/json/cos_sims_trend_plot"].upload(
            File.from_content(str(pio.to_json(cos_sims_trend_plot)))
        )
        run["results/json/test_cos_sim_diff"].upload(
            File.from_content(test_cos_sim_diff)
        )

        run.stop()


def setup_arg_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Run experiments with specified worker configuration."
    )
    parser.add_argument(
        "--worker_id",
        type=int,
        choices=[1, 2],
        help="Optional: Worker ID to use for running the experiment.",
    )
    return parser


if __name__ == "__main__":
    if is_notebook():
        # If running in a Jupyter notebook, run all experiments
        print(
            "Detected Jupyter notebook or IPython session. Running all experiments "
            "and adding 'test' neptune tag."
        )
        run_experiment(
            get_experiment_worker_config(
                experiment_config=experiment_config,
                split_parameter="datasets",
                n_splits=1,
                worker_id=0,
            )
        )
    else:
        # Command-line execution
        parser = setup_arg_parser()
        args = parser.parse_args()
        if args.worker_id:
            config_to_use = get_experiment_worker_config(
                experiment_config=experiment_config,
                split_parameter="datasets",
                n_splits=2,
                worker_id=args.worker_id,
            )
            print(f"Running experiment for worker ID = {args.worker_id}")
            run_experiment(config_to_use)
