# %%
import datetime
import os

from experiments.run_experiment import run_experiment

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["AUTOEMBEDS_CACHING"] = "true"

import multiprocessing as mp

import numpy as np
import torch as t

from auto_embeds.utils.misc import (
    get_experiment_worker_config,
    is_notebook,
    setup_arg_parser,
)

# Seed for reproducibility
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

NUM_WORKERS = 2

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
        # "linear_map",
        # "biased_linear_map",
        # "uncentered_linear_map",
        # "biased_uncentered_linear_map",
        "analytical_linear_map",
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
    # "embed_ln_weights": ["no_ln", "default_weights", "model_weights"],
    "embed_ln_weights": ["no_ln", "model_weights"],
    # "embed_ln_weights": ["default_weights"],
    "unembed_weight": ["model_weights"],
    # "unembed_ln_weights": ["no_ln", "default_weights", "model_weights"],
    "unembed_ln_weights": ["no_ln", "model_weights"],
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

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    if is_notebook():
        with mp.Pool(NUM_WORKERS) as pool:
            results = pool.starmap(
                run_worker, [(i, experiment_config) for i in range(NUM_WORKERS)]
            )
        print(
            "Detected Jupyter notebook or IPython session. Running all experiments "
            "and adding 'test' neptune tag."
        )
        results = run_experiment(
            get_experiment_worker_config(
                experiment_config=experiment_config,
                split_parameter="datasets",
                n_splits=1,
                worker_id=0,
            )
        )
    else:
        # command-line execution
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
            results = run_experiment(config_to_use)
        else:
            print("No worker ID provided. Running all experiments.")
            run_experiment(
                get_experiment_worker_config(
                    experiment_config=experiment_config,
                    split_parameter="datasets",
                    n_splits=1,
                    worker_id=0,
                )
            )

# %%
ic(results.keys())
