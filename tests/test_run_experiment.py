# %%
import pytest

from experiments.run_experiment import run_experiment


@pytest.fixture
def config_dict():
    config_dict = {
        "neptune": {
            "tags": [
                # f"{datetime.now():%Y-%m-%d}",
                # f"{datetime.now():%Y-%m-%d} big run",
                "experiment 94",
                "run group 5",
            ],
        },
        "description": ["printing to check ln is gucci"],
        "models": [
            "bigscience/bloom-560m",
            # "bigscience/bloom-1b1",
            # "bigscience/bloom-3b",
            # "gpt2",
            # "gpt2-medium",
            # "gpt2-large",
        ],
        "processings": [
            False,
        ],
        "datasets": [
            # {
            #     "name": "wikdict_en_fr_extracted",
            #     "min_length": 5,
            #     "space_configurations": [{"en": "space", "fr": "space"}],
            #     "mark_accuracy_path": "wikdict_en_fr_azure_validation",
            # },
            # {
            #     "name": "random_word_pairs",
            #     "min_length": 2,
            #     "space_configurations": [{"en": "space", "fr": "space"}],
            # },
            # {
            #     "name": "singular_plural_pairs",
            #     "min_length": 2,
            #     "space_configurations": [
            #         {"en": "space", "fr": "space"},
            #         # {"en": "no_space", "fr": "no_space"},
            #     ],
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
            # "analytical_rotation",
            # "analytical_rotation_and_reflection",
            # "roma_analytical",
            # "roma_scale_analytical",
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
        # "seeds": [1, 2, 3, 4, 5],
        "seeds": [1],
        # "loss_functions": ["cosine_similarity", "mse_loss"],
        "loss_functions": ["cosine_similarity"],
        "embed_weight": ["model_weights"],
        # "embed_ln_weights": ["no_ln", "default_weights", "model_weights"],
        "embed_ln_weights": ["no_ln", "model_weights"],
        # "embed_ln_weights": ["default_weights"],
        # "embed_ln_weights": ["default_weights"],
        # "unembed_weight": ["model_weights"],
        "unembed_weight": ["model_weights"],
        # "unembed_ln_weights": ["no_ln", "default_weights", "model_weights"],
        "unembed_ln_weights": ["no_ln", "model_weights"],
        # "unembed_ln_weights": ["default_weights"],
        # "unembed_ln_weights": ["default_weights"],
        "n_epochs": [100],
        "weight_decay": [
            0,
            # 2e-5,
        ],
        "lr": [8e-5],
    }
    return config_dict


@pytest.mark.slow
def test_run_experiment(benchmark, config_dict):
    benchmark.extra_info["min_rounds"] = 1
    benchmark.extra_info["max_time"] = 60
    benchmark.extra_info["warmup_iterations"] = 10
    local_results = benchmark(
        run_experiment, config_dict, return_local_results=True, use_neptune=False
    )
    assert isinstance(local_results, list)
