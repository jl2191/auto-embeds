# %%
import json
import multiprocessing as mp
import os

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["AUTOEMBEDS_CACHING"] = "TRUE"

import neptune
import numpy as np
import plotly.io as pio
import torch as t
from neptune.types import File
from neptune.utils import stringify_unsupported
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer

from auto_embeds.analytical import initialize_manual_transform
from auto_embeds.data import filter_word_pairs, get_cached_weights, get_dataset_path
from auto_embeds.embed_utils import (
    initialize_embed_and_unembed,
    initialize_transform_and_optim,
    train_transform,
)
from auto_embeds.metrics import (
    calc_metrics,
    initialize_loss,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.logging import logger
from auto_embeds.utils.misc import get_experiment_worker_config
from auto_embeds.verify import (
    plot_cos_sim_trend,
    prepare_verify_analysis,
    prepare_verify_datasets,
    test_cos_sim_difference,
    verify_transform,
)
from experiments.configure_experiment import (
    experiment_config,
    get_config_list,
    total_runs,
)


def run_experiment_parallel(config, num_workers):
    logger.info(f"total runs: {total_runs}")
    logger.info(f"running experiment with config: {config}")
    logger.info(f"using {num_workers} workers")
    mp.set_start_method("spawn", force=True)
    tasks = [(i, config, "datasets", num_workers) for i in range(1, num_workers + 1)]
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(run_worker, tasks)
    return [item for sublist in results for item in sublist]


def run_worker(worker_id, experiment_config, split_parameter="datasets", n_splits=2):
    config_to_use = get_experiment_worker_config(
        experiment_config=experiment_config,
        split_parameter=split_parameter,
        n_splits=n_splits,
        worker_id=worker_id,
    )
    print(f"Running experiment for worker ID = {worker_id}")
    return run_experiment(config_to_use)


def run_experiment(config_dict, return_local_results=False):
    local_results = []
    # extracting 'neptune' configuration and generating all combinations of configs
    # as a list of lists
    run = None
    neptune_config = config_dict.get("neptune", {})
    config_dict = {k: v for k, v in config_dict.items() if k != "neptune"}
    config_list = get_config_list(config_dict)
    last_model_config = None
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
            tags=neptune_config.get("tags", []),
            mode=neptune_config.get("mode", "async"),
        )
        run["config"] = stringify_unsupported(run_config)

        # tokenizer setup
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # model weights setup
        current_model_config = (model_name, processing)
        if model_weights is None or current_model_config != last_model_config:
            model_weights = get_cached_weights(model_name, processing)
            last_model_config = current_model_config
        d_model = model_weights["W_E"].shape[1]

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

        # dataset filtering
        dataset_name = dataset_config["name"]
        file_path = get_dataset_path(dataset_name)
        with open(file_path, "r", encoding="utf-8") as file:
            word_pairs = json.load(file)

        all_word_pairs = filter_word_pairs(
            tokenizer=tokenizer,
            word_pairs=word_pairs,
            discard_if_same=True,
            min_length=dataset_config["min_length"],
            space_configurations=dataset_config["space_configurations"],
            print_number=True,
            verbose_count=True,
        )

        # prepare datasets
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
            return_type="dataloader",
        )

        if "mark_accuracy_path" in dataset_config:
            azure_translations_path = get_dataset_path(
                dataset_config["mark_accuracy_path"]
            )
        else:
            azure_translations_path = None

        # initialize transformation and optimizer
        if "analytical" in transformation:
            transform = initialize_manual_transform(
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

        subset_indices = t.randint(len(train_loader.dataset), (512,))  # type: ignore
        train_loader_sample = DataLoader(
            train_loader.dataset,
            batch_size=512,
            sampler=SubsetRandomSampler(subset_indices),
        )

        train_metrics = calc_metrics(
            train_loader_sample,
            transform,
            tokenizer,
            unembed_module,
            azure_translations_path,
        )
        test_metrics = calc_metrics(
            test_loader, transform, tokenizer, unembed_module, azure_translations_path
        )

        verify_results_dict = verify_transform(
            tokenizer=tokenizer,
            transformation=transform,
            test_loader=test_loader,
            unembed_module=unembed_module,
        )

        cos_sims_trend_plot = plot_cos_sim_trend(verify_results_dict)
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

        # logging results

        run["results"] = {
            "train": train_metrics,
            "test": test_metrics,
        }

        run["results/test/cos_sims_trend_plot"].upload(cos_sims_trend_plot)
        run["results/test/json/verify_results"].upload(
            File.from_content(verify_results_json)
        )
        run["results/test/json/cos_sims_trend_plot"].upload(
            File.from_content(str(pio.to_json(cos_sims_trend_plot)))
        )
        run["results/test/json/test_cos_sim_diff"].upload(
            File.from_content(test_cos_sim_diff)
        )

        if return_local_results:
            local_results = {
                "train": train_metrics,
                "test": test_metrics,
            }

        # returning results that we are not uploading for local analysis
        # transform_weights = transform.state_dict()
        # results["transform_weights"] = transform_weights

    return local_results


if __name__ == "__main__":
    # run_experiment(experiment_config)
    run_experiment(experiment_config)
    # run_experiment_parallel(experiment_config, num_workers=2)
