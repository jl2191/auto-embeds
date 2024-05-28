# %%
import json
import multiprocessing as mp
import os

os.environ["AUTOEMBEDS_CACHING"] = "TRUE"

import neptune
import numpy as np
import plotly.io as pio
import torch as t
from IPython.core.getipython import get_ipython
from neptune.types import File
from neptune.utils import stringify_unsupported
from transformers import AutoTokenizer

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
    calc_expected_metrics,
    calc_pred_same_as_input,
    evaluate_accuracy,
    mark_translation,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.logging import logger
from auto_embeds.utils.misc import get_experiment_worker_config
from auto_embeds.verify import (
    plot_cosine_similarity_trend,
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

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass


def run_experiment_parallel(config, num_workers):
    logger.info(f"total runs: {total_runs}")
    logger.info(f"running experiment with config: {config}")
    logger.info(f"using {num_workers} workers")
    if mp.get_start_method(allow_none=True) != "spawn":
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
    return run_experiment(config_to_use, use_neptune=True)


def run_experiment(config_dict, return_local_results=False, use_neptune=False):
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
        if use_neptune:
            run = neptune.init_run(
                project="mars/language-transformations",
                tags=neptune_config.get("tags", []),
            )
            run["config"] = stringify_unsupported(run_config)

        # Tokenizer setup
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Model weights setup
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

        # Dataset filtering
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
            return_type="dataloader",
        )

        if "mark_accuracy_path" in dataset_config:
            azure_translations_path = get_dataset_path(
                dataset_config["mark_accuracy_path"]
            )
        else:
            azure_translations_path = None

        # Initialize transformation and optimizer
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
                neptune_run=run if use_neptune else None,
                azure_translations_path=azure_translations_path,
            )

        # Evaluate and log results

        expected_metrics = calc_expected_metrics(
            transform_module=transform,
            data_loader=train_loader,
        )

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
            print_acc=False,
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
            logger.info(f"mark translation accuracy: {mark_translation_acc}")

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
        pred_same_as_input = calc_pred_same_as_input(
            tokenizer=tokenizer,
            test_loader=test_loader,
            transformation=transform,
            unembed_module=unembed_module,
        )

        results = {
            "expected_metrics": expected_metrics,
            "test_accuracy": test_accuracy,
            "mark_translation_acc": mark_translation_acc,
            "cosine_similarity_test_loss": cosine_similarity_test_loss,
            "mse_test_loss": mse_test_loss,
            "pred_same_as_input": pred_same_as_input,
        }

        if use_neptune:
            run["results"] = results
            run["results/cos_sims_trend_plot"].upload(cos_sims_trend_plot)
            run["results/json/verify_results"].upload(
                File.from_content(verify_results_json)
            )
            run["results/json/cos_sims_trend_plot"].upload(
                File.from_content(str(pio.to_json(cos_sims_trend_plot)))
            )
            run["results/json/test_cos_sim_diff"].upload(
                File.from_content(test_cos_sim_diff)
            )

        if return_local_results:
            pca = t.pca_lowrank(transform.transformations[0][1], q=2)
            principal_components = pca[0]
            results["principal_components"] = principal_components.tolist()

            svds, vector_norms = {"u": None, "s": None, "v": None}, []
            for operation, transform_tensor in transform.transformations:
                if operation == "multiply":
                    u, s, v = t.linalg.svd(transform_tensor)
                    svds = {"u": u, "s": s, "v": v}
                    vector_norms = [
                        t.linalg.norm(tensors, dim=1)
                        for batch in test_loader
                        for tensors in batch
                    ]

            # large_tensors = {"pca": pca, "verify_learning": verify_learning}

            train_dataset, test_dataset = prepare_verify_datasets(
                verify_learning=verify_learning,
                batch_sizes=(train_batch_size, test_batch_size),
                top_k=top_k,
                top_k_selection_method=top_k_selection_method,
                return_type="dataset",
            )

            big_tensors = {
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
            }

            results["svds"] = svds
            results["vector_norms"] = vector_norms
            local_results.append(run_config | results | big_tensors)

        if use_neptune:
            run.stop()

        # returning results that we are not uploading for local analysis
        # transform_weights = transform.state_dict()
        # results["transform_weights"] = transform_weights

    return local_results


# %%
if __name__ == "__main__":
    # run_experiment(experiment_config, use_neptune=True)
    run_experiment_parallel(experiment_config, num_workers=2)
