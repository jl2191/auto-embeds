# %%
import json
import os
import time

import numpy as np
import torch as t
import transformer_lens as tl
import wandb

from auto_steer.data import create_data_loaders
from auto_steer.steering_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    initialize_loss,
    initialize_transform_and_optim,
    tokenize_texts,
    train_transform,
)
from auto_steer.utils.misc import repo_path_to_abs_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_sweep_for_model(model_name: str):
    sweep_config = {
        "method": "grid",
        "parameters": {
            "batch_size": {"values": [512]},
            "epochs": {"values": [100]},
            "transformation": {
                "values": [
                    "identity",
                    "translation",
                    "linear_map",
                    "biased_linear_map",
                    "uncentered_linear_map",
                    "biased_uncentered_linear_map",
                    "rotation",
                    "biased_rotation",
                    "uncentered_rotation",
                ]
            },
            "lr": {"values": [0.0001, 0.001, 0.005, 0.01]},
        },
    }

    model = tl.HookedTransformer.from_pretrained_no_processing(model_name)
    datasets_folder = repo_path_to_abs_path("datasets")
    repo_path_to_abs_path("datasets/activation_cache")

    with open(
        f"{datasets_folder}/kaikki-french-dictionary-single-word-pairs-no-hyphen.json",
        "r",
    ) as file:
        fr_en_pairs_file = json.load(file)

    en_fr_pairs = [[pair["English"], pair["French"]] for pair in fr_en_pairs_file]

    en_toks, en_attn_mask, fr_toks, fr_attn_mask = tokenize_texts(
        model,
        en_fr_pairs,
        padding_side="left",
        single_tokens_only=True,
        discard_if_same=True,
        min_length=3,
        capture_diff_case=True,
        capture_space=True,
        capture_no_space=True,
    )
    en_embeds = model.embed.W_E[en_toks].detach().clone()  # shape[batch, pos, d_model]
    fr_embeds = model.embed.W_E[fr_toks].detach().clone()  # shape[batch, pos, d_model]

    # # Creating data loaders with the current batch size
    train_loader, test_loader = create_data_loaders(
        en_embeds,
        fr_embeds,
        batch_size=512,
        train_ratio=0.97,
    )

    def train():
        d_model = model.cfg.d_model
        start_time = time.time()
        wandb.init()
        config = wandb.config

        np.random.seed(1)
        t.manual_seed(1)
        t.cuda.manual_seed(1)

        transform, optim = initialize_transform_and_optim(
            d_model,
            transformation=config.transformation,
            optim_kwargs={"lr": config.lr},
        )

        loss_module = initialize_loss("cosine_similarity")

        if optim is not None:
            transform, _ = train_transform(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                transform=transform,
                optim=optim,
                loss_module=loss_module,
                n_epochs=config.epochs,
                wandb=wandb,
            )

        accuracy = evaluate_accuracy(model, test_loader, transform, exact_match=False)

        cosine_similarity = calc_cos_sim_acc(test_loader, transform)
        print(
            f"running experiment with:"
            f"batch_size={config.batch_size}, "
            f"epochs={config.epochs}, "
            f"transformation={config.transformation}, "
            f"lr={config.lr}, "
            f"cosine similarity: {cosine_similarity}"
        )

        end_time = time.time()
        experiment_duration = end_time - start_time

        total_tokens = len(train_loader) + len(test_loader)

        wand_summary = {
            "test_accuracy": accuracy,
            "test_cos_sim": cosine_similarity,
            "experiment_duration": experiment_duration,
            "total_tokens": total_tokens,
            "model_name": model_name,
        }
        for metric, value in wand_summary.items():
            wandb.run.summary[metric] = value  # type: ignore

    wandb.finish()
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=f"single_token_experiments_6_test_{model_name}"
    )
    wandb.agent(sweep_id, train)


# %%
run_sweep_for_model("bloom-560m")
run_sweep_for_model("bloom-1b1")
run_sweep_for_model("bloom-1b7")
run_sweep_for_model("bloom-3b")
# %%

run_sweep_for_model("gpt2-small")
run_sweep_for_model("gpt2-large")
run_sweep_for_model("gpt2-xl")

# %%
run_sweep_for_model("opt-125m")
run_sweep_for_model("opt-1.3b")
run_sweep_for_model("opt-2.7b")
# %%
