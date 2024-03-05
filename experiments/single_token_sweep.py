# %%
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time

import numpy as np
import torch as t
import transformer_lens as tl

import wandb
from auto_steer.data import create_data_loaders
from auto_steer.steering_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    initialize_transform_and_optim,
    tokenize_texts,
    train_transform,
)
from auto_steer.utils.misc import (
    repo_path_to_abs_path,
)

np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)


def run_sweep_for_model(model_name: str):
    model = tl.HookedTransformer.from_pretrained_no_processing(model_name)
    d_model = model.cfg.d_model
    datasets_folder = repo_path_to_abs_path("datasets")
    repo_path_to_abs_path("datasets/activation_cache")

    def train():
        wandb.init(project=f"single_token_experiments_{model_name}")
        config = wandb.config

        with open(
            f"{datasets_folder}/kaikki-french-dictionary-single-word-pairs-no-hyphen.json",
            "r",
        ) as file:
            fr_en_pairs_file = json.load(file)

        en_fr_pairs = [[pair["English"], pair["French"]] for pair in fr_en_pairs_file]

        # Tokenize and generate embeddings for the given pairs
        en_toks, en_attn_mask, fr_toks, fr_attn_mask = tokenize_texts(
            model,
            en_fr_pairs,
            padding_side="left",
            single_tokens_only=True,
            discard_if_same=True,
            min_length=config.min_length,
            capture_diff_case=True,
            capture_space=True,
            capture_no_space=True,
        )
        en_embeds = (
            model.embed.W_E[en_toks].detach().clone()
        )  # shape[batch, pos, d_model]
        fr_embeds = (
            model.embed.W_E[fr_toks].detach().clone()
        )  # shape[batch, pos, d_model]

        start_time = time.time()
        print(
            f"Running experiment with min_length={config.min_length}, "
            f"batch_size={config.batch_size}, "
            f"epochs={config.epochs}, "
            f"transformation={config.transformation}, "
            f"lr={config.lr}, "
        )

        # Creating data loaders with the current batch size
        train_loader, test_loader = create_data_loaders(
            en_embeds,
            fr_embeds,
            batch_size=config.batch_size,
            train_ratio=0.97,
        )

        # Log the size of the train and test datasets
        total_tokens = len(train_loader) + len(test_loader)
        print(
            f"Total tokens for min_length={config.min_length}, "
            f"batch_size={config.batch_size}, "
            f"total_tokens={total_tokens}"
        )

        # Initializing transformation and optimizer with the current settings
        transform, optim = initialize_transform_and_optim(
            d_model,
            transformation=config.transformation,
            optim_kwargs={"lr": config.lr},
        )

        if optim is not None:
            transform, loss_history = train_transform(
                model,
                train_loader,
                transform,
                optim,
                config.epochs,
                wandb=wandb,
            )

        # Evaluating the model's accuracy and cosine similarity
        accuracy = evaluate_accuracy(model, test_loader, transform, exact_match=False)

        cosine_similarity = calc_cos_sim_acc(test_loader, transform)
        print(
            f"Accuracy for min_length={config.min_length}, "
            f"batch_size={config.batch_size}, "
            f"epochs={config.epochs}, "
            f"transformation={config.transformation}, "
            f"lr={config.lr}, "
            f"accuracy={accuracy}, "
            f"Cosine Similarity: {cosine_similarity}"
        )

        # Calculating experiment duration
        end_time = time.time()
        experiment_duration = end_time - start_time

        # Logging summary (per run) statistics to wandb
        wand_summary = {
            "test_accuracy": accuracy,
            "test_cos_sim": cosine_similarity,
            "experiment_duration": experiment_duration,
            "total_tokens": total_tokens,
        }
        if wandb.run is not None:
            for metric, value in wand_summary.items():
                wandb.run.summary[metric] = value
        else:
            print("Warning: wandb.run is None. Ensure wandb.init() was called.")

        wandb.finish()

    # Define the sweep configuration
    sweep_config = {
        "method": "grid",
        "parameters": {
            "min_length": {"values": [3, 4, 5]},
            "batch_size": {"values": [256, 512]},
            "epochs": {"values": [100, 200, 300]},
            "transformation": {
                "values": [
                    "identity",
                    "linear_map",
                    "rotation",
                    "translation",
                    "offset_linear_map",
                    "offset_rotation",
                    "uncentered_linear_map",
                    "uncentered_rotation",
                ]
            },
            "lr": {"values": [0.0001, 0.0002, 0.0005]},
        },
    }

    # Create the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=f"single_token_experiments_{model_name}"
    )

    # Run the sweep agent
    wandb.agent(sweep_id, train)


# Run sweeps for both models
run_sweep_for_model("bloom-560m")
run_sweep_for_model("bloom-3b")
# %%
