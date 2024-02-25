# %%
import json
import itertools
import time
import wandb
import torch as t
import transformer_lens as tl

from typing import Optional, Dict, Any
from auto_steer.steering_utils import *
from auto_steer.utils.misc import (
    repo_path_to_abs_path,
)

model = tl.HookedTransformer.from_pretrained_no_processing("bloom-560m")
device = model.cfg.device
d_model = model.cfg.d_model
n_toks = model.cfg.d_vocab_out
datasets_folder = repo_path_to_abs_path("datasets")
cache_folder = repo_path_to_abs_path("datasets/activation_cache")


def train(config: Optional[Dict[str, Any]] = None):

    wandb.init(config=config)
    config = wandb.config

    with open(f"{datasets_folder}/kaikki-french-dictionary-single-word-pairs-no-hyphen.json", "r") as file:
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
        capture_no_space=True
    )
    en_embeds = model.embed.W_E[en_toks].detach().clone()  # shape[batch, pos, d_model]
    fr_embeds = model.embed.W_E[fr_toks].detach().clone()  # shape[batch, pos, d_model]

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
    total_tokens = len(train_loader.dataset) + len(test_loader.dataset)
    print(
        f"Total tokens for min_length={config.min_length}, "
        f"batch_size={config.batch_size}, "
        f"total_tokens={total_tokens}"
    )

    # Initializing transformation and optimizer with the current settings
    initial_transformation, optim = initialize_transform_and_optim(
        d_model, transformation=config.transformation, lr=config.lr, device=device
    )

    # Training the model with the current number of epochs
    learned_transformation = train_transform(
        model, train_loader, initial_transformation, optim, config.epochs, device
    )

    # Evaluating the model's accuracy and cosine similarity
    accuracy = evaluate_accuracy(
        model, test_loader, learned_transformation, exact_match=False
    )
    cosine_similarity = calc_cos_sim_acc(test_loader, learned_transformation)
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

    # Logging the results to wandb
    wandb.log(
        {
            "min_length": config.min_length,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "transformation": config.transformation,
            "learning_rate": config.lr,
            "accuracy": accuracy,
            "cosine_similarity": cosine_similarity,
            "experiment_duration": experiment_duration,
            "total_tokens": total_tokens,
        }
    )
    wandb.finish()

# Define the sweep configuration
sweep_config = {
    'method': 'grid',
    'parameters': {
        'min_length': {
            'values': [3, 4, 5]
        },
        'batch_size': {
            'values': [256, 512]
        },
        'epochs': {
            'values': [50, 100, 200, 300]
        },
        'transformation': {
            'values': ["rotation", "linear_map"]
        },
        'lr': {
            'values': [0.0001, 0.0002, 0.0005]
        }
    }
}
# Create the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="single_token_experiments_bloom-560m")

# Run the sweep agent
wandb.agent(sweep_id, train)

# %%