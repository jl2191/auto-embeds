# %%
import json
import random
import time

import numpy as np
import torch as t
import transformer_lens as tl
import wandb
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.data import get_dataset_path
from auto_embeds.embed_utils import (
    calc_cos_sim_acc,
    evaluate_accuracy,
    filter_word_pairs,
    initialize_loss,
    initialize_transform_and_optim,
    tokenize_word_pairs,
    train_transform,
)


def run_sweep_for_model(
    model_name: str, layernorm: bool = False, no_processing: bool = False
):
    sweep_config = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "test_loss"},
        "parameters": {
            "batch_size": {"values": [64, 128, 256]},
            "epochs": {"values": [50, 100, 150]},
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
            "lr": {"distribution": "uniform", "min": 0, "max": 0.1},
            "weight_decay": {"distribution": "uniform", "min": 0, "max": 0.01},
        },
    }

    if no_processing:
        model = tl.HookedTransformer.from_pretrained_no_processing(model_name)
    else:
        model = tl.HookedTransformer.from_pretrained(model_name)

    file_path = get_dataset_path("cc_cedict_zh_en_parsed")
    with open(file_path, "r") as file:
        word_pairs = json.load(file)
    random.seed(1)
    random.shuffle(word_pairs)
    split_index = int(len(word_pairs) * 0.97)
    train_en_fr_pairs = word_pairs[:split_index]
    test_en_fr_pairs = word_pairs[split_index:]

    train_word_pairs = filter_word_pairs(
        model,
        train_en_fr_pairs,
        discard_if_same=True,
        min_length=4,
        # capture_diff_case=True,
        capture_space=True,
        capture_no_space=True,
        print_pairs=True,
        print_number=True,
        # max_token_id=100_000,
        # most_common_english=True,
        # most_common_french=True,
    )

    test_word_pairs = filter_word_pairs(
        model,
        test_en_fr_pairs,
        discard_if_same=True,
        min_length=4,
        # capture_diff_case=True,
        capture_space=True,
        capture_no_space=True,
        # print_pairs=True,
        print_number=True,
        # max_token_id=100_000,
        # most_common_english=True,
        # most_common_french=True,
    )

    train_en_toks, train_fr_toks, _, _ = tokenize_word_pairs(model, train_word_pairs)
    test_en_toks, test_fr_toks, _, _ = tokenize_word_pairs(model, test_word_pairs)

    if layernorm:
        train_en_embeds = t.nn.functional.layer_norm(
            model.embed.W_E[train_en_toks].detach().clone(), [model.cfg.d_model]
        )
        train_fr_embeds = t.nn.functional.layer_norm(
            model.embed.W_E[train_fr_toks].detach().clone(), [model.cfg.d_model]
        )
        test_en_embeds = t.nn.functional.layer_norm(
            model.embed.W_E[test_en_toks].detach().clone(), [model.cfg.d_model]
        )
        test_fr_embeds = t.nn.functional.layer_norm(
            model.embed.W_E[test_fr_toks].detach().clone(), [model.cfg.d_model]
        )
    else:
        train_en_embeds = model.embed.W_E[train_en_toks].detach().clone()
        train_fr_embeds = model.embed.W_E[train_fr_toks].detach().clone()
        test_en_embeds = model.embed.W_E[test_en_toks].detach().clone()
        test_fr_embeds = model.embed.W_E[test_fr_toks].detach().clone()
        # all of these are shape [batch, pos, d_model]

    def train():

        d_model = model.cfg.d_model
        start_time = time.time()
        wandb.init(tags=["test_run"])
        config = wandb.config

        np.random.seed(1)
        t.manual_seed(1)
        t.cuda.manual_seed(1)

        train_dataset = TensorDataset(train_en_embeds, train_fr_embeds)
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )

        test_dataset = TensorDataset(test_en_embeds, test_fr_embeds)
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=True
        )

        print("running experiment with:")
        print(f"batch_size={config.batch_size}")
        print(f"epochs={config.epochs}")
        print(f"lr={config.lr}")
        print(f"transformation={config.transformation}")
        print(f"layernorm: {layernorm}")
        print(f"no_processing: {no_processing}")
        print(f"model_name: {model_name}")

        transform = None
        optim = None

        transform, optim = initialize_transform_and_optim(
            d_model,
            transformation=config.transformation,
            optim_kwargs={"lr": config.lr, "weight_decay": config.weight_decay},
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
                plot_fig=False,
                wandb=wandb,
            )

        accuracy = evaluate_accuracy(
            model, test_loader, transform, exact_match=False, print_results=True
        )
        # print(f"Correct Percentage: {accuracy * 100:.2f}%")

        cosine_similarity = calc_cos_sim_acc(test_loader, transform)
        # print("Test Accuracy:", cosine_similarity)

        end_time = time.time()
        experiment_duration = end_time - start_time

        wand_summary = {
            "test_accuracy": accuracy,
            "test_cos_sim": cosine_similarity,
            "experiment_duration": experiment_duration,
            "train_tokens": len(train_dataset),
            "test_tokens": len(test_dataset),
            "total_tokens": len(train_dataset) + len(test_dataset),
            "layernorm": layernorm,
            "no_processing": no_processing,
            "model_name": model_name,
        }
        for metric, value in wand_summary.items():
            wandb.run.summary[metric] = value  # type: ignore

    wandb.finish()
    sweep_id = wandb.sweep(sweep=sweep_config, project="language_rotations_pilot")
    wandb.agent(sweep_id, train, count=100)


# %%
run_sweep_for_model("bloom-560m", layernorm=False, no_processing=False)
run_sweep_for_model("bloom-560m", layernorm=True, no_processing=False)
run_sweep_for_model("bloom-560m", layernorm=True, no_processing=True)

run_sweep_for_model("bloom-3b", layernorm=False, no_processing=False)
run_sweep_for_model("bloom-3b", layernorm=True, no_processing=False)
run_sweep_for_model("bloom-3b", layernorm=True, no_processing=True)

# %%

run_sweep_for_model("gpt2-small", layernorm=False, no_processing=False)
run_sweep_for_model("gpt2-small", layernorm=True, no_processing=False)
run_sweep_for_model("gpt2-small", layernorm=True, no_processing=True)

run_sweep_for_model("gpt2-xl", layernorm=False, no_processing=False)
run_sweep_for_model("gpt2-xl", layernorm=True, no_processing=False)
run_sweep_for_model("gpt2-xl", layernorm=True, no_processing=True)

# %%
run_sweep_for_model("mistral-7b", layernorm=False, no_processing=False)
run_sweep_for_model("mistral-7b", layernorm=True, no_processing=False)
run_sweep_for_model("mistral-7b", layernorm=True, no_processing=True)

run_sweep_for_model("mistral-7b-instruct", layernorm=False, no_processing=False)
run_sweep_for_model("mistral-7b-instruct", layernorm=True, no_processing=False)
run_sweep_for_model("mistral-7b-instruct", layernorm=True, no_processing=True)
