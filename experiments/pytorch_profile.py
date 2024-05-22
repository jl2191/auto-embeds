# %%
import itertools
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import plotly.express as px
import torch as t
import torch.nn as nn
import transformer_lens as tl
from torch import Tensor
from torch.optim import Optimizer
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader

from auto_embeds.data import filter_word_pairs, get_dataset_path
from auto_embeds.embed_utils import (
    initialize_loss,
    initialize_transform_and_optim,
)
from auto_embeds.metrics import (
    evaluate_accuracy,
    mark_translation,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.misc import default_device
from auto_embeds.verify import (
    plot_cosine_similarity_trend,
    prepare_verify_analysis,
    prepare_verify_datasets,
    test_cos_sim_difference,
    verify_transform,
)

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYt_ENABLE_MPS_FALLBACK"] = "1"


# %%
def train_transform(
    model: tl.HookedTransformer,
    train_loader: DataLoader[Tuple[Tensor, ...]],
    test_loader: DataLoader[Tuple[Tensor, ...]],
    transform: nn.Module,
    optim: Optimizer,
    loss_module: nn.Module,
    n_epochs: int,
    plot_fig: bool = True,
    save_fig: bool = False,
    device: Union[str, t.device] = default_device,
    neptune: Optional[Any] = None,
    azure_translations_path: Optional[Union[str, Path]] = None,
) -> Tuple[nn.Module, Dict[str, List[Dict[str, Union[float, int]]]]]:
    """Trains the transformation, returning the learned transformation and loss history.

    Args:
        model: The transformer model used for training.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the test dataset.
        transform: The transformation module to be optimized.
        optim: The optimizer for the transformation.
        loss_module: The loss function used for training.
        n_epochs: The number of epochs to train for.
        plot_fig: If True, plots the training and test loss history.
        device: The device on which the model is allocated.
        neptune: If provided, log training metrics to Weights & Biases.
        azure_translations_path: Path to JSON file for mark_translation evaluation.

    Returns:
        The learned transformation after training, the train and test loss history.
    """
    train_history = {"train_loss": [], "test_loss": [], "mark_translation_score": []}
    transform.train()
    if neptune:
        neptune.watch(transform, log="all", log_freq=500)
    # if a azure_translations_path is provided we process the azure json file into a
    # more accessible format just once to speed up the marking, passing a
    # translations_dict directly into mark_translate()
    if azure_translations_path:
        with open(azure_translations_path, "r") as file:
            allowed_translations = json.load(file)
        translations_dict = {}
        for item in allowed_translations:
            source = item["normalizedSource"]
            translations = [
                trans["normalizedTarget"]
                for trans in item["translations"]
                if trans["normalizedTarget"] is not None
            ]
            translations_dict[source] = translations
    for epoch in (epoch_pbar := tqdm(range(n_epochs))):
        for batch_idx, (en_embed, fr_embed) in enumerate(train_loader):
            optim.zero_grad()
            pred = transform(en_embed)
            train_loss = loss_module(pred.squeeze(), fr_embed.squeeze())
            info_dict = {
                "train_loss": train_loss.item(),
                "batch": batch_idx,
                "epoch": epoch,
            }
            train_history["train_loss"].append(info_dict)
            train_loss.backward()
            optim.step()
            if neptune:
                neptune.log(info_dict)
            epoch_pbar.set_description(f"train loss: {train_loss.item():.3f}")
        # Calculate and log test loss at the end of each epoch
        if epoch % 2 == 0:
            with t.no_grad():
                total_test_loss = 0
                for test_en_embed, test_fr_embed in test_loader:
                    test_pred = transform(test_en_embed)
                    test_loss = loss_module(
                        test_pred.squeeze(), test_fr_embed.squeeze()
                    )
                    total_test_loss += test_loss.item()
                avg_test_loss = total_test_loss / len(test_loader)
                info_dict = {"test_loss": avg_test_loss, "epoch": epoch}
                train_history["test_loss"].append(info_dict)
                # Calculate and log mark_translation score if azure_translations_path
                if azure_translations_path:
                    mark_translation_score = mark_translation(
                        model=model,
                        transformation=transform,
                        test_loader=test_loader,
                        translations_dict=translations_dict,
                        print_results=False,
                    )
                    info_dict.update(
                        {
                            "mark_translation_score": mark_translation_score,
                        }
                    )
                    train_history["mark_translation_score"].append(info_dict)
                if neptune:
                    neptune.log(info_dict)
    if plot_fig or save_fig:
        fig = px.line(title="Train and Test Loss with Mark Correct Score")
        fig.add_scatter(
            x=[epoch_info["epoch"] for epoch_info in train_history["train_loss"]],
            y=[epoch_info["train_loss"] for epoch_info in train_history["train_loss"]],
            name="Train Loss",
        )
        fig.add_scatter(
            x=[epoch_info["epoch"] for epoch_info in train_history["test_loss"]],
            y=[epoch_info["test_loss"] for epoch_info in train_history["test_loss"]],
            name="Test Loss",
        )
        # Plot mark_translation_score if available
        if azure_translations_path:
            fig.add_scatter(
                x=[
                    epoch_info["epoch"]
                    for epoch_info in train_history["mark_translation_score"]
                ],
                y=[
                    epoch_info["mark_translation_score"]
                    for epoch_info in train_history["mark_translation_score"]
                ],
                name="Mark Correct Score",
            )
        if plot_fig:
            fig.show()
        if save_fig:
            fig.write_image("plot.png")
    return transform, train_history


# %%
# Seed for reproducibility
# np.random.seed(1)
# t.manual_seed(1)
# t.cuda.manual_seed(1)
# try:
# get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
# get_ipython().run_line_magic("load_ext", "line_profiler")  # type: ignore
# get_ipython().run_line_magic("load_ext", "scalene")  # type: ignore
# get_ipython().run_line_magic("autoreload", "2")  # type: ignore
# except Exception:
#     pass

# Configuration for experiments
config = {
    "models": ["bloom-560m"],
    "datasets": [
        # {
        #     "name": "wikdict_en_fr_extracted",
        #     "min_length": 5,
        #     "capture_space": True,
        #     "capture_no_space": False,
        #     "mark_accuracy_path": "wikdict_en_fr_azure_validation",
        # },
        # {
        #     "name": "muse_en_fr_extracted",
        #     "min_length": 5,
        #     "capture_space": True,
        #     "capture_no_space": False,
        #     "mark_accuracy_path": "muse_en_fr_azure_validation",
        # },
        {
            "name": "cc_cedict_zh_en_extracted",
            "min_length": 2,
            "capture_space": False,
            "capture_no_space": True,
            "mark_accuracy_path": "cc_cedict_zh_en_azure_validation",
        },
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
        # "linear_map",
        # "biased_linear_map",
        # "uncentered_linear_map",
        # "biased_uncentered_linear_map",
        "rotation",
        # "biased_rotation",
        # "uncentered_rotation",
        # "rotation_translation",
    ],
    "seeds": [3],
    "n_epochs": [100],
    "lr": [8e-5],
    "weight_decay": [1e-5],
    "batch_sizes": [(64, 256)],
    "top_k": [200],
}

total_runs = 1
for value in config.values():
    if isinstance(value, list):
        total_runs *= len(value)

print(f"Total runs to be executed: {total_runs}")

# %%
# Main experiment loop
last_loaded_model = None
model = None
for (
    model_name,
    dataset_config,
    transformation,
    seed,
    n_epoch,
    lr,
    weight_decay,
    batch_size,
    top_k,
) in itertools.product(
    config["models"],
    config["datasets"],
    config["transformations"],
    config["seeds"],
    config["n_epochs"],
    config["lr"],
    config["weight_decay"],
    config["batch_sizes"],
    config["top_k"],
):
    model = tl.HookedTransformer.from_pretrained_no_processing(model_name)

    assert model is not None, "The model has not been loaded successfully."

    d_model = model.cfg.d_model
    n_toks = model.cfg.d_vocab_out

    # Dataset filtering
    dataset_name = dataset_config["name"]
    file_path = get_dataset_path(dataset_name)
    with open(file_path, "r") as file:
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
        all_word_pairs=all_word_pairs,
        seed=seed,
        keep_other_pair=True,
    )

    train_loader, test_loader = prepare_verify_datasets(
        verify_learning=verify_learning,
        batch_sizes=batch_size,
        top_k=top_k,
    )

    azure_translations_path = get_dataset_path(dataset_config["mark_accuracy_path"])

    run_config = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        **dataset_config,
        "transformation_name": transformation,
        "n_epoch": n_epoch,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "top_k": top_k,
    }

    # Initialize transformation and optimizer
    transform, optim = initialize_transform_and_optim(
        d_model,
        transformation=transformation,
        optim_kwargs={"lr": lr, "weight_decay": weight_decay},
    )
    loss_module = initialize_loss("cosine_similarity")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        if optim is not None:
            transform, loss_history = train_transform(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                transform=transform,
                optim=optim,
                loss_module=loss_module,
                n_epochs=n_epoch,
                plot_fig=True,
                azure_translations_path=azure_translations_path,
            )

    prof.export_chrome_trace("trace.json")

    # Evaluate and log results
    accuracy = evaluate_accuracy(
        model,
        test_loader,
        transform,
        exact_match=False,
        print_results=False,
        print_top_preds=False,
    )

    mark_translation_acc = mark_translation(
        model=model,
        transformation=transform,
        test_loader=test_loader,
        azure_translations_path=azure_translations_path,
        print_results=True,
    )

    verify_results_dict = verify_transform(
        model=model,
        transformation=transform,
        test_loader=test_loader,
    )

    cos_sims_trend_plot = plot_cosine_similarity_trend(verify_results_dict)
    cos_sims_trend_plot.show()

    test_cos_sim_diff = test_cos_sim_difference(verify_results_dict)
