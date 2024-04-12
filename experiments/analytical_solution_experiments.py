# %%
import os

from auto_embeds.metrics import evaluate_accuracy

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import plotly.graph_objects as go
import torch as t
import transformer_lens as tl
from IPython.core.getipython import get_ipython
from torch.utils.data import DataLoader, TensorDataset

from auto_embeds.data import prepare_data
from auto_embeds.metrics import calc_cos_sim_acc
from auto_embeds.modules import (
    ManualMatAddModule,
    ManualMatMulModule,
    ManualRotateTranslateModule,
)
from auto_embeds.utils.custom_tqdm import tqdm
from auto_embeds.utils.misc import repo_path_to_abs_path

ipython = get_ipython()
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)
try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("load_ext", "line_profiler")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass

# %%
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

# %%


def analytical_experiment(
    transform_type, train_en_embeds, train_fr_embeds, test_loader
):
    results_dict = {"transform_type": transform_type}
    if transform_type == "rotation":
        X = train_en_embeds.detach().clone().squeeze()
        Y = train_fr_embeds.detach().clone().squeeze()
        C = t.matmul(X.T, Y)
        U, _, V = t.svd(C)
        W = t.matmul(U, V.t())
        transform = ManualMatMulModule(W)
    elif transform_type == "rotation_translation":
        X = train_en_embeds.detach().clone().squeeze()
        Y = train_fr_embeds.detach().clone().squeeze()
        X_mean = t.mean(X, dim=0, keepdim=True)
        Y_mean = t.mean(Y, dim=0, keepdim=True)
        X_centered = X - X_mean
        Y_centered = Y - Y_mean
        C = t.matmul(X_centered.T, Y_centered)
        U, _, V = t.svd(C)
        W = t.matmul(U, V.t())
        X_rotated = t.matmul(X_centered, W)
        b = Y_mean - t.mean(X_rotated, dim=0, keepdim=True)
        transform = ManualRotateTranslateModule(W, b)
    elif transform_type == "translation":
        X = train_en_embeds.detach().clone().squeeze()
        Y = train_fr_embeds.detach().clone().squeeze()
        T = t.mean(Y - X, dim=0)
        transform = ManualMatAddModule(T)
    else:
        raise ValueError("Invalid transform type specified")

    accuracy = evaluate_accuracy(
        model,
        test_loader,
        transform,
        exact_match=False,
        print_results=False,
    )
    cos_sim_acc = calc_cos_sim_acc(test_loader, transform)

    results_dict["acc"] = f"{accuracy * 100:.2f}"
    results_dict["cos_sim_acc"] = f"{cos_sim_acc:.4f}"

    return results_dict


# %%
models = ["bloom-560m", "bloom-3b"]
dataset_configs = [
    {
        "dataset_name": "wikdict_en_fr_extracted",
        "min_length": 3,
        "capture_space": True,
        "capture_no_space": False,
    },
    {
        "dataset_name": "muse_en_fr_extracted",
        "min_length": 2,
        "capture_space": True,
        "capture_no_space": False,
    },
    {
        "dataset_name": "cc_cedict_zh_en_extracted",
        "min_length": 2,
        "capture_space": False,
        "capture_no_space": True,
    },
    {
        "dataset_name": "muse_zh_en_extracted_train",
        "min_length": 2,
        "capture_space": False,
        "capture_no_space": True,
    },
]
transform_types = ["rotation", "rotation_translation", "translation"]

# %%
results_by_model_dataset_transform = {}

for model_name in (model_name_pbar := tqdm(models, desc="Models")):
    model_results = results_by_model_dataset_transform.setdefault(model_name, {})
    model_name_pbar.set_description(f"Model: {model_name}")
    # Load the model here based on model_name
    # Example: model = load_model(model_name)
    print(f"Running experiments for model: {model_name}")
    # Assuming `load_model` is a function you define to load a model by its name
    model = tl.HookedTransformer.from_pretrained_no_processing(
        model_name
    )  # Adjusted to use the example model loading
    device = model.cfg.device
    d_model = model.cfg.d_model
    n_toks = model.cfg.d_vocab_out
    datasets_folder = repo_path_to_abs_path("datasets")
    cache_folder = repo_path_to_abs_path("datasets/activation_cache")

    for config in (
        config_pbar := tqdm(dataset_configs, desc=f"Datasets for {model_name}")
    ):
        dataset_results = model_results.setdefault(config["dataset_name"], {})
        config_pbar.set_description(f"Dataset: {config['dataset_name']}")
        train_en_embeds, train_fr_embeds, test_en_embeds, test_fr_embeds = prepare_data(
            model=model,
            dataset_name=config["dataset_name"],
            return_type="tensor",
            split_ratio=0.97,
            apply_ln=True,
            seed=1,
            filter_options={
                "discard_if_same": True,
                "min_length": config["min_length"],
                "capture_space": config["capture_space"],
                "capture_no_space": config["capture_no_space"],
                "print_number": True,
                "verbose_count": False,
            },
        )

        test_loader = DataLoader(
            TensorDataset(test_en_embeds, test_fr_embeds), batch_size=256
        )
        print(f"Dataset: {config['dataset_name']}")
        for transform_type in (
            transform_type_pbar := tqdm(
                transform_types, desc=f"Transforms for {config['dataset_name']}"
            )
        ):
            transform_type_pbar.set_description(f"Transform: {transform_type}")
            result_dict = analytical_experiment(
                transform_type, train_en_embeds, train_fr_embeds, test_loader
            )
            accuracy = result_dict["acc"]
            cos_sim_acc = result_dict["cos_sim_acc"]

            # Store results directly in the nested dictionary
            transform_results = dataset_results.setdefault(transform_type, [])
            transform_results.append((float(accuracy), float(cos_sim_acc)))

# Plotting
for model_name, datasets_results in results_by_model_dataset_transform.items():
    fig = go.Figure()
    for dataset_name, transform_results in datasets_results.items():
        transform_types = list(transform_results.keys())
        accuracies = [
            sum(acc_cos_sim[0] for acc_cos_sim in transform_results[transform])
            / len(transform_results[transform])
            for transform in transform_types
        ]
        fig.add_trace(
            go.Bar(
                name=dataset_name,
                x=transform_types,
                y=accuracies,
            )
        )

    fig.update_layout(
        title_text=f"Experiment Results for {model_name}",
        xaxis_title="Transform Type",
        yaxis_title="Accuracy",
        barmode="group",
        legend_title="Dataset",
    )
    fig.show()
