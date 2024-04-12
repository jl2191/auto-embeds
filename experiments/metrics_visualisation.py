# %%
import os

import numpy as np
import plotly.graph_objects as go
import torch as t
import transformer_lens as tl

from auto_embeds.data import prepare_data
from auto_embeds.metrics import calc_canonical_angles
from auto_embeds.utils.custom_tqdm import tqdm

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Seed for reproducibility
np.random.seed(1)
t.manual_seed(1)
t.cuda.manual_seed(1)

# Experiment configurations
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
layer_norm_configs = [False, True]

# Results storage
results = {}

# Main experiment loop
for model_name in tqdm(models, desc="Models"):
    model_results = results.setdefault(model_name, {})
    model = tl.HookedTransformer.from_pretrained_no_processing(model_name)
    for config in tqdm(dataset_configs, desc="Datasets"):
        dataset_name = config["dataset_name"]
        for apply_ln in layer_norm_configs:
            ln_key = "with_ln" if apply_ln else "without_ln"
            dataset_results = model_results.setdefault(dataset_name, {}).setdefault(
                ln_key, {"canonical_angles": [], "norms": []}
            )

            # Prepare data
            # Ensure that `filter_options` does not include unsupported arguments
            filter_options = {
                key: config[key]
                for key in ["min_length", "capture_space", "capture_no_space"]
            }
            train_en_embeds, train_fr_embeds, _, _ = prepare_data(
                model=model,
                dataset_name=dataset_name,
                return_type="tensor",
                split_ratio=0.97,
                apply_ln=apply_ln,
                seed=1,
                filter_options=filter_options,
            )

            # Calculate canonical angles and norms
            canonical_angles = (
                calc_canonical_angles(train_en_embeds, train_fr_embeds).cpu().numpy()
            )
            norms = (
                t.norm(t.cat((train_en_embeds, train_fr_embeds), dim=0), dim=1)
                .cpu()
                .numpy()
            )

            # Store results
            dataset_results["canonical_angles"].append(canonical_angles)
            dataset_results["norms"].append(norms)
# %%
# Plotting
for model_name, model_data in results.items():
    for dataset_name, dataset_data in model_data.items():
        for ln_key, metrics in dataset_data.items():
            canonical_angles = np.concatenate(metrics["canonical_angles"])
            norms = np.concatenate(metrics["norms"])

            # Canonical Angles Plot
            fig_ca = go.Figure()
            fig_ca.add_trace(
                go.Histogram(
                    x=canonical_angles,
                    nbinsx=50,
                    name=f"{model_name} {dataset_name} {ln_key}",
                )
            )
            fig_ca.update_layout(
                title=f"Canonical Angles - {model_name} {dataset_name} {ln_key}",
                xaxis_title="Canonical Angle",
                yaxis_title="Frequency",
                barmode="overlay",
            )
            fig_ca.show()

            # Norms Plot
            fig_norm = go.Figure()
            fig_norm.add_trace(
                go.Histogram(
                    x=norms, nbinsx=50, name=f"{model_name} {dataset_name} {ln_key}"
                )
            )
            fig_norm.update_layout(
                title=f"Norms - {model_name} {dataset_name} {ln_key}",
                xaxis_title="Norm",
                yaxis_title="Frequency",
                barmode="overlay",
            )
            fig_norm.show()
