# %%
# import numpy as np
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from auto_embeds.utils.misc import (
    create_parallel_categories_plot,
    fetch_wandb_runs_as_df,
)

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore
except Exception:
    pass


# %%
## run that was specifically for random and singular plural
# fetching data and creating DataFrame
original_df = fetch_wandb_runs_as_df(
    project_name="jl2191/language-transformations",
    tags=["actual", "2024-04-17 random and singular plural", "run group 2"],
    custom_labels={
        "transformation": {"rotation": "Rotation", "linear_map": "Linear Map"},
        "dataset": {
                "wikdict_en_fr_extracted": "wikdict_en_fr",
                "cc_cedict_zh_en_extracted": "cc_cedict_zh_en",
            },
        }
)

# %%
create_parallel_categories_plot(
    df=original_df,
    dimensions=[
        "transformation",
        "dataset",
        "seed",
        "embed_apply_ln",
        "transform_apply_ln",
        "unembed_apply_ln",
    ],
    color="max_test_accuracy",
    labels={
        "transformation": "Transformation",
        "dataset": "Dataset",
        "embed_apply_ln": "Embed LN",
        "transform_apply_ln": "Transform LN",
        "unembed_apply_ln": "Unembed LN",
    },
    title="Max Test Accuracy by LayerNorm",
    annotation_text="So here it seems like seed is the parameter that is making the "
    "most difference on our test metric. As such, averaging over this for our next "
    "graph may let us see the best performing combinations.",
)