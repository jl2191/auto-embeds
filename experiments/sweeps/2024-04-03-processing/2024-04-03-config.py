# %%
# initialisation script for 2024-04-03-processing-worker.py
import wandb


def init_sweep():
    sweep_config = {
        "method": "random",  # or grid, bayes
        "metric": {
            "name": "accuracy",
            "goal": "maximize",
        },
        "parameters": {
            "learning_rate": {"min": 0.0001, "max": 0.1},
            "n_epochs": {"values": [50, 100, 150]},
            "top_k": {"values": [100, 200, 300]},
            "transformations": {"values": ["linear_map", "rotation"]},
            "models": {"values": ["bloom-560m", "bloom-3b"]},
            "processings": {"values": [True, False]},
            "layernorms": {"values": [False]},
        },
    }

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="language-transformations")
    print(f"Sweep ID: {sweep_id}")
    return sweep_id


if __name__ == "__main__":
    init_sweep()
