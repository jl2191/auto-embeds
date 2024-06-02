# AutoEmbeds
AutoEmbeds is a library focused on exploring and experimenting with embedding transformations in transformer models.

## Getting Started
To set up your environment and start experimenting with AutoEmbeds, follow these steps:

- Ensure you have [poetry](https://python-poetry.org/docs/#installation) installed on your system.
- Clone the AutoEmbeds repository to your local machine.
- Navigate to the cloned repository directory and run `poetry install --with dev` to install all necessary dependencies.

``` bash
curl -sSL https://install.python-poetry.org | python3 -

git clone https://github.com/jl2191/AutoEmbeds.git

cd AutoEmbeds

poetry install --with dev
```

Poetry is configured to use system packages by default, which can be beneficial when working on systems with pre-installed packages like PyTorch. To change this behavior, set `options.system-site-packages` to `false` in `poetry.toml`.

## Features

### Performance Tricks

#### Caching with `@auto_embeds_cache`

AutoEmbeds leverages a custom caching mechanism to optimize performance for functions with expensive or frequently repeated computations. This is achieved using the `@auto_embeds_cache` decorator from `auto_embeds/utils/cache.py`.

- **Caching Mechanism**: The cache directory is determined by the environment variable `AUTOEMBEDS_CACHE_DIR` which defaults to `/tmp`.
- **Clearing Cache**: You can clear the cache for a specific function by calling the `clear_cache` method on the decorated function. To delete all cached data, you just simply delete the entire cache directory.
- **Disabling Cache**: The caching mechanism can be turned off entirely by setting the environment variable `AUTOEMBEDS_CACHING` to `false`.

Example usage:
```python
from auto_embeds.utils.cache import auto_embeds_cache

@auto_embeds_cache
def expensive_function(param1, param2):
    # expensive goodies here
    return computation_result

# after the first run, it should be super speedy!
result = expensive_function(1, 2)

# clear the cache for this specific function should you want to.
expensive_function.clear_cache()
```

#### Parallel Processing

AutoEmbeds also supports parallel processing to speed up experiments. This is demonstrated in `experiments/run_experiment.py` where multiprocessing is used to run experiments in parallel.

- **Parallel Execution**: The `run_experiment_parallel` function utilizes Python's `multiprocessing` module to distribute tasks across multiple workers, significantly reducing the time required for large-scale experiments.

Example usage:
```python
from experiments.run_experiment import run_experiment_parallel, experiment_config, num_workers

if __name__ == "__main__":
    run_experiment_parallel(experiment_config, num_workers)
```

## Contributing
Contributions are welcome! Here are some guidelines to follow:

- Type checking is enforced using [Pyright](https://github.com/microsoft/pyright). Please include type hints in all function definitions.
- Write tests using [Pytest](https://docs.pytest.org/en/stable/).
- Format your code with [Black](https://github.com/psf/black).
- Lint your code using [ruff](https://github.com/astral-sh/ruff).

To check / fix your code run:
```bash
pre-commit run --all-files
```
Install the git hook with:
``` bash
pre-commit install
```
To execute tests labeled as slow, run:
``` bash
pytest --runslow
```
To execute tests marked as slow and are benchmarked, run:
``` bash
pytest --runslow --benchmark-only
```

## Licensing
This project utilizes and modifies dictionary files for its experiments, adhering to their respective licenses. Below is a list of the sources and their licenses:

- **CC-CEDICT**: A comprehensive Chinese to English dictionary. Modifications to these dictionary files are covered under the same CC BY-SA 4.0 license. For more information, visit the [CC-CEDICT Downloads Page](https://www.mdbg.net/chinese/dictionary?page=cedict).

- **WikDict**: Incorporates dictionaries available under the Creative Commons BY-SA 3.0 license. For more information, please visit the [WikDict Downloads Page](https://www.wikdict.com/page/download).

We're thrilled to share that, aside from the dictionary files and their derivatives, this repository is open-sourced under the MIT license. We warmly invite you to explore, modify, and distribute the software as you see fit. For more details, please refer to the LICENSE file in this repository.
