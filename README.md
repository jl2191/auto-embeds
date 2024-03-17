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

## Licensing
This project utilizes and modifies dictionary files for its experiments, adhering to their respective licenses. Below is a list of the sources and their licenses:

- **CC-CEDICT**: A comprehensive Chinese to English dictionary. Modifications to these dictionary files are covered under the same CC BY-SA 4.0 license. For more information, visit the [CC-CEDICT Downloads Page](https://www.mdbg.net/chinese/dictionary?page=cedict).

- **WikDict**: Incorporates dictionaries available under the Creative Commons BY-SA 3.0 license. For more information, please visit the [WikDict Downloads Page](https://www.wikdict.com/page/download).

## AutoEmbeds

We're thrilled to share that, aside from the dictionary files and their derivatives, this repository is open-sourced under the MIT license. We warmly invite you to explore, modify, and distribute the software as you see fit. For more details, please refer to the LICENSE file in this repository.