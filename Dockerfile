FROM pytorch/pytorch:latest
WORKDIR /workspace
COPY poetry.lock pyproject.toml /workspace
RUN apt update \
 && sudo apt install -y vim
COPY . /workspace
ENV PATH="/workspace/.venv/bin:$PATH"
RUN poetry install --with dev
RUN wandb agent sweep_id