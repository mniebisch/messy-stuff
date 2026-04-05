FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime AS base

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN mkdir /tmp/fmp
WORKDIR /tmp/fmp

COPY requirements.txt ./
RUN pip install -r requirements.txt

FROM base AS fmp

COPY README.md pyproject.toml ./
COPY src ./src
COPY tests ./tests

RUN pip install .

RUN useradd --create-home user
USER user

FROM base AS devcontainer

ENV PYTHONPATH=src:${PYTHONPATH}

RUN useradd --create-home --shell /bin/bash devuser
USER devuser