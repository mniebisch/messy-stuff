FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime AS fmp

RUN mkdir /tmp/fmp
WORKDIR /tmp/fmp


COPY README.md pyproject.toml ./
COPY src ./src
COPY tests ./tests

RUN pip install .

FROM fmp AS fmp_dev

RUN pip install .[test,dev]

FROM fmp AS devcontainer

RUN pip uninstall fmp -y

RUN useradd --create-home devuser
USER devuser