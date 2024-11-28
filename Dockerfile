FROM pytorch/pytorch AS fmp

RUN mkdir /tmp/fmp
WORKDIR /tmp/fmp


COPY README.md pyproject.toml ./
COPY src ./src
COPY tests ./tests

RUN pip install .[test,dev]

FROM fmp AS devcontainer

RUN pip uninstall fmp -y
