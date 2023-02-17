FROM python:3.9-slim

RUN apt update
RUN apt install build-essential -y

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip
RUN python -m pip install numpy==1.22.4

COPY --chown=algorithm:algorithm docker-utils/docker-requirements.txt /opt/algorithm/
RUN python -m pip install --user -r docker-requirements.txt

COPY --chown=algorithm:algorithm src /opt/algorithm/src/
RUN python -m pip install --user ../src

# Include model weights in the Docker
COPY --chown=algorithm:algorithm models /opt/algorithm/models/
COPY --chown=algorithm:algorithm docker-utils/refactor_outputs.py /opt/algorithm

ENTRYPOINT carotid pipeline_transform /input ./models/heatmap_transform ./models/contour_transform /output $0 $@
