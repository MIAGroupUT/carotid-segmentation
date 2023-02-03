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

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

COPY --chown=algorithm:algorithm src/ /opt/algorithm/src/
RUN python -m pip install --user ./src

COPY --chown=algorithm:algorithm models/ /opt/algorithm/models/

ENTRYPOINT carotid pipeline_transform /input ./models/heatmap_transform ./models/segmentation_transform /output $0 $@
