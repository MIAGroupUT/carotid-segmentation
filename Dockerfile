FROM pytorch/pytorch

RUN apt update
RUN apt install build-essential -y

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm docker-utils/docker-requirements.txt /opt/algorithm/
RUN python -m pip install --user -r docker-requirements.txt

COPY --chown=algorithm:algorithm src/ /opt/algorithm/src/
RUN python -m pip install --user ./src

COPY --chown=algorithm:algorithm docker-utils/commands.sh /opt/algorithm/commands.sh

# Include model weights in the Docker
COPY --chown=algorithm:algorithm models /opt/algorithm/models/
COPY --chown=algorithm:algorithm docker-utils/refactor_outputs.py /opt/algorithm

ENTRYPOINT sh ./commands.sh $0 $@