FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE /app/
COPY sprite_matting /app/sprite_matting
COPY examples /app/examples
COPY tests /app/tests

RUN pip3 install --upgrade pip && pip3 install .[dev]

VOLUME ["/checkpoints", "/data"]

ENTRYPOINT ["python3", "-m", "sprite_matting.cli"]
CMD ["--help"]
