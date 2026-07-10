FROM debian:bookworm-slim AS assets

ARG LLAMA_VERSION=b9952
ARG LLAMA_SHA256=5838c7d1f93cfebb91bf89eaaf1a4c173ff92f5b3b1271d00a93dd8f4581919c
ARG MODEL_SHA256=57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8
ARG MODEL_URL=https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf?download=true

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fL --retry 5 --retry-delay 3 \
      "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_VERSION}/llama-${LLAMA_VERSION}-bin-ubuntu-x64.tar.gz" \
      -o /tmp/llama.tar.gz \
    && echo "${LLAMA_SHA256}  /tmp/llama.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/llama \
    && tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1 \
    && rm /tmp/llama.tar.gz

RUN mkdir -p /models \
    && curl -fL --retry 5 --retry-delay 5 "${MODEL_URL}" -o /models/Qwen3.5-2B-Q4_K_M.gguf \
    && echo "${MODEL_SHA256}  /models/Qwen3.5-2B-Q4_K_M.gguf" | sha256sum -c -


FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH=/opt/llama \
    LOCAL_MODEL_PATH=/models/Qwen3.5-2B-Q4_K_M.gguf \
    LLAMA_SERVER_PATH=/opt/llama/llama-server

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=assets /opt/llama /opt/llama
COPY --from=assets /models /models
COPY main.py .

RUN mkdir -p /input /output

ENTRYPOINT ["python", "main.py"]
