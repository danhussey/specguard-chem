FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY . .

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv pip install -e . && \
    rm -rf ~/.cache/uv

ENV PATH="/root/.local/bin:$PATH"

CMD ["specguard-chem", "run", "--suite", "basic", "--protocol", "L3", "--model", "heuristic", "--limit", "10"]
