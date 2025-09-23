FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
CMD ["specguard-chem", "run", "--suite", "basic", "--protocol", "L3", "--model", "heuristic", "--limit", "10"]
