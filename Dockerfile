FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --no-cache-dir -r requirements.txt && pip install -e .
CMD ["python", "scripts/run_all_experiments.py", "--config", "configs/smoke.yaml", "--outdir", "results/smoke"]
