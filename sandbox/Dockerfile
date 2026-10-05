# Isolated runner for LLM-generated code. Run with --network none (done by app/tools/sandbox.py)
FROM python:3.12-slim
RUN pip install --no-cache-dir pandas numpy scipy scikit-learn matplotlib
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp
WORKDIR /work
