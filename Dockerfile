FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    MINIT2I_SOURCE_ROOT=/app

WORKDIR /app

COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir --break-system-packages -r requirements-api.txt

COPY diffusers ./diffusers
COPY api ./api
COPY LICENSE README.md ./

RUN mkdir -p /cache/huggingface

EXPOSE 8092

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1"]
