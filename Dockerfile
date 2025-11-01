FROM python:3.10-slim

ENV APP_HOME=/app
WORKDIR ${APP_HOME}

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY azureml ./azureml
COPY vm_service ./vm_service
COPY model ./model

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser ${APP_HOME}
USER appuser

ENV FOOD_MODEL_DIR=${APP_HOME}/model
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://127.0.0.1:${UVICORN_PORT}/healthz || exit 1

CMD ["sh", "-c", "uvicorn vm_service.app:app --host ${UVICORN_HOST:-0.0.0.0} --port ${UVICORN_PORT:-8000}"]
