# Food Nutrition Inference

Production-ready assets for serving the food nutrition regression model on Azure VMs or containers. The repository only retains the inference model, the Azure ML-compatible scoring code, and a FastAPI service packaged with Docker for MLOps deployments.

## Repository Layout
- `model/`: minimal artifact bundle (`model_fp32.pt`, `training_metadata.json`, `target_normalizer.json`).
- `vm_service/app.py`: FastAPI application that loads the model artifacts and exposes `/predict`.
- `Dockerfile` & `docker-compose.yml`: containerized deployment assets.
- `requirements.txt`: lightweight inference dependencies for serving.
- `requirements-train.txt`: extras required to retrain the model.
- `requirements-dev.txt`: developer tooling and test dependencies.
- `train_food_regression.py`: training script to refresh the model artifacts locally.
- `azureml/`: Azure Machine Learning score script and environment spec.

## Retrain the Model
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-train.txt

python train_food_regression.py \
  --dataset-name mmathys/food-nutrients \
  --target-cols total_calories total_fat total_carb total_protein total_mass \
  --epochs 20 \
  --output-dir model
```

The script downloads the specified Hugging Face dataset, fine-tunes a lightweight backbone (MobileNetV3 by default), and refreshes the contents of `model/` with new weights and metadata. Adjust hyperparameters or dataset splits via `python train_food_regression.py --help`.

## Local Development
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
FOOD_MODEL_DIR=model uvicorn vm_service.app:app --host 0.0.0.0 --port 8000 --reload
```

The service exposes:
- `GET /healthz` – readiness probe.
- `POST /predict` – JSON payload with `{"instances": [{"image_base64": "..."}]}`.
- `POST /predict-file` – multipart upload for quick manual testing.

## Container Build & Run
```bash
docker build -t food-nutrition-regressor:latest .
docker run --rm -p 8000:8000 \
  -e UVICORN_PORT=8000 \
  -e FOOD_MODEL_DIR=/app/model \
  food-nutrition-regressor:latest
```

Or with docker-compose:
```bash
HOST_PORT=8000 docker compose up --build
```

Both options include container-level health checks hitting `/healthz`.

## Smoke Tests
Run the lightweight regression suite before shipping a new container or Azure deployment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pytest -q
```

## Deploy to an Azure VM
1. **Provision** an Ubuntu 22.04 VM with the necessary CPU/GPU. Open the inbound port used for inference (default `8000`) and restrict access via NSG/firewall rules.
2. **Install Docker** (recommended) following the official Microsoft docs, or alternatively Python 3.10 if you prefer running `uvicorn` directly.
3. **Copy the repo** (including `model/`) to the VM. If the model artifacts live elsewhere, set `FOOD_MODEL_DIR` accordingly.
4. **Run with Docker Compose** for managed lifecycle and automatic restart:
   ```bash
   HOST_PORT=80 docker compose up -d --build
   docker compose ps
   docker compose logs -f food-model
   ```
5. **Integrate with MLOps tooling**: attach logs to Azure Monitor, enable automated image builds (e.g., via GitHub Actions/Azure DevOps), and manage secrets (API keys, TLS) with Azure Key Vault or Docker secrets.

## Operations Checklist
- Configure HTTPS termination (Application Gateway, Nginx, or Azure Front Door).
- Rotate credentials regularly; prefer managed identity if wrapping the API with gateway auth.
- Set up alerting on container health checks, latency, and non-200 response rates.
- Automate model refresh by rebuilding the image with new artifacts and redeploying via CI/CD.
