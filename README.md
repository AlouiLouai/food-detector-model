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
  --epochs 50 \
  --model efficientnet_b4 \
  --head-dropout 0.3 \
  --output-dir model
```

The script downloads the specified Hugging Face dataset, fine-tunes a lightweight backbone (EfficientNet-B0 remains the default; the example above opts into EfficientNet-B4), and refreshes the contents of `model/` with new weights and metadata. Adjust hyperparameters or dataset splits via `python train_food_regression.py --help`.

To follow a "freeze then progressively unfreeze" routine, append for example:

```bash
  --freeze-backbone \
  --freeze-epochs 3 \
  --trainable-backbone-layers 6 \
  --full-unfreeze-epoch 24 \
  --unfreeze-lr-factor 0.5 \
  --grad-clip-norm 1.0 \
  --early-stop-patience 8 \
  --early-stop-min-delta 0.01
```

### Choosing a Backbone

The `--model` flag now supports modern pretrained encoders (`resnet50`, `efficientnet_b0/b3`, `convnext_base`, `vit_b_16`, etc.). By default, ImageNet weights are loaded and the final classification layer is replaced with a dropout-backed regression head that outputs calories, fat, carbs, protein, and mass. Useful extras:

- `--no-pretrained` – train from scratch (not recommended unless you have a large domain dataset).
- `--freeze-backbone` – fine-tune only the regression head on top of frozen pretrained features.
- `--trainable-backbone-layers 4` – when used with `--freeze-backbone`, unfreeze the last 4 backbone parameter groups for deeper fine-tuning.
- `--head-dropout 0.3` – adjust regularisation strength before the regression head.

### Training Enhancements

Recent updates add several quality-of-life improvements:

- Stronger data augmentation (random crops, flips, affine jitter, and color jitter) enabled by default to improve generalisation. Disable with `--no-augment`.
- Automatic mixed precision on CUDA for faster, more stable optimisation (toggle off with `--no-amp`).
- ReduceLROnPlateau learning-rate scheduling applied each epoch unless `--no-lr-scheduler` is supplied.
- Support for deeper backbones such as `efficientnet_b4` in addition to the previous options.
- Progressive backbone fine-tuning when `--freeze-backbone` is set: warm up on the regression head for a few epochs (`--freeze-epochs`), optionally unfreeze the last `n` backbone blocks (`--trainable-backbone-layers`), and automatically unfreeze the entire encoder later in training (`--full-unfreeze-epoch`). Learning rates for all parameter groups decay by `--unfreeze-lr-factor` after each unfreeze stage.
- Optional gradient clipping via `--grad-clip-norm` and richer per-target validation metrics saved to `training_metadata.json`.
- Early stopping support (`--early-stop-patience`, `--early-stop-min-delta`, `--early-stop-warmup`) so long runs halt automatically once the validation loss stops improving.

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
   If the API serves browser clients, add `FOOD_API_ALLOWED_ORIGINS=http://localhost:3000` (comma-separated for multiples) to your `.env` so CORS headers are emitted.
5. **Integrate with MLOps tooling**: attach logs to Azure Monitor, enable automated image builds (e.g., via GitHub Actions/Azure DevOps), and manage secrets (API keys, TLS) with Azure Key Vault or Docker secrets.

## Operations Checklist
- Configure HTTPS termination (Application Gateway, Nginx, or Azure Front Door).
- Rotate credentials regularly; prefer managed identity if wrapping the API with gateway auth.
- Set up alerting on container health checks, latency, and non-200 response rates.
- Automate model refresh by rebuilding the image with new artifacts and redeploying via CI/CD.
