 3. Ship to Azure VM

  - Copy the repo (including the refreshed model/) to the VM.
  - On the VM, install Docker/Compose if needed, then run:

    HOST_PORT=80 CONTAINER_PORT=8000 docker compose up -d --build
  - Verify from the VM: curl http://localhost:8000/healthz.

  Once that passes, open the NSG/firewall for port 80 (or whichever host port you choose) and you’re live. Let me know
  when you’re ready to add monitoring or CI/CD hooks.




  python train_food_regression.py --dataset-name mmathys/food-nutrients --train-val-test-split 0.75,0.15,0.10 --image-size 256 --batch-size 48 --epochs 45 --model efficientnet_b4 --head-dropout 0.3 --learning-rate 3e-4 --weight-decay 1e-4 --normalize-targets --freeze-backbone --freeze-epochs 3 --trainable-backbone-layers 6 --full-unfreeze-epoch 24 --unfreeze-lr-factor 0.5 --grad-clip-norm 1.0 --early-stop-patience 8 --early-stop-min-delta 0.01 --output-dir model --seed 42



------------------------------------------------------------------------------------------------------------------------------------

From L:\ML_Food:

    python -m venv .venv
    .\.venv\Scripts\activate           # Windows PowerShell: .\.venv\Scripts\Activate.ps1
    pip install --upgrade pip
    pip install -r requirements-train.txt
    This pulls inference + training deps (torch/torchvision, datasets, scikit‑learn, tqdm).

  2. Launch the training job

  - Recommended command (ImageNet init, aggressive augmentation, progressive unfreezing):

    python train_food_regression.py --dataset-name mmathys/food-nutrients --train-val-test-split 0.75,0.15,0.10 --image-size 256 --batch-size 48 --epochs 45 --model efficientnet_b4 --head-dropout 0.3 --learning-rate 3e-4 --weight-decay 1e-4 --normalize-targets --freeze-backbone --freeze-epochs 3 --trainable-backbone-layers 6 --full-unfreeze-epoch 24 --unfreeze-lr-factor 0.5 --grad-clip-norm 1.0 --early-stop-patience 8 --early-stop-min-delta 0.01 --output-dir model --seed 42
