 3. Ship to Azure VM

  - Copy the repo (including the refreshed model/) to the VM.
  - On the VM, install Docker/Compose if needed, then run:

    HOST_PORT=80 CONTAINER_PORT=8000 docker compose up -d --build
  - Verify from the VM: curl http://localhost:8000/healthz.

  Once that passes, open the NSG/firewall for port 80 (or whichever host port you choose) and you’re live. Let me know
  when you’re ready to add monitoring or CI/CD hooks.