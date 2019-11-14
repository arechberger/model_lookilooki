# Mlflow

# Logging to a Remote Tracking Server
To manage results centrally or share them across a team.

Options:
- Set up tracking server on remote machine
- Databricks Community Edition
    - requires account
    - free service that includes a hosted tracking server

## Server Installation
follow instructions from https://mc.ai/setup-mlflow-in-production/
Careful: `mlflow-tracking.service` is messed up
this is what worked:
```sh
[Unit]
Description=MLflow Tracking Server
After=network.target


[Service]
Restart=on-failure
RestartSec=30
StandardOutput=file:/root/mlflow/mllogs/stdout.log
StandardError=file:/root/mlflow/mllogs/stderr.log
User=root
ExecStart=/bin/bash -c 'PATH=/root/miniconda3/envs/mlflow_env/bin/:$PATH exec mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root sftp://mlflow_user@lookilooki:/root/mlflow/mlruns -h 0.0.0.0 -p 8000'


[Install]
WantedBy=multi-user.target
```
