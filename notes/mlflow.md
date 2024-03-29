# Mlflow

## MLflow Tracking
- runs
- runs can be grouped in experiments

### record the following types of information information
- Parameters
    - key value pair of string
- Metrics
    - key value (numeric)
    - can be updated throughout the run
    - visualization in mlflow
- Artifacts
    - output files in any format

### Where is it saved
- local files
- database
- remote Tracking Server

### Demo Notebook
`01_mlflow_tryout`

### Logging to a Remote Tracking Server
To manage results centrally or share them across a team.

Options:
- Set up tracking server on remote machine
- Databricks Community Edition
    - requires account
    - free service that includes a hosted tracking server

#### Server Installation
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
StandardOutput=file:/mlflow/mllogs/stdout.log
StandardError=file:/mlflow/mllogs/stderr.log
User=root
ExecStart=/bin/bash -c 'PATH=/root/miniconda3/envs/mlflow_env/bin/:$PATH exec mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root sftp://mlflow_user@lookilooki.root.sx:/mlflow/mlruns -h 0.0.0.0 -p 8000'


[Install]
WantedBy=multi-user.target
```
#### Notes to sftp artifact transmission
- id_ed25519 not supported
- id_rsa with passphrase not supported
- on mac use `ssh-keygen -t rsa -b 4096 -C "email@email.com" -m PEM`

source: https://github.com/paramiko/paramiko/issues/340#issuecomment-492448662

> This is a problem with recent macOS, whose OpenSSL ssh-keygen has changed the default format. Now a generated SSH key starts with:
> 
> ```
> -----BEGIN OPENSSH PRIVATE KEY-----
> ```
> 
> instead of the supported
> 
> ```
> -----BEGIN RSA PRIVATE KEY-----
> ```
> 
> To generate a supported key, use the following command:
> 
> ```
> ssh-keygen -t rsa -b 4096 -C "email@email.com" -m PEM
> ```


### Has integrations with popular ml libraries
e.g. [pytorch](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#module-mlflow.pytorch):
- `mlflow.pytorch.log_model(pytorch_model, artifact_path, conda_env=None, ...)`
Log a PyTorch model as an MLflow artifact for the current run.

- `mlflow.pytorch.load_model(model_uri)`

### example of a mlflow run using pytorch
- uses tensorboard as well
https://github.com/mlflow/mlflow/tree/master/examples/pytorch

