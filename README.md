# MLOps Zoomcamp 2023 Week 3

![Prefect logo](./images/logo.svg)

---

This repo contains Python code to accompany the videos that show how to use Prefect for MLOps. We will create workflows that you can orchestrate and observe..

# Setup

## Clone the repo

Clone the repo locally.

## Install packages

Create a conda environment:
```bash
conda create -n prefect-ops3 python=3.11 pip
conda activate prefect-ops3
```

In the conda environment, install all package dependencies with:
```bash
pip install -r requirements.txt
```
## Start the Prefect server locally

Create another window and activate your conda environment. Start the Prefect API server locally with 

```bash
prefect server start
```

## Optional: use Prefect Cloud for added capabilties
Signup and use for free at https://app.prefect.cloud