# XGBoost DJL Serving GKE Marketplace Application

This application deploys an XGBoost model using Deep Java Library (DJL) Serving on Google Kubernetes Engine (GKE).

## Prerequisites

- A GKE cluster.
- `helm` and `kubectl` installed (for local testing).

## Configuration

The following parameters can be configured:

- `model.url`: The URL of the XGBoost model to serve. Default: `s3://djl-ai/models/xgb/xgb.zip`
- `replicaCount`: Number of replicas for the deployment. Default: `1`
- `image.tag`: The DJL Serving image tag. Default: `cpu-full`

## Local Testing

View generated template

```bash
helm template my-xgboost ./djl-serving > xgb-manifest.yaml
```

To test the Helm chart locally:

```bash
helm install my-xgboost ./djl-serving --set model.url=s3://djl-ai/models/xgb/xgb.zip
```

Clean up
```bash
helm list
helm uninstall my-xgboost
```

## Marketplace Deployment

This package is structured for Google Cloud Marketplace. It includes `schema.yaml` and `manifest.yaml` which define the UI and deployment structure.
