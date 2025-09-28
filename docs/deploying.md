# Model Deployment Framework Comparison

## Introduction

Serving trained PyTorch models in production requires robust inference frameworks. In practice, many model-serving tools have been developed, each with different design trade-offs. This report compares six such frameworks—**LitServe**, **MOSEC**, **MLServer**, **NVIDIA Triton Inference Server**, **Ray Serve**, and **MLflow**—focusing on key features relevant to PyTorch deployment. 

In particular, we examine their support for:
- CPU and GPU execution
- Ability to serve multiple models (and related caching)
- Integration with manifest or configuration files
- Built-in security or authentication mechanisms

The goal is a comprehensive, side-by-side comparison to guide deployment choices in different environments.

## Feature Comparison

### CPU/GPU Support

All six frameworks support **CPU-only inference**. GPU acceleration is available in most, though the level of support varies:

- **NVIDIA Triton** explicitly supports both GPU and CPU-only modes, allowing models to run on multiple GPUs concurrently for high throughput
- **LitServe** supports GPU inference with auto-scaling across multiple GPUs
- **MOSEC** allows pipelined stages to utilize CPU and GPU devices in the same service, enabling mixed CPU/GPU workloads
- **Ray Serve** can schedule model replicas on GPUs (including fractional GPUs) using Ray's resource management and autoscaling
- **MLServer** (the Python-based Seldon inference server) can leverage GPUs through its inference runtimes (for example, using a PyTorch runtime when available)
- **MLflow's** serving (`mlflow models serve` command) is framework-agnostic: if the logged PyTorch model uses GPU tensors, it will use GPUs, but MLflow itself does not provide specialized GPU orchestration beyond the model code

### Multi-Model Support and Caching

Multi-model serving allows a single server process to host multiple models simultaneously:

- **LitServe** is designed for compound pipelines and explicitly supports serving multiple models in one server
- **MOSEC** offers a "multiple route" service mode to serve multiple model endpoints under one service
- **MLServer** was built with multi-model serving in mind: it can load multiple models in the same process and parallelize inference across them
- **NVIDIA Triton** can concurrently perform inference on multiple models using the same GPU and supports having many models in its model repository
- **Ray Serve** naturally supports multiple models via its deployment graph: one can deploy different models as separate Serve deployments or compose them into a single inference pipeline
- **MLflow's** built-in model serving focuses on a single model at a time; serving multiple models typically requires running multiple instances of the server

**Regarding caching:**
- **LitServe** provides mechanisms to cache intermediate results or use model warmup examples as static inputs, giving the developer explicit control over caching behavior
- **Ray Serve** can take advantage of Ray's in-memory object store to cache inputs or outputs across requests, though it has no dedicated caching feature for model weights
- **Triton, MLServer, and MOSEC** emphasize dynamic batching and pipelining for throughput rather than explicit caching of model outputs

### Manifest and Configuration Integration

Frameworks differ in how models are configured and loaded:

- **Triton** uses a model repository structure: each model has a directory containing the model files and a `config.pbtxt` manifest (a Protocol Buffer text file) that specifies the model's inputs, outputs, max batch size, and other settings

- **MLflow** uses an `MLmodel` YAML file in each model directory as a manifest to list the model flavors (e.g. `python_function`, `pytorch`) and their configuration

- **MLServer** is often deployed via Kubernetes InferenceService (KServe) manifests: the InferenceService YAML references an MLServer runtime that loads the model from a storage URI. Model configuration is handled through standard Kubernetes CRDs

- **LitServe, MOSEC, and Ray Serve** rely on Python code or simple CLI commands for deployment; they do not use an external manifest file format. Models and routes are defined programmatically (or via config objects) rather than via a static manifest

### Security and Authentication

Out-of-the-box security features vary significantly:

- **Triton** includes support for secure deployment: it can be configured with TLS/SSL for gRPC or HTTP endpoints, and its command-line options allow enabling SSL authentication for gRPC requests. Triton also recommends best practices like running as non-root and restricting model repository updates

- **Ray Serve** itself is a library running on a Ray cluster; it inherits the cluster's security setup. In managed environments (such as Anyscale), Ray Serve can be used with mTLS, RBAC, and monitoring, providing robust security features in production deployments

- **LitServe and MOSEC** have no built-in authentication by default: LitServe's self-managed mode leaves authentication to the user (the managed Lightning platform adds token/password options)

- **MLServer** similarly does not include its own auth mechanism, assuming it will be placed behind a secure gateway or used within a Kubernetes cluster with its own ingress security

- **MLflow's** open-source serve is very basic (no auth), although in production (e.g. Databricks Model Serving) one can enforce tokens and SSL at the service layer

## Summary Table

| Framework | CPU | GPU | Multi-Model | Manifest | Security |
|-----------|-----|-----|-------------|----------|----------|
| **LitServe** | ✅ Yes | ✅ Yes (multi-GPU autoscale) | ✅ Yes (compound pipelines) | ❌ No | ❌ None (DIY) |
| **MOSEC** | ✅ Yes | ✅ Yes (CPU/GPU pipelining) | ✅ Yes (multiple routes) | ❌ No | ❌ None |
| **MLServer** | ✅ Yes | ✅ Yes (via runtimes) | ✅ Yes | ✅ Yes (KServe/JSON) | ❌ None |
| **Triton** | ✅ Yes | ✅ Yes (multi-GPU, batching) | ✅ Yes | ✅ Yes (config.pbtxt) | ✅ TLS/SSL support |
| **Ray Serve** | ✅ Yes | ✅ Yes (fractional GPU) | ✅ Yes | ❌ No | ✅ Cluster-level (TLS/RBAC) |
| **MLflow** | ✅ Yes | ✅ Yes (depends on model) | ❌ No | ✅ Yes (MLmodel) | ⚠️ Basic (token/SSL) |

## Conclusion

In summary, all surveyed frameworks can serve PyTorch models on CPU or GPU, but differ in specialization:

- **🏆 Triton** excels at high-throughput GPU serving with rich configuration and optional TLS security
- **🚀 LitServe and MOSEC** focus on ease of use and flexibility, allowing multi-model pipelines and autoscaling (especially LitServe on GPUs)
- **☸️ MLServer** offers Kubernetes-native multi-model serving aligned with Seldon/KServe
- **🐍 Ray Serve** provides a highly scalable, Python-native approach with dynamic batching and integration into Ray clusters
- **📦 MLflow** serving is simplest and is suited for single-model endpoints packaged with an MLmodel

### Recommendations by Use Case:

- **For strict security and GPU performance:** NVIDIA Triton
- **For flexible Python pipelines:** LitServe or Ray Serve
- **For Kubernetes/MLflow integration:** MLServer or MLflow serving
- **For simple single-model deployment:** MLflow
- **For enterprise/production scale:** Ray Serve or Triton

Each framework's documentation contains further details for production deployment considerations.

## References

1. **LitServe** - Lightning AI, LitServe Documentation, lightning.ai (2023)
2. **MOSEC** - K. Yang, Z. Liu, P. Cheng, MOSEC: Model Serving made Efficient in the Cloud, 2021 (GitHub)
3. **MLServer** - Seldon Technologies, MLServer Documentation, seldon.io (2024)
4. **Triton** - NVIDIA, Triton Inference Server User Guide, docs.nvidia.com (2024)
5. **Ray Serve** - Ray Project, Ray Serve Documentation, ray.io (2024)
6. **MLflow** - Databricks, MLflow Models Documentation, mlflow.org (2024)
