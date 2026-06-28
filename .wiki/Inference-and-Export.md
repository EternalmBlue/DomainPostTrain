# Inference And Export

Use this page after training or when you want to serve or export a merged model.

## Merge adapter into a full model

The pipeline can merge automatically. When `merge.adapter_dir` is `null`, merge selects the first available adapter in this order:

```text
GRPO -> DPO -> Fact-SFT -> CPT
```

Manual merge entrypoint:

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.yaml
```

Expected output:

```text
outputs/merged_model/
```

## Run single-text inference

```bash
python scripts/inference/run_inference.py --config configs/domain_post_training.yaml --model_path outputs/merged_model "What can this assistant answer from the documentation?"
```

Expected result: the script loads `outputs/merged_model` and prints the model completion.

## Start the Flask service

```bash
python serve_inference.py --host 0.0.0.0 --port 8000 --model_path outputs/merged_model
```

Service endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /openapi.json` | OpenAPI schema. |
| `GET /docs` | Browser docs page. |
| `GET /v1/models` | OpenAI-compatible model list. |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions. |
| `POST /generate` | Simple generation endpoint. |

OpenAI-compatible request example:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-domain-model",
    "messages": [{"role": "user", "content": "What can this assistant answer from the documentation?"}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

## Export GGUF

GGUF is the recommended default local-deployment export path. The project does not download a third-party GGUF reference.

```bash
python scripts/model_artifacts/export_gguf.py --config configs/domain_post_training.yaml --backend llama_cpp --llama_cpp_dir /path/to/llama.cpp
```

Output is controlled by:

```yaml
gguf:
  output_dir: "models/gguf"
  output_name: "DomainPostTrain-Q4_K_M.gguf"
  quantization_method: "q4_k_m"
```

## Export ONNX

Install optional dependencies first:

```bash
python -m pip install -r requirements-onnx.txt
```

Then run:

```bash
python scripts/model_artifacts/export_onnx.py --config configs/domain_post_training.yaml
```

ONNX export settings live under `onnx` in `configs/domain_post_training.yaml`.

## Related pages

- [Installation](Installation)
- [Configuration](Configuration)
- [Troubleshooting](Troubleshooting)

