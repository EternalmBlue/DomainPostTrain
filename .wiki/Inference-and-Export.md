# 推理与导出 / Inference And Export

## 中文

当你已经完成训练，或需要启动推理服务、合并 adapter、导出模型时，使用本页。

## 合并 adapter 到完整模型

流水线可以自动合并。`merge.adapter_dir` 为 `null` 时，合并阶段按以下顺序选择第一个可用 adapter：

```text
GRPO -> DPO -> Fact-SFT -> CPT
```

手动合并入口：

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.yaml
```

预期输出：

```text
outputs/merged_model/
```

## 单条文本推理

```bash
python scripts/inference/run_inference.py --config configs/domain_post_training.yaml --model_path outputs/merged_model "What can this assistant answer from the documentation?"
```

预期结果：脚本加载 `outputs/merged_model` 并打印模型回答。

## 启动 Flask 服务

```bash
python serve_inference.py --host 0.0.0.0 --port 8000 --model_path outputs/merged_model
```

服务端点：

| Endpoint | 用途 |
|---|---|
| `GET /openapi.json` | OpenAPI schema。 |
| `GET /docs` | 浏览器文档页。 |
| `GET /v1/models` | OpenAI-compatible 模型列表。 |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions。 |
| `POST /generate` | 简单生成接口。 |

OpenAI-compatible 请求示例：

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

## 导出 GGUF

GGUF 是推荐的默认本地部署导出路径。项目不会下载第三方 GGUF reference。

```bash
python scripts/model_artifacts/export_gguf.py --config configs/domain_post_training.yaml --backend llama_cpp --llama_cpp_dir /path/to/llama.cpp
```

输出由以下配置控制：

```yaml
gguf:
  output_dir: "models/gguf"
  output_name: "DomainPostTrain-Q4_K_M.gguf"
  quantization_method: "q4_k_m"
```

## 导出 ONNX

先安装可选依赖：

```bash
python -m pip install -r requirements-onnx.txt
```

再运行：

```bash
python scripts/model_artifacts/export_onnx.py --config configs/domain_post_training.yaml
```

ONNX 导出设置位于 `configs/domain_post_training.yaml` 的 `onnx` 配置块。

## 相关页面

- [安装](Installation)
- [配置](Configuration)
- [故障排查](Troubleshooting)

---

## English

Use this page after training or when you want to serve or export a merged model.

## Merge Adapter into a Full Model

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

## Run Single-Text Inference

```bash
python scripts/inference/run_inference.py --config configs/domain_post_training.yaml --model_path outputs/merged_model "What can this assistant answer from the documentation?"
```

Expected result: the script loads `outputs/merged_model` and prints the model completion.

## Start the Flask Service

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

## Related Pages

- [Installation](Installation)
- [Configuration](Configuration)
- [Troubleshooting](Troubleshooting)
