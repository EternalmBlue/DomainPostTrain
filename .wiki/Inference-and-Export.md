# 推理与导出 / Inference And Export

## 中文

当你已经完成训练，或需要启动推理服务、合并 adapter、导出模型时，使用本页。

## 合并 adapter 到完整模型

流水线可以自动合并。`merge.adapter_dir` 为 `null` 且没有传入 `--adapter_dir` 时，合并阶段按启用阶段选择 adapter：

```text
enabled GRPO -> enabled DPO -> enabled Fact-SFT -> CPT
```

如果要合并某个已存在的 adapter，直接指定路径：

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.yaml --adapter_dir outputs/grpo_adapter
```

预期输出：

```text
outputs/merged_model/
```

## 单条文本推理

```bash
python scripts/inference/run_inference.py --config configs/domain_post_training.yaml --model_path outputs/merged_model --device auto "What can this assistant answer from the documentation?"
```

预期结果：脚本加载 `outputs/merged_model` 并打印模型回答。

常用参数：

| 参数 | 用途 |
|---|---|
| `--device cuda|cuda:0|cpu|auto` | 选择推理设备；默认是 `cuda`。 |
| `--dtype auto|float16|bfloat16|float32` | 覆盖加载 dtype。 |
| `--system_prompt` | 覆盖默认 system prompt。 |
| `--raw_prompt` | 不套 chat prompt，直接使用输入文本。 |
| `--allow_markdown` | 不清理 markdown。 |
| `--return_reasoning` | 调试时返回模型输出中的 `<think>` 内容。 |

## 启动 Flask 服务

```bash
python serve_inference.py --host 0.0.0.0 --port 8000 --model_path outputs/merged_model --device auto
```

服务端点：

| Endpoint | 用途 |
|---|---|
| `GET /health` | 健康检查，返回模型路径、served model name、设备和 OpenAI-compatible 状态。 |
| `GET /openapi.json` | OpenAPI schema。 |
| `GET /docs` | 浏览器文档页。 |
| `GET /v1/models` | OpenAI-compatible 模型列表。 |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions。 |
| `POST /generate` | 简单生成接口。 |
| `POST /` | `/generate` 的别名。 |

健康检查：

```bash
curl http://localhost:8000/health
```

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

常用服务参数：

| 参数或环境变量 | 用途 |
|---|---|
| `--device` / `FLASK_INFRA_DEVICE` | 默认 `cuda`，可设 `auto` 或 `cpu`。 |
| `--dtype` / `FLASK_INFRA_DTYPE` | 默认 `float16`。 |
| `--served_model_name` | `/v1/models` 和 chat completions 中暴露的模型名。 |
| `--raw_prompt` | 服务端不自动套 prompt。 |
| `--allow_markdown` | 不清理 markdown。 |
| `--return_reasoning` | 调试时返回 reasoning 字段。 |

兼容性边界：服务提供基础 OpenAI-compatible chat completions；不要假设支持 streaming、tools/function calling 或多模态输入。

## 导出 GGUF

GGUF 是推荐的默认本地部署导出路径。运行 GGUF 导出前，先准备本地 `llama.cpp` 目录或使用可用的 Unsloth backend。

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

默认配置中的 `onnx.device` 是 `cuda`。CPU-only 机器请传入 `--device cpu` 或修改配置：

```bash
python scripts/model_artifacts/export_onnx.py --config configs/domain_post_training.yaml --device cpu
```

常用参数：

| 参数 | 用途 |
|---|---|
| `--model_path` | 合并模型目录，默认 `training.merged_output_dir`。 |
| `--output_dir` | ONNX 输出目录，默认 `onnx.output_dir`。 |
| `--opset` | ONNX opset，默认配置为 `17`。 |
| `--external_data` / `--no_external_data` | 是否使用 ONNX external data。 |
| `--validate` / `--no_validate` | 是否运行 `onnx.checker`。 |
| `--ort_check` | 运行 ONNX Runtime shape check。 |

ONNX 导出设置位于 `configs/domain_post_training.yaml` 的 `onnx` 配置块。

## 相关页面

- [安装](Installation)
- [配置](Configuration)
- [操作手册](Operations-Runbook)
- [故障排查](Troubleshooting)

---

## English

Use this page after training or when you want to serve, merge, or export a model.

## Merge Adapter into a Full Model

The pipeline can merge automatically. When `merge.adapter_dir` is `null` and no `--adapter_dir` argument is provided, merge selects among enabled stages:

```text
enabled GRPO -> enabled DPO -> enabled Fact-SFT -> CPT
```

To merge a specific existing adapter, pass it directly:

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.yaml --adapter_dir outputs/grpo_adapter
```

Expected output:

```text
outputs/merged_model/
```

## Run Single-Text Inference

```bash
python scripts/inference/run_inference.py --config configs/domain_post_training.yaml --model_path outputs/merged_model --device auto "What can this assistant answer from the documentation?"
```

Expected result: the script loads `outputs/merged_model` and prints the model completion.

Common flags:

| Flag | Purpose |
|---|---|
| `--device cuda|cuda:0|cpu|auto` | Select inference device; default is `cuda`. |
| `--dtype auto|float16|bfloat16|float32` | Override load dtype. |
| `--system_prompt` | Override the default system prompt. |
| `--raw_prompt` | Use input text directly without chat prompt wrapping. |
| `--allow_markdown` | Do not strip markdown. |
| `--return_reasoning` | Return model-emitted `<think>` content for debugging. |

## Start the Flask Service

```bash
python serve_inference.py --host 0.0.0.0 --port 8000 --model_path outputs/merged_model --device auto
```

Service endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check with model path, served model name, device, and OpenAI-compatible status. |
| `GET /openapi.json` | OpenAPI schema. |
| `GET /docs` | Browser docs page. |
| `GET /v1/models` | OpenAI-compatible model list. |
| `POST /v1/chat/completions` | OpenAI-compatible chat completions. |
| `POST /generate` | Simple generation endpoint. |
| `POST /` | Alias for `/generate`. |

Health check:

```bash
curl http://localhost:8000/health
```

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

Common service flags:

| Flag or env var | Purpose |
|---|---|
| `--device` / `FLASK_INFRA_DEVICE` | Defaults to `cuda`; set `auto` or `cpu` when needed. |
| `--dtype` / `FLASK_INFRA_DTYPE` | Defaults to `float16`. |
| `--served_model_name` | Model name exposed by `/v1/models` and chat completions. |
| `--raw_prompt` | Do not wrap prompts server-side. |
| `--allow_markdown` | Do not strip markdown. |
| `--return_reasoning` | Return a reasoning field for debugging. |

Compatibility boundary: the service provides basic OpenAI-compatible chat completions. Do not assume streaming, tools/function calling, or multimodal input support.

## Export GGUF

GGUF is the recommended default local-deployment export path. Prepare a local `llama.cpp` checkout or use an available Unsloth backend before running GGUF export.

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

The default config sets `onnx.device` to `cuda`. On CPU-only machines, pass `--device cpu` or change the config:

```bash
python scripts/model_artifacts/export_onnx.py --config configs/domain_post_training.yaml --device cpu
```

Common flags:

| Flag | Purpose |
|---|---|
| `--model_path` | Merged model directory; defaults to `training.merged_output_dir`. |
| `--output_dir` | ONNX output directory; defaults to `onnx.output_dir`. |
| `--opset` | ONNX opset; default config is `17`. |
| `--external_data` / `--no_external_data` | Toggle ONNX external data. |
| `--validate` / `--no_validate` | Toggle `onnx.checker` validation. |
| `--ort_check` | Run an ONNX Runtime shape check. |

ONNX export settings live under `onnx` in `configs/domain_post_training.yaml`.

## Related Pages

- [Installation](Installation)
- [Configuration](Configuration)
- [Operations Runbook](Operations-Runbook)
- [Troubleshooting](Troubleshooting)
