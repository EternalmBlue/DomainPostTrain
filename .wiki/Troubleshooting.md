# 故障排查 / Troubleshooting

## 中文

当命令失败、训练中断或预期产物缺失时，使用本页。

## 先看失败报告

完整流水线失败时先打开：

```text
outputs/reports/failure_report.md
outputs/reports/pipeline_report.md
```

如果失败发生在 CPT 前，还要看：

```text
outputs/logs/preflight_report.md
outputs/logs/discovered_corpus.json
```

退出码和报告位置见 [操作手册](Operations-Runbook)。

## `ModuleNotFoundError: No module named ...`

适用：安装、冒烟测试、训练、推理。

常见原因：当前激活的 Python 环境没有安装所需包。

修复：

```bash
python -m pip install -r requirements.txt
```

仅 ONNX 相关失败：

```bash
python -m pip install -r requirements-onnx.txt
```

验证：

```bash
python -m compileall pipeline scripts serve_inference.py
python scripts/diagnostics/check_training_environment.py
```

## CUDA out of memory

适用：CPT、Fact-SFT、DPO、GRPO。

常见原因：序列长度、batch size、GRPO 生成数量或 LoRA rank 超出 GPU 显存。

修复：

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 768
  load_in_4bit: true
  gradient_checkpointing: true

peft:
  r: 8
  lora_alpha: 8
```

GRPO 还可以降低：

```yaml
grpo:
  num_generations: 2
  max_completion_length: 128
```

## `cuda_available: False`

适用：环境诊断、训练、推理、ONNX 导出。

常见原因：PyTorch CUDA wheel 与驱动不匹配、当前虚拟环境不是训练环境、没有 NVIDIA 驱动，或机器本身没有 GPU。

检查：

```bash
python scripts/diagnostics/check_training_environment.py
```

如果 `nvidia-smi` 能看到 GPU 但 `cuda_available: False`，优先重装与机器匹配的 PyTorch。真实训练不要在这个状态下继续；CPU 只适合冒烟测试。

## Corpus preflight blocked training

适用：完整流水线 CPT 前失败。

常见原因：语料中包含高风险内容、疑似密钥、内部路径、过长代码块或不适合分发的材料。

修复：

1. 打开 `outputs/logs/preflight_report.md`。
2. 删除或替换报告中的高风险内容。
3. 重新运行流水线。

只有在私有、离线、已确认数据可训练的环境里，才使用：

```bash
python scripts/training/train_pipeline.py --config configs/my_domain.yaml --allow_unsafe_corpus
```

## 没有有效 GRPO reward examples

适用：`scripts/training/train_grpo.py`。

常见原因：每条 GRPO 行需要能构造出 prompt，并至少需要一个奖励信号字段。

修复示例：

```json
{"prompt":"Answer from the documentation.","reference_answer":"Only documented facts.","required_terms":["documentation"]}
```

确认每条数据至少包含以下字段之一：

- `reference_answer`
- `required_terms`
- `forbidden_terms`
- `must_refuse`

字段别名见 [数据契约](Data-Contracts)。

## DPO 行被拒绝

适用：DPO 数据集准备。

常见原因：字段缺失、值为空，或 `chosen == rejected`。

修复：确保每行都有非空 `prompt`、`chosen`、`rejected`，并且两个回答不同。可接受字段别名见 [数据契约](Data-Contracts)。

## 未找到上游 adapter

适用：Fact-SFT、DPO、GRPO、merge。

常见原因：后续阶段需要上游 adapter，但该 adapter 尚未生成，或配置中的阶段开关没有启用对应 adapter 自动选择。

修复选项：

- 先运行上游阶段。
- 把 `base_adapter_dir` 指向已存在的 adapter。
- 合并特定 adapter 时使用 `merge.adapter_dir` 或 `merge_adapter.py --adapter_dir`。
- 只有在明确实验需要时，才把相关 `require_*_adapter` 选项设为 `false`。

## Reward judge API key 缺失

适用：`grpo.reward_judge.enabled=true` 的 GRPO。

常见原因：`api_key_env` 指定的环境变量没有设置。

修复：

```bash
export GRPO_REWARD_JUDGE_API_KEY="your-key"
```

Windows PowerShell:

```powershell
$env:GRPO_REWARD_JUDGE_API_KEY = "your-key"
```

## Reward judge 响应必须包含数值 score

适用：GRPO 外部 reward judge。

常见原因：judge 返回了纯文本、没有 JSON 对象的 markdown、字符串 score，或遗漏 `score`。

修复：更新 judge prompt 或服务，使 OpenAI-compatible 响应的 `choices[0].message.content` 包含类似 JSON：

```json
{"score": 0.8, "reason": "The answer follows the reference and avoids forbidden terms."}
```

程序只读取 `score`。HTTP body 直接返回 `{"score": 0.8}` 不够；它必须放在 chat completions envelope 的 assistant content 中。详见 [GRPO 与 Reward Judge](GRPO-and-Reward-Judge)。

## Flask 服务启动了，但 `/v1/chat/completions` 失败

常见原因：

- `--model_path` 没有指向合并后的模型。
- 模型依赖缺失。
- 请求中的 `messages` payload 格式错误。
- 服务启动在 `cuda`，但机器没有可用 CUDA。

验证：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

然后使用 [推理与导出](Inference-and-Export) 里的最小 chat completions 请求重试。CPU-only 机器启动服务时使用 `--device cpu` 或 `--device auto`。

## 静态检查通过但训练失败

`compileall` 只检查 Python 语法。它不会加载数据集、导入所有 ML 依赖、分配 GPU 显存或运行 TRL。

下一步检查：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers, datasets, peft, trl; print('training deps ok')"
python scripts/diagnostics/check_training_environment.py
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

---

## English

Use this page when a command fails, training stops, or expected artifacts are missing.

## Check Failure Reports First

For full-pipeline failures, open:

```text
outputs/reports/failure_report.md
outputs/reports/pipeline_report.md
```

If the failure happened before CPT, also check:

```text
outputs/logs/preflight_report.md
outputs/logs/discovered_corpus.json
```

See [Operations Runbook](Operations-Runbook) for exit codes and report locations.

## `ModuleNotFoundError: No module named ...`

Applies to: install, smoke test, training, inference.

Likely cause: the active Python environment does not contain the required package.

Fix:

```bash
python -m pip install -r requirements.txt
```

For ONNX-only failures:

```bash
python -m pip install -r requirements-onnx.txt
```

Verify:

```bash
python -m compileall pipeline scripts serve_inference.py
python scripts/diagnostics/check_training_environment.py
```

## CUDA Out of Memory

Applies to: CPT, Fact-SFT, DPO, GRPO.

Likely cause: sequence length, batch size, number of GRPO generations, or LoRA rank is too high for the GPU.

Fix:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 768
  load_in_4bit: true
  gradient_checkpointing: true

peft:
  r: 8
  lora_alpha: 8
```

For GRPO, also lower:

```yaml
grpo:
  num_generations: 2
  max_completion_length: 128
```

## `cuda_available: False`

Applies to: environment diagnostics, training, inference, ONNX export.

Likely cause: PyTorch CUDA wheel and driver mismatch, the wrong virtual environment, missing NVIDIA driver, or a CPU-only machine.

Check:

```bash
python scripts/diagnostics/check_training_environment.py
```

If `nvidia-smi` sees a GPU but `cuda_available: False`, reinstall the PyTorch build that matches the machine. Do not continue real training in this state; CPU is only practical for smoke tests.

## Corpus Preflight Blocked Training

Applies to: full pipeline failures before CPT.

Likely cause: corpus contains high-risk content, possible secrets, internal paths, long code blocks, or data that should not be distributed.

Fix:

1. Open `outputs/logs/preflight_report.md`.
2. Remove or replace the high-risk material.
3. Rerun the pipeline.

Use this only in a private, offline, approved training environment:

```bash
python scripts/training/train_pipeline.py --config configs/my_domain.yaml --allow_unsafe_corpus
```

## No Valid GRPO Reward Examples

Applies to: `scripts/training/train_grpo.py`.

Likely cause: each GRPO row needs a constructible prompt and at least one reward signal.

Fix:

```json
{"prompt":"Answer from the documentation.","reference_answer":"Only documented facts.","required_terms":["documentation"]}
```

Verify that each row has at least one of:

- `reference_answer`
- `required_terms`
- `forbidden_terms`
- `must_refuse`

See [Data Contracts](Data-Contracts) for accepted aliases.

## DPO Row Is Rejected

Applies to: DPO dataset preparation.

Likely cause: missing field, empty value, or `chosen == rejected`.

Fix: ensure every row has non-empty `prompt`, `chosen`, and `rejected`, and that the two answers differ. See [Data Contracts](Data-Contracts) for accepted aliases.

## Upstream Adapter Not Found

Applies to: Fact-SFT, DPO, GRPO, merge.

Likely cause: a later stage expects an upstream adapter that has not been produced, or stage switches do not select the adapter you expected.

Fix options:

- Run the upstream stage first.
- Point `base_adapter_dir` to an existing adapter.
- For a specific merge, use `merge.adapter_dir` or `merge_adapter.py --adapter_dir`.
- Set the relevant `require_*_adapter` option to `false` only for intentional experiments.

## Reward Judge API Key Is Missing

Applies to: GRPO with `grpo.reward_judge.enabled=true`.

Likely cause: the environment variable named by `api_key_env` is not set.

Fix:

```bash
export GRPO_REWARD_JUDGE_API_KEY="your-key"
```

Windows PowerShell:

```powershell
$env:GRPO_REWARD_JUDGE_API_KEY = "your-key"
```

## Reward Judge Response Must Contain Numeric Score

Applies to: GRPO external reward judge.

Likely cause: the judge returned prose, markdown without a JSON object, a string score, or omitted `score`.

Fix: update the judge prompt or service so `choices[0].message.content` in the OpenAI-compatible response contains JSON like:

```json
{"score": 0.8, "reason": "The answer follows the reference and avoids forbidden terms."}
```

The program reads only `score`. Returning `{"score": 0.8}` as the raw HTTP body is not enough; it must appear inside the assistant content of a chat completions envelope. See [GRPO And Reward Judge](GRPO-and-Reward-Judge).

## Flask Service Starts but `/v1/chat/completions` Fails

Likely causes:

- `--model_path` does not point to a merged model.
- Model dependencies are missing.
- The request uses a malformed `messages` payload.
- The service started with `cuda` on a machine without available CUDA.

Verify:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Then retry a minimal chat completions request from [Inference And Export](Inference-and-Export). On CPU-only machines, start the service with `--device cpu` or `--device auto`.

## Static Check Passes but Training Fails

`compileall` only checks Python syntax. It does not load datasets, import all ML dependencies, allocate GPU memory, or run TRL.

Next checks:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers, datasets, peft, trl; print('training deps ok')"
python scripts/diagnostics/check_training_environment.py
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```
