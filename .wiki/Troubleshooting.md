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
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --allow_unsafe_corpus
```

## 没有有效 GRPO reward examples

适用：`scripts/training/train_grpo.py`。

常见原因：Judge 已关闭，且数据行没有与已启用内置奖励匹配的信号。Judge 开启时，能构造 prompt 的 prompt-only 行是有效的。

修复示例：

```json
{"prompt":"Answer from the documentation.","reference_answer":"Only documented facts.","required_terms":["documentation"]}
```

Judge 关闭时，确认每条数据至少包含一个与 `builtin_rewards` 匹配的字段：

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

常见原因：后续阶段需要上游 adapter，但该 adapter 尚未生成，或者目录存在却不是完整 PEFT adapter。有效目录根部必须同时包含 `adapter_config.json`，以及 `adapter_model.safetensors` 或 `adapter_model.bin`。

修复选项：

- 先运行上游阶段。
- 把 `base_adapter_dir` 指向完整 adapter。GRPO 会按“显式路径 -> DPO -> Fact-SFT -> CPT”回退，并在错误中列出每个无效目录缺少的文件。
- 合并特定 adapter 时使用 `merge.adapter_dir` 或 `merge_adapter.py --adapter_dir`。
- 只有在明确实验需要时，才把相关 `require_*_adapter` 选项设为 `false`。

## Reward judge API key 缺失

适用：`grpo.reward_judge.enabled=true` 的 GRPO。

常见原因：Git 忽略的 local YAML 中 `api_key` 为空，同时 `api_key_env` 指定的可选回退环境变量也没有设置。

修复：

```yaml
grpo:
  reward_judge:
    api_key: "your-key"
```

`api_key` 是明文凭据，只能放在不提交的 `configs/domain_post_training.local.yaml`。如果选择环境变量回退：

```bash
export GRPO_REWARD_JUDGE_API_KEY="your-key"
```

Windows PowerShell:

```powershell
$env:GRPO_REWARD_JUDGE_API_KEY = "your-key"
```

## Reward judge 响应不符合 `grpo_judge_v2`

适用：GRPO 外部 reward judge。

常见原因：Judge 返回纯文本、Markdown、旧的标量 `score` 格式、额外/缺失字段、越界维度或未知违规标识。

修复：更新 judge prompt 或服务，使 OpenAI-compatible 响应的 `choices[0].message.content` 包含类似 JSON：

```json
{"schema_version":"grpo_judge_v2","dimension_scores":{"task_fulfillment":0.8,"factual_grounding":null,"explicit_constraints":1.0,"safety_refusal":1.0,"relevance_clarity":0.8},"violations":[],"reason":"The response follows the supplied constraints."}
```

严格 JSON 必须放在 chat completions envelope 的 assistant content 中；HTTP body 直接返回该对象不够。本地代码会对五维分数加权并应用违规硬上限，不读取 Judge 提供的最终 `score`。详见 [GRPO 与 Reward Judge](GRPO-and-Reward-Judge)。

## Reward judge 的 `message.content` 为空

适用：会先输出内部推理的外部 Judge。

常见原因：较小的输出预算全部消耗在服务的 `reasoning_content`，没有剩余 token 返回最终 v2 JSON；或超时设置过短。

修复：

```yaml
grpo:
  reward_judge:
    max_tokens: 4096
    timeout_seconds: 120
```

这里的 `max_tokens` 是 Judge 响应预算。增加 `grpo.max_completion_length` 只会增加被训练策略的候选长度，不能修复 Judge 空响应。

## `grad_norm` 是 NaN 或 Inf

适用：CPT、Fact-SFT、DPO、GRPO。

不要因为 loss 有限或报告显示 completed 就忽略它；非有限梯度可能意味着 adapter 没有有效更新。保持以下默认：

```yaml
training:
  bf16: auto
  fp16: auto
  torch_dtype: auto
  abort_on_nonfinite_grad_norm: true
```

Fact-SFT、DPO、GRPO 使用相同的阶段级设置。支持 BF16 时优先 BF16；仍失败时检查学习率、量化、loss scaling、数据异常和依赖版本。重跑后对比相邻 adapter 的 LoRA tensor，确认权重实际变化。

## GRPO reward 很低、近似常数或截断率很高

检查训练日志和报告中的 Judge reward 均值/方差以及 `completions/clipped_ratio`。近似常数的 reward 无法产生有效排序；高截断表示 Judge 持续看到不完整候选。

优先检查 Judge rubric 与训练数据是否匹配，并改善 EOS/停止行为。只有在完整回答确实需要更多空间时才提高 `grpo.max_completion_length`。不要用 Judge 的 `max_tokens` 调整策略候选长度。

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
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --allow_unsafe_corpus
```

## No Valid GRPO Reward Examples

Applies to: `scripts/training/train_grpo.py`.

Likely cause: the judge is disabled and rows have no signal matching an enabled built-in reward. With the judge enabled, prompt-only rows are valid when a prompt can be constructed.

Fix:

```json
{"prompt":"Answer from the documentation.","reference_answer":"Only documented facts.","required_terms":["documentation"]}
```

With the judge disabled, verify that every row has a field matching `builtin_rewards`:

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

Likely cause: an upstream adapter has not been produced, or a directory exists but is not a complete PEFT adapter. A valid root contains `adapter_config.json` plus either `adapter_model.safetensors` or `adapter_model.bin`.

Fix options:

- Run the upstream stage first.
- Point `base_adapter_dir` to a complete adapter. GRPO falls back through explicit path -> DPO -> Fact-SFT -> CPT and reports missing files for invalid candidates.
- For a specific merge, use `merge.adapter_dir` or `merge_adapter.py --adapter_dir`.
- Set the relevant `require_*_adapter` option to `false` only for intentional experiments.

## Reward Judge API Key Is Missing

Applies to: GRPO with `grpo.reward_judge.enabled=true`.

Likely cause: `api_key` is empty in the Git-ignored local YAML and the optional environment fallback named by `api_key_env` is also unset.

Fix:

```yaml
grpo:
  reward_judge:
    api_key: "your-key"
```

`api_key` is plaintext and belongs only in the untracked `configs/domain_post_training.local.yaml`. If you choose the environment fallback:

```bash
export GRPO_REWARD_JUDGE_API_KEY="your-key"
```

Windows PowerShell:

```powershell
$env:GRPO_REWARD_JUDGE_API_KEY = "your-key"
```

## Reward Judge Response Does Not Match `grpo_judge_v2`

Applies to: GRPO external reward judge.

Likely cause: the judge returned prose, Markdown, the legacy scalar `score` format, missing or extra fields, out-of-range dimensions, or unknown violations.

Fix: update the judge prompt or service so `choices[0].message.content` in the OpenAI-compatible response contains JSON like:

```json
{"schema_version":"grpo_judge_v2","dimension_scores":{"task_fulfillment":0.8,"factual_grounding":null,"explicit_constraints":1.0,"safety_refusal":1.0,"relevance_clarity":0.8},"violations":[],"reason":"The response follows the supplied constraints."}
```

Strict JSON must appear inside assistant content in the chat completions envelope; returning it as the raw HTTP body is insufficient. Local code weights dimensions and applies violation caps; it does not read a final judge `score`. See [GRPO And Reward Judge](GRPO-and-Reward-Judge).

## Reward Judge `message.content` Is Empty

Applies to: external judges that produce internal reasoning before the final answer.

Likely cause: a small output budget was consumed by provider `reasoning_content`, leaving no tokens for final v2 JSON, or the timeout is too short.

Fix:

```yaml
grpo:
  reward_judge:
    max_tokens: 4096
    timeout_seconds: 120
```

This `max_tokens` is the judge response budget. Raising `grpo.max_completion_length` only lengthens policy candidates and cannot fix an empty judge response.

## `grad_norm` Is NaN or Inf

Applies to: CPT, Fact-SFT, DPO, GRPO.

Do not ignore this because loss is finite or the report says completed; non-finite gradients can mean the adapter did not update meaningfully. Keep these defaults:

```yaml
training:
  bf16: auto
  fp16: auto
  torch_dtype: auto
  abort_on_nonfinite_grad_norm: true
```

Fact-SFT, DPO, and GRPO use matching stage-level fields. BF16 is preferred when supported. If failure remains, inspect learning rate, quantization, loss scaling, anomalous rows, and dependency versions. After rerunning, compare LoRA tensors across adjacent adapters to confirm real changes.

## GRPO Reward Is Low, Nearly Constant, or Highly Clipped

Inspect judge reward mean/variance and `completions/clipped_ratio` in logs and reports. Near-constant rewards provide no useful ranking, while high clipping means the judge repeatedly sees incomplete candidates.

First verify rubric/data alignment and improve EOS/stop behavior. Raise `grpo.max_completion_length` only when complete answers genuinely require more room. Do not use judge `max_tokens` to tune policy candidate length.

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
