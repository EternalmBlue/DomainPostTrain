# 故障排查 / Troubleshooting

## 中文

当命令失败、训练中断或预期产物缺失时，使用本页。

## `ModuleNotFoundError: No module named ...`

适用：安装、smoke test、训练、推理。

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

## 没有有效 GRPO reward examples

适用：`scripts/training/train_grpo.py`。

常见原因：每条 GRPO 行需要 `prompt`，并至少需要一个奖励信号字段。

修复示例：

```json
{"prompt":"Answer from the documentation.","reference_answer":"Only documented facts.","required_terms":["documentation"]}
```

确认每条数据至少包含以下字段之一：

- `reference_answer`
- `required_terms`
- `forbidden_terms`
- `must_refuse`

## DPO 行被拒绝

适用：DPO 数据集准备。

常见原因：字段缺失、值为空，或 `chosen == rejected`。

修复：确保每行都有非空 `prompt`、`chosen`、`rejected`，并且两个回答不同。

## Base adapter not found

适用：Fact-SFT、DPO、GRPO。

常见原因：后续阶段需要上游 adapter，但该 adapter 尚未生成。

修复选项：

- 先运行上游阶段。
- 把 `base_adapter_dir` 指向已存在的 adapter。
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

修复：更新 judge prompt 或服务，使 assistant message 包含类似 JSON：

```json
{"score": 0.8, "reason": "The answer follows the reference and avoids forbidden terms."}
```

程序只读取 `score`。

## Flask 服务启动了，但 `/v1/chat/completions` 失败

常见原因：

- `--model_path` 没有指向合并后的模型。
- 模型依赖缺失。
- 请求中的 `messages` payload 格式错误。

验证：

```bash
curl http://localhost:8000/v1/models
```

然后使用 [推理与导出](Inference-and-Export) 里的最小 chat completions 请求重试。

## 静态检查通过但训练失败

`compileall` 只检查 Python 语法。它不会加载数据集、导入所有 ML 依赖、分配 GPU 显存或运行 TRL。

下一步检查：

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers, datasets, peft, trl; print('training deps ok')"
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

---

## English

Use this page when a command fails, training stops, or expected artifacts are missing.

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

## No Valid GRPO Reward Examples

Applies to: `scripts/training/train_grpo.py`.

Likely cause: each GRPO row needs `prompt` and at least one reward signal.

Fix:

```json
{"prompt":"Answer from the documentation.","reference_answer":"Only documented facts.","required_terms":["documentation"]}
```

Verify that each row has at least one of:

- `reference_answer`
- `required_terms`
- `forbidden_terms`
- `must_refuse`

## DPO Row Is Rejected

Applies to: DPO dataset preparation.

Likely cause: missing field, empty value, or `chosen == rejected`.

Fix: ensure every row has non-empty `prompt`, `chosen`, and `rejected`, and that the two answers differ.

## Base Adapter Not Found

Applies to: Fact-SFT, DPO, GRPO.

Likely cause: a later stage expects an upstream adapter that has not been produced.

Fix options:

- Run the upstream stage first.
- Point `base_adapter_dir` to an existing adapter.
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

Fix: update the judge prompt or service so the assistant message contains JSON like:

```json
{"score": 0.8, "reason": "The answer follows the reference and avoids forbidden terms."}
```

The program reads only `score`.

## Flask Service Starts but `/v1/chat/completions` Fails

Likely causes:

- `--model_path` does not point to a merged model.
- Model dependencies are missing.
- The request uses a malformed `messages` payload.

Verify:

```bash
curl http://localhost:8000/v1/models
```

Then retry a minimal chat completions request from [Inference And Export](Inference-and-Export).

## Static Check Passes but Training Fails

`compileall` only checks Python syntax. It does not load datasets, import all ML dependencies, allocate GPU memory, or run TRL.

Next checks:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers, datasets, peft, trl; print('training deps ok')"
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```
