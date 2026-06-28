# Troubleshooting

Use this page when a command fails or output artifacts are missing.

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

## CUDA out of memory

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

## No valid GRPO reward examples

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

## DPO row is rejected

Applies to: DPO dataset preparation.

Likely cause: missing field, empty value, or `chosen == rejected`.

Fix: ensure every row has non-empty `prompt`, `chosen`, and `rejected`, and that the two answers differ.

## Base adapter not found

Applies to: Fact-SFT, DPO, GRPO.

Likely cause: a later stage expects an upstream adapter that has not been produced.

Fix options:

- Run the upstream stage first.
- Point `base_adapter_dir` to an existing adapter.
- Set the relevant `require_*_adapter` option to `false` only for intentional experiments.

## Reward judge API key is missing

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

## Reward judge response must contain numeric score

Applies to: GRPO external reward judge.

Likely cause: the judge returned prose, markdown without a JSON object, a string score, or omitted `score`.

Fix: update the judge prompt or service so the assistant message contains JSON like:

```json
{"score": 0.8, "reason": "The answer follows the reference and avoids forbidden terms."}
```

The program reads only `score`.

## Flask service starts but `/v1/chat/completions` fails

Likely causes:

- `--model_path` does not point to a merged model.
- Model dependencies are missing.
- The request uses a malformed `messages` payload.

Verify:

```bash
curl http://localhost:8000/v1/models
```

Then retry a minimal chat completions request from [Inference And Export](Inference-and-Export).

## Static check passes but training fails

`compileall` only checks Python syntax. It does not load datasets, import all ML dependencies, allocate GPU memory, or run TRL.

Next checks:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers, datasets, peft, trl; print('training deps ok')"
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

