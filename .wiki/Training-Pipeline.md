# Training Pipeline

Use this page when running, skipping, or debugging training stages.

## Stage order

```text
CPT -> Fact-SFT -> optional DPO -> optional GRPO -> merge -> quality eval
```

The pipeline produces PEFT adapters first, then merges the selected adapter into a full Hugging Face model.

## Full pipeline

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml
```

Important outputs:

| Output | When it appears |
|---|---|
| `outputs/lora_adapter/` | CPT adapter. |
| `outputs/fact_sft_adapter/` | Fact-SFT adapter. |
| `outputs/dpo_adapter/` | DPO adapter when `dpo.enabled=true`. |
| `outputs/grpo_adapter/` | GRPO adapter when `grpo.enabled=true`. |
| `outputs/merged_model/` | Merged model. |
| `outputs/cpt_dataset/coverage_report.md` | CPT coverage report. |
| `outputs/eval/eval_report.md` | Post-training quality evaluation report. |
| `outputs/reports/pipeline_report.md` | Pipeline summary report. |

## Smoke test

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

Use this before real GPU training. It validates wiring without downloading the default base model.

## Skip stages

Stage skipping is explicit:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo --skip_grpo
```

Use skip flags only when the required upstream adapter already exists or the stage is intentionally disabled.

## Run DPO only by configuration

```yaml
dpo:
  enabled: true
  input_path: "data/dpo/preference_examples.jsonl"
```

DPO input rows require `prompt`, `chosen`, and `rejected`, with `chosen != rejected`.

## Run GRPO directly

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.yaml --max_steps 10
```

GRPO runs after DPO or Fact-SFT. It samples multiple completions per prompt and applies built-in rewards plus the optional external reward judge. See [GRPO And Reward Judge](GRPO-and-Reward-Judge).

## Merge behavior

When `merge.adapter_dir` is `null`, adapter merge auto-selects the newest available stage in priority order:

```text
GRPO -> DPO -> Fact-SFT -> CPT
```

Set `merge.adapter_dir` only when you want to merge a specific adapter.

## Static verification

If training dependencies are unavailable:

```bash
python -m compileall pipeline scripts serve_inference.py
```

This verifies syntax only. It does not validate CUDA, dataset loading, model loading, or TRL behavior.

