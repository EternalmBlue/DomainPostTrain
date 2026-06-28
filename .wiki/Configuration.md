# Configuration

Use this page when adapting DomainPostTrain to another domain, changing stage outputs, or tuning training behavior.

The full parameter reference lives in `configs/README.md`. This page focuses on the settings users usually edit first.

## Main config file

The default config is:

```text
configs/domain_post_training.yaml
```

Recommended workflow:

```bash
cp configs/domain_post_training.yaml configs/my_domain.yaml
```

Then run commands with:

```bash
python scripts/training/train_pipeline.py --config configs/my_domain.yaml
```

## Path rules

- Training artifacts usually resolve relative to the repository root, such as `outputs/lora_adapter`.
- Data input paths are usually resolved relative to the config file first. In the default config, `../data/...` points back to repository-level `data/...`.
- `null` means the code should use its default behavior.

## Replace these first

| Setting | Why it matters |
|---|---|
| `base_model_repo_id` | Hugging Face Hub repo used by `download_models.py`. |
| `base_model_name_or_path` | Actual local path or model id loaded for training, merge, and inference. |
| `corpus.input_paths` | CPT source documents for the domain. |
| `fact_sft.input_paths` | Fact-SFT JSONL data. |
| `dpo.input_path` | DPO preference data, if DPO is enabled. |
| `grpo.input_path` | GRPO reward-prompt data, if GRPO is enabled. |
| `eval.question_file` | Post-training quality evaluation questions. |
| `fact_sft.system_prompt` | Domain role, knowledge boundary, and safety boundary. |

## Stage switches

```yaml
fact_sft:
  enabled: true

dpo:
  enabled: false

grpo:
  enabled: false
```

The full pipeline starts with CPT, then runs enabled later stages. You can also skip stages from the command line. See [Training Pipeline](Training-Pipeline).

## Memory-sensitive defaults

When GPU memory is tight, start with smaller sequence length, smaller LoRA rank, 4-bit loading, and gradient checkpointing:

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

## Validation and quality evaluation

Training validation and post-training quality evaluation are different:

- Validation set: used during training for loss/eval signal.
- Quality evaluation: run after training to check factual answers, safe refusals, and regressions.

The mock data is small, so validation is disabled by default:

```yaml
corpus:
  validation_mode: "none"
fact_sft:
  validation_ratio: 0
dpo:
  validation_ratio: 0
```

For real projects, prefer independent held-out CPT documents:

```yaml
corpus:
  validation_mode: "separate_sources"
  validation_sources:
    - "../data/cpt_validation/source_documents"
```

## Related pages

- [Data Contracts](Data-Contracts)
- [Training Pipeline](Training-Pipeline)
- [GRPO And Reward Judge](GRPO-and-Reward-Judge)
- [Troubleshooting](Troubleshooting)

