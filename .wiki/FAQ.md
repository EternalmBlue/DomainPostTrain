# FAQ

Use this page for conceptual questions. Use [Troubleshooting](Troubleshooting) for exact errors and symptoms.

## Is `.wiki/` automatically published as the GitHub Wiki?

No. GitHub Wiki content lives in a separate Git repository named like `OWNER/REPO.wiki.git`. The `.wiki/` directory in the main repo is a source directory. Publish it by following [Publishing](Publishing).

## Why keep Wiki pages separate from the README?

The README should stay short enough to explain what the project is and how to get started. The Wiki holds deeper operational guides, configuration notes, troubleshooting, and maintenance procedures.

## What is the difference between validation and quality evaluation?

Validation runs during training and produces loss/eval signals. Quality evaluation runs after training to inspect factual answers, safe refusals, and base-capability regressions.

## When should I enable DPO?

Enable DPO when you have preference data with `prompt`, `chosen`, and `rejected`, and you want the model to prefer one answer style or behavior over another.

## When should I enable GRPO?

Enable GRPO when you have reward prompts and computable reward signals, and you want on-policy reward optimization after Fact-SFT or DPO.

## Why must local reward models expose an OpenAI-compatible API?

The GRPO reward judge is intentionally standardized around `base_url`, `api_key_env`, and `model`. Local models, DeepSeek, GLM, and other hosted judges are all called through the same OpenAI-compatible chat completions shape. This avoids separate code paths for local model files, Hub IDs, and custom Python reward hooks.

## Does the external reward judge replace built-in rewards?

No. Built-in rewards can still run. If `grpo.reward_judge.enabled=true`, the external judge is appended to the reward functions.

## Can I put private domain documents in `data/`?

Do not publish private documents, credentials, customer tickets, internal prompts, source code, or license-restricted corpora in a derivative public repository. Use private storage and confirm licensing before sharing.

## Is ONNX required?

No. ONNX export is optional. The default local-deployment export path is GGUF.

## Why did `pyyaml` fail in a local check?

If the active Python environment lacks `pyyaml`, YAML parsing probes will fail with `ModuleNotFoundError: No module named 'yaml'`. Install `requirements.txt` in the active environment before running config-loading checks.

