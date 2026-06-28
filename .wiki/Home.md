# DomainPostTrain Wiki

DomainPostTrain is a reproducible LLM post-training pipeline for turning domain documents, factual SFT examples, preference data, and reward prompts into LoRA/QLoRA adapters, a merged model, quality evaluation reports, and local inference artifacts.

中文提示: 这个 `.wiki` 目录是 GitHub Wiki 的源文件目录。把它放在主仓库里不会自动发布 Wiki, 发布步骤见 [Publishing](Publishing).

## Start here

- New user: [Quick Start](Quick-Start)
- Installing dependencies: [Installation](Installation)
- Changing training behavior: [Configuration](Configuration)
- Replacing sample data: [Data Contracts](Data-Contracts)
- Running the full pipeline: [Training Pipeline](Training-Pipeline)
- Using GRPO reward optimization: [GRPO And Reward Judge](GRPO-and-Reward-Judge)
- Serving or exporting a model: [Inference And Export](Inference-and-Export)
- Something failed: [Troubleshooting](Troubleshooting)
- Contributing changes: [Contributing](Contributing)

## Common tasks

| I want to... | Read this |
|---|---|
| Run the shortest successful local check | [Quick Start](Quick-Start) |
| Install CUDA or ONNX dependencies | [Installation](Installation) |
| Replace the mock domain with my own data | [Data Contracts](Data-Contracts) |
| Tune batch size, LoRA rank, validation, or stage outputs | [Configuration](Configuration) |
| Enable DPO or GRPO | [Training Pipeline](Training-Pipeline) |
| Use a local, DeepSeek, or GLM reward judge | [GRPO And Reward Judge](GRPO-and-Reward-Judge) |
| Export GGUF or ONNX | [Inference And Export](Inference-and-Export) |
| Publish these files to GitHub Wiki | [Publishing](Publishing) |

## Pipeline overview

```text
CPT -> Fact-SFT -> optional DPO -> optional GRPO -> merge -> quality eval -> inference/export
```

The repository ships only static mock data for the fictional `AsterHelp` domain. Before training or publishing a derivative project, replace the sample corpus, SFT rows, preference rows, reward prompts, and quality evaluation questions with data you are licensed to use.

## Primary repository docs

The Wiki is task-oriented. The main repository keeps the source of truth for compact project overview and full references:

- `README.md`: project overview and core workflow.
- `configs/README.md`: full bilingual configuration reference.
- `data/README.md`: packaged mock data summary and publication warning.
- `scripts/README.md`: script grouping.
- `CONTRIBUTING.md`: contribution checks.
- `SECURITY.md`: security reporting and data safety boundary.

## Project status

- Default dependencies target CUDA 12.6 GPU training.
- ONNX export is optional and uses `requirements-onnx.txt`.
- GRPO reward-model scoring is standardized through `grpo.reward_judge`, which calls an OpenAI-compatible chat completions API.
- GitHub Wiki content is maintained in `.wiki/` and must be copied to the separate `OWNER/REPO.wiki.git` repository to go live.

