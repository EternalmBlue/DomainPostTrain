# DomainPostTrain Wiki

## 中文

DomainPostTrain 是一个面向领域大模型后训练的可复现流水线。它把领域文档、事实 SFT 样本、偏好数据和 GRPO 奖励提示组织成 LoRA/QLoRA 训练链路，并产出 adapter、合并模型、质量评估报告、本地推理服务和导出产物。

本 Wiki 默认使用中文说明。每个页面下半部分提供英文版，方便后续同步给英文读者或国际协作者。

注意：主仓库里的 `.wiki/` 只是 GitHub Wiki 的源目录，不会自动上线。真正上线需要把这些 Markdown 文件复制并推送到独立的 `OWNER/REPO.wiki.git` 仓库。操作见 [发布 Wiki](Publishing)。

## 从这里开始

- 第一次使用： [快速开始](Quick-Start)
- 安装依赖： [安装](Installation)
- 修改训练配置： [配置](Configuration)
- 替换样例数据： [数据契约](Data-Contracts)
- 跑完整训练链路： [训练流水线](Training-Pipeline)
- 使用 GRPO 与奖励 judge： [GRPO 与 Reward Judge](GRPO-and-Reward-Judge)
- 推理、合并和导出模型： [推理与导出](Inference-and-Export)
- 排查失败： [故障排查](Troubleshooting)
- 贡献代码或文档： [贡献指南](Contributing)

## 常见任务入口

| 我想要... | 阅读 |
|---|---|
| 用最短路径验证本地链路 | [快速开始](Quick-Start) |
| 安装 CUDA、PyTorch 或 ONNX 依赖 | [安装](Installation) |
| 把 mock 数据替换成自己的领域数据 | [数据契约](Data-Contracts) |
| 调整 batch size、LoRA rank、验证集或输出目录 | [配置](Configuration) |
| 启用 DPO 或 GRPO | [训练流水线](Training-Pipeline) |
| 接入本地、DeepSeek 或 GLM 奖励 judge | [GRPO 与 Reward Judge](GRPO-and-Reward-Judge) |
| 导出 GGUF 或 ONNX | [推理与导出](Inference-and-Export) |
| 把 `.wiki` 发布到 GitHub Wiki | [发布 Wiki](Publishing) |

## 当前训练链路

```text
CPT -> Fact-SFT -> optional DPO -> optional GRPO -> merge -> quality eval -> inference/export
```

仓库自带的 `AsterHelp` 只是虚构领域的静态 mock 数据。真正训练或发布衍生项目之前，请替换 CPT 文档、SFT 样本、DPO 偏好样本、GRPO 奖励样本、质量评估问题和 system prompt，并确认数据来源、授权和安全边界。

## 主仓库文档边界

Wiki 是任务型说明。主仓库仍保留更紧凑的项目级文档：

- `README.md`：项目概览和核心流程。
- `configs/README.md`：完整双语配置参考。
- `data/README.md`：打包 mock 数据说明和发布提醒。
- `scripts/README.md`：脚本分组。
- `CONTRIBUTING.md`：贡献检查。
- `SECURITY.md`：安全报告和数据安全边界。

## 当前状态

- 默认依赖面向 CUDA 12.6 GPU 训练。
- ONNX 导出是可选能力，依赖在 `requirements-onnx.txt`。
- GRPO 模型评分已标准化为 `grpo.reward_judge`，通过 OpenAI-compatible chat completions API 调用外部 judge。
- 本 Wiki 的源文件维护在 `.wiki/`，需要同步到独立的 GitHub Wiki 仓库后才会对读者生效。

---

## English

DomainPostTrain is a reproducible LLM post-training pipeline for turning domain documents, factual SFT examples, preference data, and GRPO reward prompts into LoRA/QLoRA adapters, a merged model, quality evaluation reports, local inference services, and export artifacts.

This Wiki is Chinese-first by default. Each page also includes an English section for international collaborators.

Note: `.wiki/` in the main repository is only the source directory. It does not automatically publish to GitHub Wiki. To go live, copy these Markdown files to the separate `OWNER/REPO.wiki.git` repository. See [Publishing](Publishing).

## Start Here

- First-time user: [Quick Start](Quick-Start)
- Installing dependencies: [Installation](Installation)
- Changing training behavior: [Configuration](Configuration)
- Replacing sample data: [Data Contracts](Data-Contracts)
- Running the full pipeline: [Training Pipeline](Training-Pipeline)
- Using GRPO and reward judging: [GRPO And Reward Judge](GRPO-and-Reward-Judge)
- Serving or exporting a model: [Inference And Export](Inference-and-Export)
- Something failed: [Troubleshooting](Troubleshooting)
- Contributing changes: [Contributing](Contributing)

## Common Tasks

| I want to... | Read this |
|---|---|
| Run the shortest local verification | [Quick Start](Quick-Start) |
| Install CUDA, PyTorch, or ONNX dependencies | [Installation](Installation) |
| Replace the mock domain with my own data | [Data Contracts](Data-Contracts) |
| Tune batch size, LoRA rank, validation, or output paths | [Configuration](Configuration) |
| Enable DPO or GRPO | [Training Pipeline](Training-Pipeline) |
| Use a local, DeepSeek, or GLM reward judge | [GRPO And Reward Judge](GRPO-and-Reward-Judge) |
| Export GGUF or ONNX | [Inference And Export](Inference-and-Export) |
| Publish `.wiki` to GitHub Wiki | [Publishing](Publishing) |

## Pipeline Overview

```text
CPT -> Fact-SFT -> optional DPO -> optional GRPO -> merge -> quality eval -> inference/export
```

The repository ships only static mock data for the fictional `AsterHelp` domain. Before training or publishing a derivative project, replace the corpus, SFT rows, preference rows, reward prompts, quality evaluation questions, and system prompts with data you are licensed to use.

## Primary Repository Docs

The Wiki is task-oriented. The main repository keeps compact project-level references:

- `README.md`: project overview and core workflow.
- `configs/README.md`: full bilingual configuration reference.
- `data/README.md`: packaged mock data summary and publication warning.
- `scripts/README.md`: script grouping.
- `CONTRIBUTING.md`: contribution checks.
- `SECURITY.md`: security reporting and data safety boundary.

## Project Status

- Default dependencies target CUDA 12.6 GPU training.
- ONNX export is optional and uses `requirements-onnx.txt`.
- GRPO reward-model scoring is standardized through `grpo.reward_judge`, which calls an OpenAI-compatible chat completions API.
- GitHub Wiki content is maintained in `.wiki/` and must be copied to the separate `OWNER/REPO.wiki.git` repository to go live.
