# 常见问题 / FAQ

## 中文

本页回答概念性问题。具体错误和症状请看 [故障排查](Troubleshooting)。

## `.wiki/` 会自动发布成 GitHub Wiki 吗？

不会。GitHub Wiki 内容位于独立 Git 仓库，名称形如 `OWNER/REPO.wiki.git`。主仓库中的 `.wiki/` 是源目录。发布步骤见 [发布 Wiki](Publishing)。

## 为什么 Wiki 要和 README 分开？

README 应保持足够短，用来说明项目是什么、为什么有用以及最快如何开始。Wiki 承载更深的操作指南、配置说明、排障、FAQ 和维护流程。

## validation 和 quality evaluation 有什么区别？

validation 在训练过程中运行，提供 loss/eval 信号。quality evaluation 在训练后运行，用来检查事实回答、安全拒答和基础能力回归。

## 什么时候启用 DPO？

当你有包含 `prompt`、`chosen`、`rejected` 的偏好数据，并希望模型更偏好某种回答风格或行为时，启用 DPO。

## 什么时候启用 GRPO？

当你有奖励 prompt 和可计算奖励信号，并希望在 Fact-SFT 或 DPO 之后做 on-policy reward optimization 时，启用 GRPO。

## 为什么本地 reward model 也要暴露 OpenAI-compatible API？

GRPO reward judge 使用 `base_url`、`api_key_env`、`model` 作为核心配置。本地模型、DeepSeek、GLM 和其他托管 judge 都通过同一种 OpenAI-compatible chat completions 形状调用。这样训练链路只负责标准 HTTP 调用，模型部署、并发、限流和显存管理由外部 judge 服务处理。

## 外部 reward judge 会替代内置奖励吗？

不会。内置奖励仍可运行。如果 `grpo.reward_judge.enabled=true`，外部 judge 会追加到 reward functions 中。

## 可以把私有领域文档放进 `data/` 吗？

不要在公开衍生仓库中发布私有文档、凭据、客户工单、内部 prompt、源代码或许可证受限语料。请使用私有存储，并在分享前确认授权。

## ONNX 是必需的吗？

不是。ONNX 导出是可选能力。默认本地部署导出路径是 GGUF。

## 为什么本地检查会报 `pyyaml` 缺失？

如果当前 Python 环境没有 `pyyaml`，YAML 解析探针会失败并报 `ModuleNotFoundError: No module named 'yaml'`。运行配置加载检查前，请在当前环境安装 `requirements.txt`。

---

## English

Use this page for conceptual questions. Use [Troubleshooting](Troubleshooting) for exact errors and symptoms.

## Is `.wiki/` Automatically Published as the GitHub Wiki?

No. GitHub Wiki content lives in a separate Git repository named like `OWNER/REPO.wiki.git`. The `.wiki/` directory in the main repo is a source directory. Publish it by following [Publishing](Publishing).

## Why Keep Wiki Pages Separate from the README?

The README should stay short enough to explain what the project is and how to get started. The Wiki holds deeper operational guides, configuration notes, troubleshooting, and maintenance procedures.

## What Is the Difference Between Validation and Quality Evaluation?

Validation runs during training and produces loss/eval signals. Quality evaluation runs after training to inspect factual answers, safe refusals, and base-capability regressions.

## When Should I Enable DPO?

Enable DPO when you have preference data with `prompt`, `chosen`, and `rejected`, and you want the model to prefer one answer style or behavior over another.

## When Should I Enable GRPO?

Enable GRPO when you have reward prompts and computable reward signals, and you want on-policy reward optimization after Fact-SFT or DPO.

## Why Must Local Reward Models Expose an OpenAI-Compatible API?

The GRPO reward judge uses `base_url`, `api_key_env`, and `model` as its core configuration. Local models, DeepSeek, GLM, and other hosted judges are all called through the same OpenAI-compatible chat completions shape. The training pipeline only handles the standard HTTP call; model deployment, concurrency, rate limits, and GPU memory are owned by the external judge service.

## Does the External Reward Judge Replace Built-in Rewards?

No. Built-in rewards can still run. If `grpo.reward_judge.enabled=true`, the external judge is appended to the reward functions.

## Can I Put Private Domain Documents in `data/`?

Do not publish private documents, credentials, customer tickets, internal prompts, source code, or license-restricted corpora in a derivative public repository. Use private storage and confirm licensing before sharing.

## Is ONNX Required?

No. ONNX export is optional. The default local-deployment export path is GGUF.

## Why Did `pyyaml` Fail in a Local Check?

If the active Python environment lacks `pyyaml`, YAML parsing probes will fail with `ModuleNotFoundError: No module named 'yaml'`. Install `requirements.txt` in the active environment before running config-loading checks.
