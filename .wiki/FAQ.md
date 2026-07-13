# 常见问题 / FAQ

## 中文

本页回答概念性问题。具体错误和症状请看 [故障排查](Troubleshooting)。

## `.wiki/` 会自动发布成 GitHub Wiki 吗？

不会。GitHub Wiki 内容位于独立 Git 仓库，名称形如 `OWNER/REPO.wiki.git`。主仓库中的 `.wiki/` 是源目录。发布步骤见 [发布 Wiki](Publishing)。

## validation 和 quality evaluation 有什么区别？

validation 在训练过程中提供 loss/eval 信号。quality evaluation 在训练后检查事实回答、安全拒答和基础能力回归。当前内置评估是启发式 smoke gate，不是生产安全认证；发布前仍需独立 Judge 或人工复核安全样例。

## `base_model_repo_id` 和 `base_model_name_or_path` 重复吗？

不重复。`base_model_repo_id` 是 `download_models.py` 的下载来源；`base_model_name_or_path` 是训练、合并和推理实际加载的位置。默认配置会从前者下载到 `models/base-model`，再通过后者读取本地快照。

## 为什么 DPO 和 GRPO 默认开启？

默认模板表达完整后训练链路。真实训练前必须准备 DPO/GRPO 数据并配置 Judge；不需要某阶段时可以在配置中关闭或使用流水线 `--skip_*` 参数。

## GRPO 只提供 `prompt` 可以吗？

可以，但仅限外部 Judge 开启时。Judge 关闭后，每行必须至少包含一个与已启用内置奖励匹配的信号。

## 外部 Judge 会替代内置奖励吗？

Judge 是默认奖励器，四个内置奖励默认全部关闭。把奖励名加入 `builtin_rewards` 后，它们会与 Judge 一起运行；如果关闭 Judge，则至少要启用一个内置奖励。

## 为什么本地 Judge 也要暴露 OpenAI-compatible API？

训练链路统一使用 `base_url`、`model` 和 Key 调用 `/v1/chat/completions`。本地模型、DeepSeek、GLM 和其他服务使用同一接口，部署、并发、限流和显存管理由 Judge 服务负责。

## Judge Key 应该放在哪里？

默认直接写在 Git 忽略的 `configs/domain_post_training.local.yaml` 的 `reward_judge.api_key` 中。它是明文凭据，不能提交或分享。`api_key_env` 只是 `api_key` 为空时的可选回退。

## `max_completion_length` 和 Judge 的 `max_tokens` 有什么区别？

`grpo.max_completion_length` 限制被训练策略生成的候选回答；`grpo.reward_judge.max_tokens` 限制 Judge 单次评分响应。候选被截断看前者和 `completions/clipped_ratio`；Judge 的 `message.content` 为空则看后者，推理型 Judge 推荐从 `4096` 开始。

## 如何确认 GRPO 真的更新了模型？

不要只看有限 loss 或 `Status: completed`。检查所有阶段的 `grad_norm` 是否有限，对比 GRPO 前后 LoRA tensor 是否变化，并检查 reward 方差及 `completions/clipped_ratio`。

## 可以把私有领域文档放进 `data/` 吗？

不要在公开衍生仓库中发布私有文档、凭据、客户工单、内部 prompt、源代码或许可证受限语料。远端 Judge 还会收到完整 prompt、候选回答和评判上下文；敏感数据不能离开环境时应使用本地 Judge。

## ONNX 是必需的吗？

不是。ONNX 导出是可选能力。默认本地部署导出路径是 GGUF。

## 为什么本地检查会报 `pyyaml` 缺失？

当前 Python 环境没有 `pyyaml` 时，YAML 解析会报 `ModuleNotFoundError: No module named 'yaml'`。请在实际执行命令的环境中安装依赖。

---

## English

Use this page for conceptual questions. Use [Troubleshooting](Troubleshooting) for exact errors and symptoms.

## Is `.wiki/` Automatically Published as the GitHub Wiki?

No. GitHub Wiki content lives in a separate repository named like `OWNER/REPO.wiki.git`. The main repository's `.wiki/` directory is the source. See [Publishing](Publishing).

## What Is the Difference Between Validation and Quality Evaluation?

Validation provides loss/eval signals during training. Quality evaluation checks factual answers, safe refusals, and regressions after training. The built-in evaluation is a heuristic smoke gate, not production safety certification; independently judge or manually review safety cases before release.

## Are `base_model_repo_id` and `base_model_name_or_path` Duplicates?

No. `base_model_repo_id` is the source used by `download_models.py`; `base_model_name_or_path` is what training, merge, and inference actually load. The default config downloads the former into `models/base-model` and loads that local snapshot through the latter.

## Why Are DPO and GRPO Enabled by Default?

The default template represents the complete post-training chain. Real training requires DPO/GRPO data and a configured judge. Disable an unneeded stage in YAML or skip it with the pipeline's `--skip_*` flags.

## Can a GRPO Row Contain Only `prompt`?

Yes, when the external judge is enabled. With the judge disabled, every row needs at least one signal matching an enabled built-in reward.

## Does the External Judge Replace Built-in Rewards?

The judge is the default reward provider, and all four built-in rewards are disabled by default. Rewards listed in `builtin_rewards` run alongside the judge. If the judge is disabled, at least one built-in reward is required.

## Why Must a Local Judge Expose an OpenAI-Compatible API?

The training path uniformly calls `/v1/chat/completions` using `base_url`, `model`, and a key. Local models, DeepSeek, GLM, and other services use the same contract, while deployment, concurrency, rate limiting, and memory management remain the judge service's responsibility.

## Where Should the Judge Key Live?

Put it directly in `reward_judge.api_key` inside the Git-ignored `configs/domain_post_training.local.yaml`. It is a plaintext secret and must not be committed or shared. `api_key_env` is only an optional fallback when `api_key` is empty.

## How Do `max_completion_length` and Judge `max_tokens` Differ?

`grpo.max_completion_length` limits candidate responses from the policy being trained. `grpo.reward_judge.max_tokens` limits each judge response. Candidate truncation points to the former and `completions/clipped_ratio`; empty judge `message.content` points to the latter. Start reasoning judges at `4096`.

## How Do I Know GRPO Really Updated the Model?

Do not rely only on finite loss or `Status: completed`. Confirm finite `grad_norm` across stages, compare LoRA tensors before and after GRPO, and inspect reward variance and `completions/clipped_ratio`.

## Can I Put Private Domain Documents in `data/`?

Do not publish private documents, credentials, customer tickets, internal prompts, source code, or license-restricted corpora. A remote judge also receives the full prompt, candidate response, and evaluation context; use a local judge when sensitive data cannot leave the environment.

## Is ONNX Required?

No. ONNX export is optional. The default local-deployment export path is GGUF.

## Why Did `pyyaml` Fail in a Local Check?

Without `pyyaml` in the active Python environment, YAML parsing fails with `ModuleNotFoundError: No module named 'yaml'`. Install dependencies in the same environment that runs the command.
