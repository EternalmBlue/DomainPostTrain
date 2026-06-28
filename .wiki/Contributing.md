# 贡献指南 / Contributing

## 中文

当你准备向 DomainPostTrain 提交代码、数据或文档变更时，使用本页。

## 贡献边界

保持仓库领域中立。不要加入：

- 专有客户数据
- 真实凭据或 token
- 私有 URL
- 内部 system prompt
- 产品专属语料
- 许可证受限训练数据

中文提示：开源前先检查 `data/`、`configs/`、`outputs/`、`models/` 和 Wiki 示例，不要泄露私有信息。

## 推荐检查

运行：

```bash
python -m compileall pipeline scripts
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

如果因为本机缺少 ML 依赖无法运行 smoke test，请说明：

- 精确失败原因
- 已运行的静态检查
- 环境限制，例如无 CUDA 或缺少包管理器

## 文档变更规则

修改行为时：

1. 只有项目高层行为或最快启动方式变化时，才更新 `README.md`。
2. 配置语义变化时，更新 `configs/README.md`。
3. 用户流程、排障或发布方式变化时，更新 `.wiki/` 页面。
4. JSON/JSONL 行契约变化时，更新 `Data-Contracts.md`。
5. 修复的问题仍可能影响旧版本或常见环境时，更新 `Troubleshooting.md`。

## GRPO 变更的测试预期

至少运行：

```bash
py -m unittest tests.test_grpo_core
py -m compileall pipeline scripts tests
```

GRPO reward judge 变更应覆盖：

- 配置校验
- judge disabled 行为
- judge enabled builder 行为
- JSON score 解析
- score clamp
- 不记录敏感密钥的元数据

## 安全报告

安全问题请遵循 `SECURITY.md`：打开 private advisory 或直接联系维护者，并提供复现步骤、预期影响和已验证的安全缓解方案。

---

## English

Use this page when preparing code, data, or documentation changes for DomainPostTrain.

## Contribution Boundary

Keep the repository domain-neutral. Do not add:

- proprietary customer data
- real credentials or tokens
- private URLs
- internal system prompts
- product-specific corpora
- license-restricted training data

Note: before open-sourcing, check `data/`, `configs/`, `outputs/`, `models/`, and Wiki examples to avoid leaking private information.

## Recommended Checks

Run:

```bash
python -m compileall pipeline scripts
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

If the smoke test cannot run because the local machine lacks ML dependencies, include:

- the exact failure reason
- the static checks you did run
- any environment constraints, such as no CUDA or missing package manager

## Documentation Changes

When changing behavior:

1. Update `README.md` only for high-level project behavior or fastest-start guidance.
2. Update `configs/README.md` when config semantics change.
3. Update `.wiki/` pages when user procedures, troubleshooting, or publishing guidance changes.
4. Update `Data-Contracts.md` when any JSON/JSONL row contract changes.
5. Update `Troubleshooting.md` when a fixed bug can still affect older versions or common setups.

## Test Expectations for GRPO Changes

At minimum, run:

```bash
py -m unittest tests.test_grpo_core
py -m compileall pipeline scripts tests
```

GRPO reward judge changes should cover:

- config validation
- disabled judge behavior
- enabled judge builder behavior
- JSON score parsing
- score clamping
- secret-free metadata

## Security Reports

For security issues, follow `SECURITY.md`: open a private advisory or contact the maintainer directly, and include reproduction steps, expected impact, and any safe mitigation already tested.
