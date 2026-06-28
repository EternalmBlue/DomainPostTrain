# Scripts

命令行入口按工作流分组：

```text
scripts/
  training/      CPT、Fact-SFT、DPO、GRPO 和完整训练入口
  model_artifacts/  模型下载、adapter merge、ONNX 导出、GGUF 导出
  inference/     推理、训练后质量评估、可选 DPO rejected 答案填充
  diagnostics/   本地环境检查
  _shared/       内部 bootstrap helper
```

建议从仓库根目录运行命令，保证 `configs/domain_post_training.yaml` 里的相对路径解析一致。

English summary:

- `training/`: CPT, Fact-SFT, DPO, GRPO, and full pipeline training entrypoints.
- `model_artifacts/`: model download, adapter merge, ONNX export, and GGUF export.
- `inference/`: inference, post-training quality evaluation, and optional DPO rejected-answer filling.
- `diagnostics/`: local environment checks.
- `_shared/`: internal script bootstrap helper.
