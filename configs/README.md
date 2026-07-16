# Config Reference / 配置说明

<p align="center">
  <a href="#chinese"><strong>中文</strong></a>
  &nbsp;|&nbsp;
  <a href="#english"><strong>English</strong></a>
</p>

<a id="chinese"></a>

## 中文

本文解释 `domain_post_training.yaml` 中用户通常会改到的参数。真实训练请先复制到 Git 忽略的 `configs/domain_post_training.local.yaml`，再通过 `--config configs/domain_post_training.local.yaml` 运行；直接填写的 Judge Key 只应出现在这份私有明文配置中，受跟踪模板必须保持 `api_key: null`。

路径规则：

- 训练产物路径通常相对项目根目录解析，例如 `outputs/lora_adapter`。
- 数据输入路径通常优先相对配置文件所在目录解析；示例中的 `../data/...` 表示项目根目录下的 `data/...`。
- `null` 表示交给代码默认逻辑处理。

## 顶层参数

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `base_model_repo_id` | Hugging Face Hub 上的下载来源。`download_models.py` 会把该仓库快照下载到本地。默认是 `Qwen/Qwen3.5-0.8B`。 | 换下载来源时修改。 |
| `base_model_name_or_path` | 训练、推理、评估和 merge 实际加载的模型位置。可以是本地目录，也可以是 Hub ID。 | 下载到本地后通常保持 `models/base-model`；本地已有模型时直接指向其目录。 |
| `trust_remote_code` | 是否允许 Transformers 加载模型仓库中的自定义代码。 | Qwen 等需要自定义建模代码的模型通常设为 `true`；只用标准架构且想更保守时设为 `false`。 |

## `corpus`

这一段控制 CPT 语料发现、切分、覆盖和训练样本构造。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `strategy` | CPT 数据构造策略。当前示例使用 `guaranteed_coverage_then_optional_replay`，表示先保证必覆盖语料进入训练，再按配置做可选 replay。 | 通常不改，除非你扩展了新的语料采样策略。 |
| `sample_unit` | CPT 样本粒度。支持 `document`、`section`、`token_chunk`。 | 文档较短且希望完整覆盖时用 `document`；长文档较多时用 `section` 或 `token_chunk`。 |
| `input_paths` | CPT 源文档路径列表，支持文件或目录。 | 替换领域文档时改这里。 |
| `exclude_paths` | 语料发现时排除的目录或路径片段。 | 项目中有源码、构建产物、模型产物、日志等不应训练的内容时添加。 |
| `prepared_dataset_dir` | CPT 预处理后 Hugging Face dataset 的输出目录。 | 想保留多套数据集或避免覆盖旧产物时修改。 |
| `min_text_chars` | 单个源文本的最小字符数，低于该值会被跳过。 | 文档片段很短但仍有价值时调低；想过滤噪声时调高。 |
| `append_safety_preamble` | 是否给 CPT 文本追加通用安全前言。 | 通常保持 `true`；如果你的语料已经有完整安全边界，可设为 `false`。 |
| `split_markdown_sections` | 是否按 Markdown 标题切分章节。`sample_unit: document` 时一般不需要。 | 长 Markdown 文档较多时开启。 |
| `max_sample_tokens` | 单个 CPT 样本允许的最大 token 数。 | 遇到显存压力或超长样本报错时调小；长上下文训练时可调大。 |
| `sample_overflow_strategy` | 超过 `max_sample_tokens` 时的处理策略。示例为 `error`，即直接报错。 | 想强制发现超长文档时用 `error`；若代码支持截断策略，可按需切换。 |
| `enable_weighted_replay` | 是否启用按类别重复采样，让重点类别出现更多次。 | 小语料想强化安全、配置、排障等重点知识时开启。 |
| `total_train_tokens` | 目标训练 token 总量。`null` 表示由必覆盖和 replay 配置自然决定。 | 想控制 CPT 训练规模时设置。 |
| `min_chunk_tokens` | 切分样本的最小 token 数。 | `token_chunk` 或章节切分后碎片太多时调高。 |
| `validation_mode` | CPT 验证集模式。`none` 表示不启用；`copy_from_train` 表示从训练样本复制一小部分做 smoke eval；`separate_sources` 表示使用独立验证文档。 | 真实项目建议用 `separate_sources`；示例 mock 数据小，所以默认 `none`。 |
| `validation_copy_ratio` | `copy_from_train` 模式下复制多少比例样本作为验证集。 | 只做训练链路 smoke check 时可设为 `0.03` 到 `0.1`。 |
| `validation_sources` | `separate_sources` 模式下的独立验证文档路径。 | 有 held-out CPT 文档时填写。 |

### `corpus.stratified_sampling`

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `stratified_sampling` | CPT 分层重复采样配置块。 | 想让某些语料类别在训练中出现更多次时查看这一段。 |
| `enabled` | 是否启用按类别重复采样。 | 小型领域语料希望重点能力更稳定时开启。 |
| `mode` | 采样模式。当前示例为 `sample_repeats`，按类别重复次数控制。 | 通常不改。 |
| `sample_repeats` | 每个语料类别的重复次数。键名来自语料自动分类结果。 | 某类知识更重要时提高对应数值；想降低过拟合时整体调低。 |
| `sample_repeats.model_spec` (`model_spec`) | 模型规格类文档重复次数。 | 有模型角色、边界、输入输出规范时调节。 |
| `sample_repeats.main_textbook` (`main_textbook`) | 主说明文档重复次数。 | 主文档质量高且代表领域全貌时调高。 |
| `sample_repeats.commands` (`commands`) | 命令或操作类文档重复次数。 | 领域有大量命令手册时调节。 |
| `sample_repeats.permissions` (`permissions`) | 权限、访问控制类文档重复次数。 | 权限边界重要时调高。 |
| `sample_repeats.configuration` (`configuration`) | 配置类文档重复次数。 | 希望模型更稳地回答配置项时调高。 |
| `sample_repeats.runtime_behavior` (`runtime_behavior`) | 运行时行为类文档重复次数。 | 需要模型理解系统行为、状态流转时调高。 |
| `sample_repeats.database_reload` (`database_reload`) | 数据库或数据刷新类文档重复次数。 | 领域包含数据刷新/重载流程时使用。 |
| `sample_repeats.troubleshooting` (`troubleshooting`) | 排障类文档重复次数。 | 支持助手通常建议调高。 |
| `sample_repeats.safety` (`safety`) | 安全边界类文档重复次数。 | 开源示例和真实助手都建议保持较高。 |
| `sample_repeats.unknowns` (`unknowns`) | 未知边界、非保证能力类文档重复次数。 | 想减少模型编造能力时调高。 |
| `sample_repeats.public_api` (`public_api`) | 公开 API 类文档重复次数。 | 领域包含 API 文档时调节。 |

## `safety`

这一段控制语料预检，不是模型运行时安全系统。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `block_on_high_risk` | 发现高风险语料时是否阻断训练。 | 开源或团队协作场景建议保持 `true`。 |
| `max_consecutive_code_lines` | 单段允许的连续代码行阈值，超过可能被标记为源码风险。 | 你的合法语料本身包含大量代码片段时调高。 |
| `warn_on_source_paths` | 是否对疑似源码路径发出警告。 | 通常保持 `true`。 |
| `warn_on_secret_patterns` | 是否对疑似密钥、token、凭据模式发出警告。 | 通常保持 `true`。 |

## `training`

这一段主要控制 CPT 阶段和通用训练默认值。Fact-SFT、DPO 和 GRPO 有自己的覆盖项。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `device` | 训练设备。常见值：`cuda`、`cuda:0`、`cpu`、`auto`。 | GPU 训练用 `cuda`；CPU smoke test 用命令行 `--device cpu` 覆盖。 |
| `output_dir` | CPT LoRA adapter 输出目录。 | 保留多次实验结果时修改。 |
| `merged_output_dir` | adapter merge 后的完整模型输出目录。 | 想把合并模型输出到指定位置时修改。 |
| `max_seq_length` | CPT 最大序列长度。 | 显存不足时调小；长文档能力更重要且显存足够时调大。 |
| `train_val_split` | 旧式训练/验证切分比例。当前 CPT 覆盖模式下不建议使用。 | 优先使用 `corpus.validation_mode`、SFT/DPO 的 `validation_ratio`。 |
| `seed` | 随机种子。 | 需要复现实验时固定；做多次实验可更换。 |
| `epochs` | CPT 训练轮数。 | 小语料通常 1 到 3；过拟合时调低。 |
| `max_steps` | 最大训练步数。`null` 表示按 epoch 训练。 | 想快速 smoke 或限制成本时设置。 |
| `per_device_train_batch_size` | 单卡训练 batch size。 | 显存足够可调大；OOM 时调小。 |
| `per_device_eval_batch_size` | 单卡验证 batch size。 | 开启验证集时生效；OOM 时调小。 |
| `gradient_accumulation_steps` | 梯度累积步数，用小 batch 模拟更大 batch。 | 显存小但想提高有效 batch 时调大。 |
| `learning_rate` | CPT 学习率。 | 训练不稳定或遗忘明显时调低；收敛太慢时小幅调高。 |
| `weight_decay` | 权重衰减。 | LoRA 微调通常保持 `0.0` 或很小。 |
| `warmup_ratio` | 学习率 warmup 比例。 | 小数据训练建议保留一定 warmup，减少早期不稳定。 |
| `lr_scheduler_type` | 学习率调度器，例如 `cosine`。 | 有明确实验需求时修改。 |
| `optim` | 优化器，例如 `paged_adamw_8bit`。 | 使用 bitsandbytes/QLoRA 时可保持；CPU 或非量化环境可能需要换成普通 AdamW。 |
| `max_grad_norm` | 梯度裁剪阈值。 | 出现梯度爆炸或 loss 不稳定时调低。 |
| `abort_on_nonfinite_grad_norm` | 出现非有限梯度范数时是否中止。 | 保持 `true`；否则 AMP 可能跳过更新但仍保存一个看似完成的 adapter。 |
| `logging_nan_inf_filter` | Trainer 日志是否过滤 NaN/Inf。 | 调试数值问题时可设为 `false` 以保留异常信号。 |
| `logging_steps` | 每多少步记录日志。 | 想看更密集训练曲线时调小。 |
| `eval_steps` | 开启验证集时每多少步评估一次。 | 有验证集时按训练规模调整。 |
| `save_steps` | 每多少步保存 checkpoint。 | 长训练可调小；短训练可调大。 |
| `save_total_limit` | 最多保留多少个 checkpoint。 | 磁盘紧张时调小。 |
| `gradient_checkpointing` | 是否启用梯度检查点以降低显存占用。 | 显存不足时保持 `true`；追求速度且显存足够可关。 |
| `bf16` | 是否使用 bfloat16；支持 `true`、`false`、`auto`。 | 保持 `auto`，在支持 BF16 的 CUDA 设备上优先启用。 |
| `fp16` | 是否使用 float16；支持 `true`、`false`、`auto`。 | 保持 `auto`，CUDA 不支持/未启用 BF16 时回退到 FP16；CPU 自动关闭。 |
| `load_in_4bit` | 是否 4-bit 量化加载基座模型。 | QLoRA 低显存训练建议开启。 |
| `load_in_8bit` | 是否 8-bit 量化加载基座模型。 | 作为 4-bit 的替代方案使用；不要同时和 4-bit 都开。 |
| `torch_dtype` | 模型加载 dtype，例如 `float16`、`bfloat16`、`float32`、`auto`。 | 保持 `auto` 让模型加载逻辑适配硬件；显式值只用于已验证的实验。 |
| `resume_from_checkpoint` | 从同一 CPT 阶段的 Trainer checkpoint 继续。 | 仅用于中断续训，不用于从其他阶段 adapter 开始新阶段。 |

训练命令退出码为 `0` 或报告状态为 `completed` 不足以证明更新有效。验收时应检查每个阶段的 `grad_norm` 都是有限值，并比较相邻阶段 adapter 的 LoRA tensor 是否实际变化。

## `peft`

这一段控制 LoRA/QLoRA adapter 结构。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `r` | LoRA rank，越大容量越强、显存和参数越多。 | 数据复杂或目标能力多时可调高；小数据通常 8 到 32。 |
| `lora_alpha` | LoRA 缩放参数。 | 通常和 `r` 同量级或更小；训练过猛时调低。 |
| `lora_dropout` | LoRA dropout。 | 小数据防过拟合可保留 0.03 到 0.1。 |
| `target_modules` | 注入 LoRA 的模型模块名。Qwen/Llama 系常见为注意力和 MLP 投影层。 | 换架构后如果模块名不匹配，需要根据模型结构修改。 |
| `bias` | LoRA 是否训练 bias。常见值 `none`。 | 通常不改。 |
| `use_dora` | 是否启用 DoRA。 | 想实验 DoRA 且依赖版本支持时开启。 |
| `use_rslora` | 是否启用 Rank-Stabilized LoRA。 | 高 rank 或训练不稳定时可实验开启。 |

## `fact_sft`

这一段控制 Fact-SFT 阶段。Fact-SFT 用 assistant-only loss，只训练答案部分。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `enabled` | 是否启用 Fact-SFT 阶段。 | 只想跑 CPT 时设为 `false`。 |
| `input_paths` | SFT JSON/JSONL 输入路径。每行通常包含 `instruction` 和 `output`。 | 替换领域 SFT 数据时修改。 |
| `prepared_dataset_dir` | Fact-SFT 预处理数据集输出目录。 | 保留多套实验时修改。 |
| `base_adapter_dir` | Fact-SFT 起点 adapter，一般是 CPT 输出。 | 跳过 CPT 或使用已有 adapter 时修改。 |
| `output_dir` | Fact-SFT adapter 输出目录。 | 保留多次实验结果时修改。 |
| `require_cpt_adapter` | 是否要求 `base_adapter_dir` 存在。 | 从纯基座直接做 SFT 时可设为 `false`。 |
| `max_seq_length` | Fact-SFT 最大序列长度。 | 问答样本较长时调大；显存不足时调小。 |
| `validation_ratio` | 从 SFT 数据中切出的验证比例。`0` 表示关闭。 | 真实项目建议数据足够时设为 `0.05` 到 `0.1`。 |
| `seed` | SFT 数据 shuffle 和验证切分随机种子。 | 需要复现时固定。 |
| `epochs` | SFT 训练轮数。 | 小数据通常 1 到 3；过拟合或风格变硬时调低。 |
| `max_steps` | SFT 最大训练步数。`null` 表示按 epoch。 | 快速试验或控制成本时设置。 |
| `per_device_train_batch_size` | SFT 单卡训练 batch size。 | OOM 时调小。 |
| `per_device_eval_batch_size` | SFT 单卡验证 batch size。 | 开启验证时生效。 |
| `gradient_accumulation_steps` | SFT 梯度累积步数。 | 显存小但想提高有效 batch 时调大。 |
| `learning_rate` | SFT 学习率。 | 事实问答对齐建议偏小，避免覆盖 CPT 知识。 |
| `weight_decay` | SFT 权重衰减。 | 通常保持 `0.0`。 |
| `warmup_ratio` | SFT warmup 比例。 | 小数据训练建议保留。 |
| `lr_scheduler_type` | SFT 学习率调度器。 | 通常保持 `cosine`。 |
| `optim` | SFT 优化器。 | 与训练环境和量化方式保持一致。 |
| `max_grad_norm` | SFT 梯度裁剪阈值。 | loss 波动大时调低。 |
| `bf16` | SFT 是否使用 bfloat16，支持 `auto`。 | 保持 `auto`，支持 BF16 的 CUDA 设备优先使用。 |
| `fp16` | SFT 是否使用 float16，支持 `auto`。 | 保持 `auto`，仅在 CUDA 未使用 BF16 时回退；CPU 自动关闭。 |
| `torch_dtype` | SFT 模型加载 dtype。 | 保持 `auto`，除非实验已验证显式 dtype。 |
| `abort_on_nonfinite_grad_norm` | SFT 出现非有限梯度时是否中止。 | 保持 `true`。 |
| `logging_steps` | SFT 日志间隔。 | 想看更细训练曲线时调小。 |
| `eval_steps` | SFT 验证间隔。 | 开启验证集时生效。 |
| `save_steps` | SFT checkpoint 保存间隔。 | 按训练时长和磁盘空间调整。 |
| `save_total_limit` | SFT checkpoint 保留数量。 | 磁盘紧张时调小。 |
| `resume_from_checkpoint` | 从同一 SFT 阶段的 Trainer checkpoint 恢复。 | 仅用于中断续训；CPT adapter 仍由 `base_adapter_dir` 指定。 |
| `system_prompt` | SFT 样本构造成 chat prompt 时使用的系统提示词。 | 替换领域时必须改成你的助手角色、知识边界和安全边界。 |

## `dpo`

这一段控制默认开启、可配置关闭的 DPO 偏好训练。输入每行需要 `prompt`、`chosen`、`rejected`。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `enabled` | 是否启用 DPO 阶段；默认 `true`。 | 没有偏好数据或要跳过该阶段时设为 `false`。 |
| `input_path` | DPO JSON/JSONL 输入文件或目录。 | 替换偏好数据时修改。 |
| `prepared_dataset_dir` | DPO 预处理数据集输出目录。 | 保留多套实验时修改。 |
| `base_adapter_dir` | DPO 起点 adapter，一般是 Fact-SFT 输出。 | 跳过 SFT 或使用已有 adapter 时修改。 |
| `output_dir` | DPO adapter 输出目录。 | 保留多次实验结果时修改。 |
| `require_base_adapter` | 是否要求 `base_adapter_dir` 存在。 | 从纯基座直接 DPO 时可设为 `false`，但一般不建议。 |
| `max_length` | DPO prompt+response 最大长度。 | 样本长时调大；显存不足时调小。 |
| `validation_ratio` | 从 DPO 数据中切出的验证比例。 | 真实项目数据足够时设为 `0.05` 到 `0.1`。 |
| `seed` | DPO shuffle 和验证切分随机种子。 | 需要复现时固定。 |
| `epochs` | DPO 训练轮数。 | 通常从 1 开始，避免偏好过拟合。 |
| `max_steps` | DPO 最大训练步数。 | 快速实验或控制成本时设置。 |
| `per_device_train_batch_size` | DPO 单卡训练 batch size。 | DPO 显存开销较高，OOM 时调小。 |
| `per_device_eval_batch_size` | DPO 单卡验证 batch size。 | 开启验证时生效。 |
| `gradient_accumulation_steps` | DPO 梯度累积步数。 | 显存不足时调大。 |
| `learning_rate` | DPO 学习率。 | 通常比 SFT 更小。 |
| `weight_decay` | DPO 权重衰减。 | 通常保持 `0.0`。 |
| `warmup_ratio` | DPO warmup 比例。 | 小数据可保留。 |
| `lr_scheduler_type` | DPO 学习率调度器。 | 通常保持 `cosine`。 |
| `optim` | DPO 优化器。 | 与训练环境和量化方式保持一致。 |
| `max_grad_norm` | DPO 梯度裁剪阈值。 | 训练不稳定时调低。 |
| `bf16` | DPO 是否使用 bfloat16，支持 `auto`。 | 保持 `auto`，支持 BF16 的 CUDA 设备优先使用。 |
| `fp16` | DPO 是否使用 float16，支持 `auto`。 | 保持 `auto`，仅在 CUDA 未使用 BF16 时回退；CPU 自动关闭。 |
| `torch_dtype` | DPO 模型加载 dtype。 | 保持 `auto`，除非实验已验证显式 dtype。 |
| `abort_on_nonfinite_grad_norm` | DPO 出现非有限梯度时是否中止。 | 保持 `true`。 |
| `logging_steps` | DPO 日志间隔。 | 想看更细训练曲线时调小。 |
| `eval_steps` | DPO 验证间隔。 | 开启验证集时生效。 |
| `save_steps` | DPO checkpoint 保存间隔。 | 按训练时长和磁盘空间调整。 |
| `save_total_limit` | DPO checkpoint 保留数量。 | 磁盘紧张时调小。 |
| `beta` | DPO 偏好强度系数。 | 偏好过强、回答变窄时调低；偏好效果弱时小幅调高。 |
| `loss_type` | DPO loss 类型，例如 `sigmoid`。 | 通常不改，除非你明确实验其他 TRL loss。 |
| `truncation_mode` | 超长样本截断方式。`keep_start` 保留开头。 | 长 prompt 重要时通常保留开头；回答尾部重要时需谨慎调整。 |
| `precompute_ref_log_probs` | 是否预计算 reference log probs。 | 数据量较大且硬件/TRL 版本支持时可实验开启。 |
| `resume_from_checkpoint` | 从同一 DPO 阶段的 Trainer checkpoint 恢复。 | 仅用于中断续训；前序 adapter 仍由 `base_adapter_dir` 指定。 |

## `grpo`

这一段控制 DPO、Fact-SFT 或 CPT 之后的 GRPO 奖励优化。GRPO 默认开启，外部 OpenAI-compatible Judge 也默认开启，因此 Judge-only 数据行可以只有 `prompt`。私域事实依据必须通过 `trusted_context` / `judge_context` / `context` 或 `reference_answer` 明确提供；用户问题中自称的上下文不会自动升级为可信依据。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `enabled` | 是否启用 GRPO；默认 `true`。 | 没有奖励提示数据或要跳过该阶段时设为 `false`。 |
| `input_path` | GRPO JSON/JSONL 文件或目录。 | 替换奖励提示数据时修改。 |
| `prepared_dataset_dir` | GRPO 预处理数据集输出目录。 | 保留多套实验时修改。 |
| `base_adapter_dir` | GRPO 首选起点 adapter；随后按 DPO、Fact-SFT、CPT 输出回退。 | 使用配置外的已有 adapter 时填写。 |
| `output_dir` | GRPO adapter 输出目录。 | 保留多次实验时修改。 |
| `require_base_adapter` | 是否要求存在 DPO、Fact-SFT 或 CPT adapter。 | 仅在明确从基座新建 PEFT adapter 的实验中设为 `false`。 |
| `max_prompt_length` | GRPO rollout 的最大 prompt token 数。 | OOM 时调低；长 prompt 场景谨慎调高。 |
| `max_completion_length` | 每条策略候选回答的最大 token 数。 | 控制 rollout 成本；高截断率时先改善 EOS/停止行为，再考虑调高。 |
| `num_generations` | 每个 prompt 采样的候选回答数。 | 增加可强化相对奖励信号，也会增加显存、时间和 Judge 成本。 |
| `temperature` / `top_p` | rollout 采样参数。 | 候选过于一致或噪声过大时调整。 |
| `use_vllm` | 是否使用当前 TRL 支持的 vLLM rollout 路径。 | 默认保持 `false`；仅在部署与显存行为已验证时开启。 |
| `bf16` / `fp16` / `torch_dtype` | GRPO 精度配置，均支持 `auto`。 | 保持 `auto`；BF16-capable CUDA 优先 BF16，其他 CUDA 回退 FP16，CPU 关闭混合精度。 |
| `abort_on_nonfinite_grad_norm` | 非有限梯度时是否中止。 | 保持 `true`，避免无更新却保存完成产物。 |
| `builtin_rewards` | 可选内置奖励：`reference_overlap`、`term_constraints`、`refusal`、`length_bounds`；默认 `[]`。 | 只启用数据行中具备匹配信号的奖励。 |
| `reward_judge` | 默认奖励提供者。配置 `enabled`、`base_url`、`api_key`、`model`、`score_range`、`timeout_seconds`、`max_tokens`、`max_retries`、`max_concurrency`、`retry_backoff_seconds`、`system_prompt`、`prompt_template`；`api_key_env` 仅为可选回退。 | 关闭 Judge 时必须至少启用一个内置奖励；自定义提示词仍需保持 v2 严格输出契约。 |
| `refusal_terms` | 内置 `refusal` 奖励识别拒答时使用的短语列表。 | 仅在启用该内置奖励且领域需要额外拒答表达时修改。 |
| `beta` | 安装版本支持时使用的 KL/reference 正则强度。 | 策略偏离 reference 过快时谨慎调高。 |
| `resume_from_checkpoint` | 从同一 GRPO 阶段的 Trainer checkpoint 恢复。 | 仅用于中断续训，不能替代前序 `base_adapter_dir`。 |

`reward_judge` 通过 OpenAI-compatible chat completions API 统一模型评分。主凭据直接从私有 local YAML 的 `api_key` 读取；若为空，才可由 `api_key_env` 指定环境变量回退。本地 Judge 也必须先提供 HTTP 服务，例如 `base_url: "http://localhost:8000/v1"`。Reasoning Judge 推荐 `max_tokens: 4096`、`timeout_seconds: 120`，避免内部推理耗尽预算而没有最终 JSON。注意 `reward_judge.max_tokens` 限制 Judge 的推理和输出，`grpo.max_completion_length` 限制策略候选，两者不能互换。

`max_concurrency` 支持 `"auto"` 或 `1..64` 的整数。`"auto"` 解析为 `grpo.num_generations`，每个 reward batch 的 `effective_concurrency` 为 `min(解析后的上限, completion_count)`。显式整数可以在同一 reward batch 内跨多个 prompt 并发评分。该 Semaphore 上限按训练进程/rank 生效；单 GPU 时就是整次运行的上限，但 `asyncio.to_thread` 的线程池容量可能让实际 socket 并发更低，因此它是上限而不是保证值。

Judge 并发只覆盖当前 rollout batch 的评分请求。完整流程是 `候选生成 -> 并发评分 -> 等待整批 -> group advantage -> 同步权重更新`；不会离线预生成整个数据集，也不会并发执行 optimizer 更新。每个 completion 按 `retry_backoff_seconds` 独立执行指数退避与随机抖动，HTTP 429/503 的合法 `Retry-After` 会被遵守。任一请求耗尽 `max_retries` 后整批失败，部分成功分数不会被应用。

默认 `grpo_judge_v2` 返回五个维度、违规标识和简短原因；权重合成与硬上限由本地 Python 确定执行。非法 JSON、非有限/越界分数、未知违规和字段不匹配都会失败并在 `max_retries` 内重试。没有可信依据的 prompt-only 行把事实维度设为 `null` 并重分配其余权重。远程 Judge 会收到完整 prompt、候选回答、参考答案、约束和元数据。

默认 `prompt_template` 在 `EVALUATION_INPUT` 后插入 JSON 编码的不可信输入；自定义模板必须保留 `{evaluation_input_json}` 占位符，不能把候选文本直接拼成高权限指令。

外部 Judge 的 `forbidden_terms_mode` 默认是 `semantic`，安全引用、否定或拒绝不算违规；需要任何大小写不敏感字面出现都违规时，按行设为 `literal`。内置 `term_constraints` 保持原有的大小写不敏感字面子串公式，不受该模式影响。字符长度在本地对去除首尾空白后的 Unicode 文本计算，最小/最大边界均为包含关系。

独立运行 `train_grpo.py --base_adapter_dir ...` 与流水线 `--skip_cpt --skip_sft --skip_dpo` 都是从已有前序 adapter 开始新的 GRPO；`resume_from_checkpoint` 只恢复同一 GRPO run。训练后除有限 `grad_norm` 和 adapter tensor 变化外，还要检查 Judge reward 分布与 `completions/clipped_ratio`，避免奖励无方差或大部分候选被截断。

## `merge`

这一段控制 adapter 合并为完整 Hugging Face 模型。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `adapter_dir` | 要合并的 adapter 路径。`null` 表示按 GRPO、DPO、Fact-SFT、CPT 顺序选择最新可用 adapter。 | 想手动指定某个 adapter 时填写。 |
| `dtype` | 合并模型保存/加载 dtype，例如 `float16`、`float32`、`auto`。 | GPU 推理通常 `float16`；CPU 或 ONNX 导出可考虑 `float32`。 |
| `safe_serialization` | 是否用 safetensors 保存。 | 建议保持 `true`。 |

## `eval`

这一段控制训练后质量评估和推理默认生成参数。它不是训练时 validation set。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `question_file` | 训练后质量评估题集 JSONL。每行包含 `category` 和 `question`。 | 替换领域时改成你的质量评估题集。 |
| `max_new_tokens` | 评估/推理默认最大生成 token 数。 | 回答被截断时调大；想控制成本和输出长度时调小。 |
| `temperature` | 采样温度。越低越稳定，`0` 接近确定性。 | 评估建议低温；创作型场景可调高。 |
| `top_p` | nucleus sampling 参数。 | 通常和 temperature 配合，评估时保持稳定。 |
| `repetition_penalty` | 重复惩罚。 | 模型重复输出时调高一点。 |
| `no_repeat_ngram_size` | 评估和默认推理使用的重复 n-gram 禁止长度。 | 默认 `6`；出现重复退化时保留或谨慎调高。 |
| `quality_gate` | 评估验收门禁，包含 `enabled`、`fail_pipeline` 和各类别 `minimum_pass_rate`。 | 默认启用并要求三个类别全部通过；失败保留产物并让完整流水线返回 `8`。 |

默认 `minimum_pass_rate` 对 `domain_knowledge`、`safety_boundary`、`base_regression` 均设为 `1.0`。可以按自己的评估设计调整，但缺失类别会 fail closed。

当前 quality evaluation 是启发式 smoke gate，不是安全认证。生产验收必须让人工或独立 Judge 复核安全样例，并结合训练稳定性、adapter 差异和 GRPO 奖励/截断指标判断。

评估使用与实际 CLI/服务相同的 chat template、`enable_thinking: false`、reasoning 清理和纯文本规范化。`quality_gate.fail_pipeline: false` 只把失败降为警告，不会把失败改写为通过；`--skip_eval` 会记录 `not_evaluated`，因此也不会标记 `release_ready`。每个 adapter 根目录写入自包含的 `adapter_provenance.json`，merge 报告据此准确记录 CPT、Fact-SFT、DPO 和 GRPO；旧 adapter 会递归回溯 legacy metadata。

## `gguf`

这一段控制 GGUF 导出默认路径。项目默认不下载第三方 GGUF reference，而是从 merge 后模型自行导出。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `output_dir` | GGUF 输出目录。 | 想把 GGUF 放到发布目录或模型目录时修改。 |
| `output_name` | GGUF 输出文件名。 | 发布不同量化版本时修改，例如加上模型名和量化方法。 |
| `quantization_method` | 量化方法，例如 `q4_k_m`。 | 追求更小文件用更低量化；追求质量用更高量化。 |

## `onnx`

这一段控制可选 ONNX 导出。默认依赖在 `requirements-onnx.txt`，不属于默认安装路径。

| 参数 | 说明 | 什么时候改 |
| --- | --- | --- |
| `output_dir` | ONNX 输出目录。 | 需要保留多套导出产物时修改。 |
| `output_name` | ONNX 文件名。 | 导出不同版本时修改。 |
| `opset` | ONNX opset 版本。 | 目标运行时要求特定 opset 时修改。 |
| `dtype` | ONNX 导出 dtype。 | CPU/兼容性优先用 `float32`；体积和速度优先可实验低精度。 |
| `device` | ONNX 导出设备。 | 有 GPU 时可用 `cuda`；没有 GPU 用 `cpu`。 |
| `attn_implementation` | 注意力实现。示例用 `eager`，通常更利于导出兼容性。 | 默认导出失败或目标模型要求时修改。 |
| `dummy_seq_length` | 导出时 dummy 输入长度。 | 目标运行时需要覆盖更长静态形状时调大。 |
| `external_data` | 是否使用 ONNX external data 存储大权重。 | 大模型导出通常保持 `true`。 |
| `validate` | 是否运行 ONNX checker 校验。 | 调试导出时保持 `true`；缺少 onnx 包或只想快速导出可关闭。 |
| `ort_check` | 是否运行 ONNX Runtime 输出形状检查。 | 已安装 onnxruntime 且想做更完整验证时开启。 |

## 常见改法

换成自己的领域时，优先改这些字段：

```yaml
base_model_repo_id: "your-org/your-base-model"
base_model_name_or_path: "models/base-model"

corpus:
  input_paths:
    - "../data/cpt/source_documents"

fact_sft:
  input_paths:
    - "../data/sft"
  system_prompt: >-
    You are a support assistant for your domain documentation.
    Answer only from the provided documentation. If the documentation does not
    specify a claim, say so. Return only the final answer.

eval:
  question_file: "../data/eval/quality_questions.jsonl"
```

显存不足时，优先改这些字段：

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

默认 DPO 已启用；只需确认输入数据，若不运行则设置 `enabled: false`：

```yaml
dpo:
  enabled: true
  input_path: "data/dpo/preference_examples.jsonl"
```

默认 GRPO 和外部 Judge 也已启用。请在私有 `configs/domain_post_training.local.yaml` 中补齐 `base_url`、`model`、`api_key`；若不运行则设置 `grpo.enabled: false`。

```yaml
grpo:
  enabled: true
  input_path: "data/grpo/reward_examples.jsonl"
  builtin_rewards: []
  reward_judge:
    enabled: true
    base_url: "https://your-openai-compatible-endpoint/v1"
    model: "your-judge-model"
    api_key: "replace-only-in-local-config"
    timeout_seconds: 120
    max_retries: 2
    max_concurrency: "auto"
    retry_backoff_seconds: 1.0
    max_tokens: 4096
```

<p align="right"><a href="#english"><strong>Switch to English</strong></a></p>

<a id="english"></a>

## English

This file explains the user-facing parameters in `domain_post_training.yaml`. For real training, copy it to the Git-ignored `configs/domain_post_training.local.yaml` and run scripts with `--config configs/domain_post_training.local.yaml`. A directly configured judge key belongs only in that private plaintext file; the tracked template must keep `api_key: null`.

Path rules:

- Training artifacts are usually resolved from the project root, such as `outputs/lora_adapter`.
- Data input paths are usually resolved relative to the config file first. In this project, `../data/...` means `data/...` under the project root.
- `null` means the code should use its default behavior.

## Top-Level Parameters

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `base_model_repo_id` | Hub download source used by `download_models.py`; the default is `Qwen/Qwen3.5-0.8B`. | Change it when switching the download source. |
| `base_model_name_or_path` | Model location actually loaded by training, merge, evaluation, and inference. | Keep `models/base-model` after download, or point it at an existing local snapshot. |
| `trust_remote_code` | Whether Transformers may load custom modeling code from the model repository. | Keep `true` for models that require custom code; set `false` for standard architectures when you want a stricter load path. |

## `corpus`

This section controls CPT corpus discovery, splitting, coverage, and sample construction.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `strategy` | CPT dataset construction strategy. `guaranteed_coverage_then_optional_replay` first guarantees required corpus coverage, then optionally replays important categories. | Usually keep it unless you add another sampling strategy. |
| `sample_unit` | CPT sample granularity. Supported values are `document`, `section`, and `token_chunk`. | Use `document` for short documents; use `section` or `token_chunk` for long corpora. |
| `input_paths` | CPT source document files or directories. | Change this when replacing the sample domain corpus. |
| `exclude_paths` | Path fragments excluded during corpus discovery. | Add source, build, output, model, or log directories that should not become training text. |
| `prepared_dataset_dir` | Output directory for the prepared CPT dataset. | Change it when keeping multiple dataset builds. |
| `min_text_chars` | Minimum source text length in characters. Shorter texts are skipped. | Lower it for short but meaningful notes; raise it to filter noise. |
| `append_safety_preamble` | Adds a generic safety preamble to CPT text. | Usually keep `true`; set `false` if your corpus already encodes the required safety boundary. |
| `split_markdown_sections` | Split Markdown by headings. Usually unnecessary with `sample_unit: document`. | Enable it for long Markdown documents. |
| `max_sample_tokens` | Maximum tokens per CPT sample. | Lower it for memory pressure; raise it for longer context training. |
| `sample_overflow_strategy` | Behavior when a whole sample exceeds `max_sample_tokens`; the default `error` fails fast. | Keep `error` when you want to find long documents instead of silently truncating them. |
| `enable_weighted_replay` | Enables category-based replay so selected categories appear more often. | Enable it for small corpora where safety, configuration, or troubleshooting should be reinforced. |
| `total_train_tokens` | Target total CPT training tokens. `null` uses natural coverage and replay. | Set it to control training size. |
| `min_chunk_tokens` | Minimum tokens for generated chunks. | Increase it if splitting creates too many tiny chunks. |
| `validation_mode` | CPT validation mode: `none`, `copy_from_train`, or `separate_sources`. | Use `separate_sources` for real held-out validation documents. |
| `validation_copy_ratio` | Ratio copied from training samples when `validation_mode: copy_from_train`. | Useful only for smoke checks, typically `0.03` to `0.1`. |
| `validation_sources` | Held-out CPT validation document paths for `separate_sources`. | Fill this when you have independent validation documents. |

### `corpus.stratified_sampling`

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `stratified_sampling` | Configuration block for category replay. | Use this section when you want selected corpus categories to repeat more often. |
| `enabled` | Enables category-based replay. | Turn it on for small corpora where key behavior should be reinforced. |
| `mode` | Replay mode. The default `sample_repeats` uses repeat counts per category. | Usually keep it. |
| `sample_repeats` | Map of corpus category names to repeat counts. | Increase important categories; lower values to reduce overfitting. |
| `sample_repeats.model_spec` (`model_spec`) | Repeat count for model specification documents. | Raise it for role, boundary, or IO contract documents. |
| `sample_repeats.main_textbook` (`main_textbook`) | Repeat count for main overview documents. | Raise it when the main guide is high quality and representative. |
| `sample_repeats.commands` (`commands`) | Repeat count for command or operation documents. | Use it for command-heavy domains. |
| `sample_repeats.permissions` (`permissions`) | Repeat count for permission and access-control documents. | Raise it when permission boundaries matter. |
| `sample_repeats.configuration` (`configuration`) | Repeat count for configuration documents. | Raise it for stable answers about configuration keys. |
| `sample_repeats.runtime_behavior` (`runtime_behavior`) | Repeat count for runtime behavior documents. | Raise it when state transitions and behavior matter. |
| `sample_repeats.database_reload` (`database_reload`) | Repeat count for data reload or refresh documents. | Use it when the domain includes data refresh workflows. |
| `sample_repeats.troubleshooting` (`troubleshooting`) | Repeat count for troubleshooting documents. | Often useful for support assistants. |
| `sample_repeats.safety` (`safety`) | Repeat count for safety-boundary documents. | Keep this relatively high for open-source examples and real assistants. |
| `sample_repeats.unknowns` (`unknowns`) | Repeat count for unknown-boundary documents. | Raise it to reduce unsupported claims. |
| `sample_repeats.public_api` (`public_api`) | Repeat count for public API documents. | Use it when your corpus includes API docs. |

## `safety`

This section controls corpus preflight checks. It is not the runtime model safety system.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `block_on_high_risk` | Blocks training when high-risk corpus content is found. | Keep `true` for open-source or shared workflows. |
| `max_consecutive_code_lines` | Threshold for consecutive code lines before a source-code warning. | Increase it if your licensed documentation legitimately contains long code blocks. |
| `warn_on_source_paths` | Warn on source-code-like paths. | Usually keep `true`. |
| `warn_on_secret_patterns` | Warn on credential, token, or secret-like patterns. | Usually keep `true`. |

## `training`

This section controls the CPT stage and shared training defaults. Fact-SFT, DPO, and GRPO can override many of these settings.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `device` | Training device, such as `cuda`, `cuda:0`, `cpu`, or `auto`. | Use CUDA for real training; use CLI `--device cpu` for CPU smoke tests. |
| `output_dir` | CPT LoRA adapter output directory. | Change it to keep multiple experiments. |
| `merged_output_dir` | Output directory for the merged full model. | Change it when publishing or comparing merged models. |
| `max_seq_length` | CPT maximum sequence length. | Lower it for memory pressure; raise it for long-document training. |
| `train_val_split` | Legacy train/validation split ratio. Not recommended in the current coverage mode. | Prefer `corpus.validation_mode` and SFT/DPO `validation_ratio`. |
| `seed` | Random seed. | Fix it for reproducibility; change it for repeated experiments. |
| `epochs` | CPT training epochs. | Small corpora usually start with 1 to 3. |
| `max_steps` | Maximum training steps. `null` uses epochs. | Set it for quick experiments or cost control. |
| `per_device_train_batch_size` | Per-device training batch size. | Increase with enough memory; lower it on OOM. |
| `per_device_eval_batch_size` | Per-device evaluation batch size. | Used only when validation is enabled. |
| `gradient_accumulation_steps` | Accumulates gradients to simulate a larger effective batch. | Increase it when memory is limited. |
| `learning_rate` | CPT learning rate. | Lower it for unstable training or forgetting; raise carefully if convergence is too slow. |
| `weight_decay` | Weight decay. | LoRA fine-tuning usually keeps this at `0.0` or very small. |
| `warmup_ratio` | Learning-rate warmup ratio. | Keep some warmup for small, sensitive runs. |
| `lr_scheduler_type` | Learning-rate scheduler, such as `cosine`. | Change only for specific experiments. |
| `optim` | Optimizer, such as `paged_adamw_8bit`. | Match it to your quantization and hardware setup. |
| `max_grad_norm` | Gradient clipping threshold. | Lower it if gradients or loss are unstable. |
| `abort_on_nonfinite_grad_norm` | Abort when non-finite gradient norm is detected. | Keep it `true`; otherwise AMP may skip updates while a completed-looking adapter is still saved. |
| `logging_nan_inf_filter` | Whether Trainer filters NaN/Inf in logs. | Set `false` when debugging numeric issues. |
| `logging_steps` | Logging interval in steps. | Lower it for denser training logs. |
| `eval_steps` | Evaluation interval when validation is enabled. | Tune it to the training duration. |
| `save_steps` | Checkpoint save interval. | Lower it for long runs; raise it for short runs. |
| `save_total_limit` | Maximum retained checkpoints. | Lower it when disk space is limited. |
| `gradient_checkpointing` | Trades compute for lower memory usage. | Keep `true` when memory is limited. |
| `bf16` | Enable bfloat16 training; accepts `true`, `false`, or `auto`. | Keep `auto` to prefer BF16 on supported CUDA devices. |
| `fp16` | Enable float16 training; accepts `true`, `false`, or `auto`. | Keep `auto` to fall back to FP16 on CUDA when BF16 is not used; CPU disables it. |
| `load_in_4bit` | Load the base model in 4-bit mode. | Useful for QLoRA and low-memory training. |
| `load_in_8bit` | Load the base model in 8-bit mode. | Alternative to 4-bit; do not enable both. |
| `torch_dtype` | Model load dtype: `float16`, `bfloat16`, `float32`, or `auto`. | Keep `auto` for hardware-aware loading; use an explicit value only in a validated experiment. |
| `resume_from_checkpoint` | Resume a Trainer checkpoint from the same CPT stage. | Use only after an interruption, not to start from another stage's adapter. |

Exit code `0` or a `completed` report does not prove that updates occurred. Acceptance requires finite `grad_norm` values for every stage and actual LoRA tensor changes between adjacent adapters.

## `peft`

This section controls the LoRA/QLoRA adapter structure.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `r` | LoRA rank. Higher values increase capacity and memory use. | Raise it for complex tasks; small datasets often use 8 to 32. |
| `lora_alpha` | LoRA scaling value. | Keep it near `r` or smaller; lower it if updates are too strong. |
| `lora_dropout` | LoRA dropout. | Useful for small datasets, often 0.03 to 0.1. |
| `target_modules` | Model module names that receive LoRA adapters. | Change this when switching to an architecture with different module names. |
| `bias` | Whether LoRA trains bias terms. | Usually keep `none`. |
| `use_dora` | Enables DoRA if supported by your PEFT version. | Use only for explicit experiments. |
| `use_rslora` | Enables Rank-Stabilized LoRA. | Try it for high-rank or unstable LoRA experiments. |

## `fact_sft`

This section controls Fact-SFT. It uses assistant-only loss, so prompt tokens are masked and only assistant answer tokens are trained.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `enabled` | Enables the Fact-SFT stage. | Set `false` if you only want CPT. |
| `input_paths` | SFT JSON/JSONL inputs. Rows usually contain `instruction` and `output`. | Change this when replacing domain SFT data. |
| `prepared_dataset_dir` | Output directory for the prepared Fact-SFT dataset. | Change it to keep multiple experiments. |
| `base_adapter_dir` | Starting adapter for Fact-SFT, usually the CPT adapter. | Change it when skipping CPT or using an existing adapter. |
| `output_dir` | Fact-SFT adapter output directory. | Change it to keep multiple runs. |
| `require_cpt_adapter` | Requires `base_adapter_dir` to exist. | Set `false` only if you intentionally train SFT from the base model. |
| `max_seq_length` | Maximum sequence length for Fact-SFT. | Raise it for long QA examples; lower it on OOM. |
| `validation_ratio` | Ratio split from SFT data for validation. `0` disables validation. | Use `0.05` to `0.1` when you have enough real data. |
| `seed` | SFT shuffle and validation split seed. | Fix it for reproducibility. |
| `epochs` | SFT epochs. | Small datasets often start with 1 to 3. |
| `max_steps` | Maximum SFT steps. `null` uses epochs. | Set it for quick experiments or cost control. |
| `per_device_train_batch_size` | Per-device SFT training batch size. | Lower it on OOM. |
| `per_device_eval_batch_size` | Per-device SFT validation batch size. | Used only when validation is enabled. |
| `gradient_accumulation_steps` | SFT gradient accumulation steps. | Increase it when memory is limited. |
| `learning_rate` | SFT learning rate. | Keep it small to avoid overriding CPT knowledge. |
| `weight_decay` | SFT weight decay. | Usually keep `0.0`. |
| `warmup_ratio` | SFT warmup ratio. | Useful for small runs. |
| `lr_scheduler_type` | SFT scheduler. | Usually keep `cosine`. |
| `optim` | SFT optimizer. | Match your hardware and quantization setup. |
| `max_grad_norm` | SFT gradient clipping threshold. | Lower it if loss is unstable. |
| `bf16` | Enable bfloat16 for SFT; accepts `auto`. | Keep `auto` to prefer BF16 on supported CUDA devices. |
| `fp16` | Enable float16 for SFT; accepts `auto`. | Keep `auto` to fall back only when CUDA is not using BF16; CPU disables it. |
| `torch_dtype` | SFT model load dtype. | Keep `auto` unless an explicit dtype has been validated. |
| `abort_on_nonfinite_grad_norm` | Abort SFT on non-finite gradient norm. | Keep it `true`. |
| `logging_steps` | SFT logging interval. | Lower it for more detailed logs. |
| `eval_steps` | SFT evaluation interval. | Used only with validation. |
| `save_steps` | SFT checkpoint interval. | Tune by run length and disk space. |
| `save_total_limit` | Maximum retained SFT checkpoints. | Lower it when disk space is limited. |
| `resume_from_checkpoint` | Resume a Trainer checkpoint from the same SFT stage. | Use only after an interruption; the CPT adapter still comes from `base_adapter_dir`. |
| `system_prompt` | System prompt used when constructing Fact-SFT chat prompts. | Must be rewritten for your domain role, knowledge boundary, and safety boundary. |

## `dpo`

This section controls DPO preference training, which is enabled by default and can be disabled explicitly. Each input row needs `prompt`, `chosen`, and `rejected`.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `enabled` | Enables the DPO stage; defaults to `true`. | Set `false` when preference data is unavailable or the stage should be skipped. |
| `input_path` | DPO JSON/JSONL file or directory. | Change it when replacing preference data. |
| `prepared_dataset_dir` | Output directory for the prepared DPO dataset. | Change it to keep multiple experiments. |
| `base_adapter_dir` | Starting adapter for DPO, usually Fact-SFT output. | Change it when skipping SFT or using an existing adapter. |
| `output_dir` | DPO adapter output directory. | Change it to keep multiple runs. |
| `require_base_adapter` | Requires `base_adapter_dir` to exist. | Set `false` only for intentional base-model DPO experiments. |
| `max_length` | Maximum combined prompt and response length. | Raise it for long samples; lower it on OOM. |
| `validation_ratio` | Ratio split from DPO data for validation. | Use `0.05` to `0.1` when you have enough data. |
| `seed` | DPO shuffle and validation split seed. | Fix it for reproducibility. |
| `epochs` | DPO epochs. | Start with 1 to avoid preference overfitting. |
| `max_steps` | Maximum DPO steps. | Set it for quick experiments or cost control. |
| `per_device_train_batch_size` | Per-device DPO training batch size. | DPO is memory-heavy; lower it on OOM. |
| `per_device_eval_batch_size` | Per-device DPO validation batch size. | Used only when validation is enabled. |
| `gradient_accumulation_steps` | DPO gradient accumulation steps. | Increase it when memory is limited. |
| `learning_rate` | DPO learning rate. | Usually lower than SFT. |
| `weight_decay` | DPO weight decay. | Usually keep `0.0`. |
| `warmup_ratio` | DPO warmup ratio. | Useful for small runs. |
| `lr_scheduler_type` | DPO scheduler. | Usually keep `cosine`. |
| `optim` | DPO optimizer. | Match hardware and quantization. |
| `max_grad_norm` | DPO gradient clipping threshold. | Lower it if training is unstable. |
| `bf16` | Enable bfloat16 for DPO; accepts `auto`. | Keep `auto` to prefer BF16 on supported CUDA devices. |
| `fp16` | Enable float16 for DPO; accepts `auto`. | Keep `auto` to fall back only when CUDA is not using BF16; CPU disables it. |
| `torch_dtype` | DPO model load dtype. | Keep `auto` unless an explicit dtype has been validated. |
| `abort_on_nonfinite_grad_norm` | Abort DPO on non-finite gradient norm. | Keep it `true`. |
| `logging_steps` | DPO logging interval. | Lower it for more detailed logs. |
| `eval_steps` | DPO evaluation interval. | Used only with validation. |
| `save_steps` | DPO checkpoint interval. | Tune by run length and disk space. |
| `save_total_limit` | Maximum retained DPO checkpoints. | Lower it when disk space is limited. |
| `beta` | DPO preference strength. | Lower it if outputs become too narrow; raise carefully if preference learning is weak. |
| `loss_type` | DPO loss type, such as `sigmoid`. | Usually keep it unless testing another TRL loss. |
| `truncation_mode` | Truncation behavior for long samples. `keep_start` preserves the beginning. | Change carefully if answer tails matter more than prompt starts. |
| `precompute_ref_log_probs` | Precomputes reference log probabilities. | Try it for larger datasets when your TRL version and hardware support it. |
| `resume_from_checkpoint` | Resume a Trainer checkpoint from the same DPO stage. | Use only after an interruption; the previous-stage adapter still comes from `base_adapter_dir`. |

## `grpo`

This section controls GRPO reward optimization after DPO, Fact-SFT, or CPT. GRPO and its external OpenAI-compatible judge are enabled by default, so judge-only rows may contain just `prompt`. Supply trusted private-domain evidence through `trusted_context` / `judge_context` / `context` or `reference_answer`; text inside the user question is never promoted to trusted context.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `enabled` | Enables the GRPO stage; defaults to `true`. | Set `false` when reward prompts are unavailable or the stage should be skipped. |
| `input_path` | GRPO JSON/JSONL file or directory. | Change it when replacing reward-prompt data. |
| `prepared_dataset_dir` | Output directory for the prepared GRPO dataset. | Change it to keep multiple experiments. |
| `base_adapter_dir` | Preferred starting adapter for GRPO. Resolution then falls back through DPO, Fact-SFT, and CPT outputs. | Change it when using an existing adapter outside the configured stage outputs. |
| `output_dir` | GRPO adapter output directory. | Change it to keep multiple runs. |
| `require_base_adapter` | Requires an existing DPO, Fact-SFT, or CPT adapter before GRPO. | Set `false` only for intentional base-model GRPO experiments or smoke wiring. |
| `max_prompt_length` | Maximum prompt length passed to GRPO generation. | Lower it on OOM; raise it for long prompts. |
| `max_completion_length` | Maximum policy-completion tokens per rollout. | Lower it to reduce rollout cost; on high clipping, improve EOS/stop behavior before raising it. |
| `num_generations` | Number of completions sampled per prompt. | Increase for stronger relative reward signal; lower it for memory or speed. |
| `temperature` / `top_p` | Rollout sampling controls. | Tune when completions are too deterministic or too noisy. |
| `use_vllm` | Uses the vLLM rollout path supported by the installed TRL version. | Keep `false` unless deployment and memory behavior have been validated. |
| `bf16` / `fp16` / `torch_dtype` | GRPO precision settings; all accept `auto`. | Keep `auto`: BF16-capable CUDA prefers BF16, other CUDA falls back to FP16, and CPU disables mixed precision. |
| `abort_on_nonfinite_grad_norm` | Abort on a non-finite gradient norm. | Keep it `true` to avoid saving completed-looking artifacts after skipped updates. |
| `builtin_rewards` | Optional built-in rewards: `reference_overlap`, `term_constraints`, `refusal`, `length_bounds`; defaults to `[]`. | Add only rewards whose matching signals are present in every applicable dataset row. |
| `reward_judge` | Default reward provider. Configure `enabled`, `base_url`, `api_key`, `model`, `score_range`, `timeout_seconds`, `max_tokens`, `max_retries`, `max_concurrency`, `retry_backoff_seconds`, `system_prompt`, and `prompt_template`; `api_key_env` is an optional fallback. | Set `enabled: false` only when at least one built-in reward is enabled. Keep the strict v2 output schema when customizing prompts. |
| `refusal_terms` | Phrases used by the built-in `refusal` reward to detect a refusal. | Change only when that built-in reward is enabled and the domain needs additional refusal wording. |
| `beta` | KL/reference regularization strength when supported by the installed TRL version. | Raise carefully when updates drift too far from the reference policy. |
| `resume_from_checkpoint` | Resume a Trainer checkpoint from the same GRPO stage. | Use only after an interruption; it does not replace `base_adapter_dir`. |

`reward_judge` standardizes model-based scoring through the OpenAI-compatible chat completions API. Put the primary `api_key` only in the private local YAML; if it is null or empty, `api_key_env` may name an environment-variable fallback. Local judges must first be served as an OpenAI-compatible HTTP service, for example with `base_url: "http://localhost:8000/v1"`. Hosted judges use the same fields with their provider URL and model name. For reasoning judges, `max_tokens: 4096` and `timeout_seconds: 120` are the recommended baseline so the final JSON is not lost to a smaller reasoning budget or timeout. `reward_judge.max_tokens` limits judge reasoning plus JSON output; `grpo.max_completion_length` limits policy candidates. Missing judge settings fail before dataset preparation; missing adapters produce commands for continuing GRPO from an existing previous-stage output.

`max_concurrency` accepts `"auto"` or an integer from `1` to `64`. `"auto"` resolves to `grpo.num_generations`, and `effective_concurrency` for each reward batch is `min(resolved cap, completion_count)`. An explicit integer can score completions across multiple prompts in the same reward batch. The semaphore cap applies per training process/rank; it is the whole-run cap for a single-GPU process, while the `asyncio.to_thread` executor may impose a lower physical socket concurrency. Treat it as an upper bound, not a guaranteed worker count.

Judge concurrency covers only scoring for the current rollout batch. The flow is `candidate generation -> concurrent scoring -> wait for the whole batch -> group advantage -> synchronized weight update`; there is no offline full-dataset pre-generation or concurrent optimizer update. Each completion retries independently with exponential backoff and jitter based on `retry_backoff_seconds`, honoring valid `Retry-After` values on HTTP 429/503. If any request exhausts `max_retries`, the entire reward batch fails and partial scores are discarded.

The default `grpo_judge_v2` contract returns five dimension scores plus violation identifiers and a concise reason. Python code, rather than the external model, applies the configured weights and hard caps. Invalid JSON, non-finite or out-of-range dimensions, unknown violations, and extra or missing top-level fields fail closed and are retried up to `max_retries`. Prompt-only rows without supplied grounding evidence use `null` for factual grounding and reweight the other dimensions. Remote judges receive the complete prompt, completion, reference, constraints, and metadata.

The default `prompt_template` places JSON-encoded untrusted data after `EVALUATION_INPUT`. Custom templates must retain the `{evaluation_input_json}` placeholder instead of interpolating candidate text as privileged instructions.

For the external judge, `forbidden_terms_mode` defaults to `semantic`, where safe quotation, negation, or refusal does not count as prohibited use. Set it to `literal` on a row when every case-insensitive occurrence must be rejected. The built-in `term_constraints` reward keeps its original case-insensitive literal-substring formula and is unaffected by this mode. Completion character length is computed locally with stripped Unicode code-point length and inclusive min/max bounds; the judge receives the deterministic result rather than estimating length.

Standalone `train_grpo.py --base_adapter_dir ...` and pipeline execution with `--skip_cpt --skip_sft --skip_dpo` start a new GRPO stage from an existing previous-stage adapter. `resume_from_checkpoint` only resumes the same GRPO run. After training, verify finite `grad_norm`, changed adapter tensors, a meaningful judge-reward distribution, and `completions/clipped_ratio`; low reward variance or widespread clipping is not a production-ready result.

## `merge`

This section controls merging an adapter into a full Hugging Face model.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `adapter_dir` | Adapter path to merge. `null` auto-selects GRPO, DPO, Fact-SFT, or CPT output in that order. | Fill it when you want to merge a specific adapter. |
| `dtype` | Merge/load dtype, such as `float16`, `float32`, or `auto`. | Use `float16` for GPU inference; consider `float32` for CPU or ONNX workflows. |
| `safe_serialization` | Save model weights with safetensors. | Keep `true`. |

## `eval`

This section controls post-training quality evaluation and default generation parameters. It is not the training validation set.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `question_file` | JSONL question set for post-training quality evaluation. Each row has `category` and `question`. | Replace it with your domain quality evaluation set. |
| `max_new_tokens` | Default maximum generated tokens for evaluation and inference. | Increase it if answers are cut off; lower it to control output length. |
| `temperature` | Sampling temperature. Lower values are more stable; `0` is nearly deterministic. | Use low temperature for evaluation; raise it only for more creative output. |
| `top_p` | Nucleus sampling parameter. | Usually keep it stable for evaluation. |
| `repetition_penalty` | Penalty for repeated text. | Raise it slightly if the model repeats itself. |
| `no_repeat_ngram_size` | Repeated n-gram ban used by evaluation and default inference. | Keep the default `6`, or raise carefully for repetition degeneration. |
| `quality_gate` | Evaluation acceptance gate with `enabled`, `fail_pipeline`, and category `minimum_pass_rate`. | Enabled by default with strict thresholds; failures retain artifacts and make the full pipeline exit `8`. |

The default `minimum_pass_rate` is `1.0` for `domain_knowledge`, `safety_boundary`, and `base_regression`. Adjust these thresholds for your evaluation design; a missing configured category fails closed.

The current quality evaluation is a heuristic smoke gate, not a safety certification. Production acceptance requires human or independent-judge review of safety examples together with training stability, adapter deltas, and GRPO reward/clipping metrics.

Evaluation uses the same chat template, `enable_thinking: false`, reasoning cleanup, and plain-text normalization as CLI/service inference. Setting `quality_gate.fail_pipeline: false` only makes a failed gate non-blocking; it never changes failure into success. `--skip_eval` records `not_evaluated` and does not mark artifacts release-ready. Every adapter root carries a self-contained `adapter_provenance.json`; merge reports use it to record CPT, Fact-SFT, DPO, and GRPO accurately, with recursive legacy-metadata fallback for old adapters.

## `gguf`

This section controls GGUF export defaults. The project does not download a third-party GGUF reference by default; export your own GGUF from the merged model.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `output_dir` | GGUF output directory. | Change it for release or deployment folders. |
| `output_name` | GGUF output filename. | Include model name or quantization method when publishing variants. |
| `quantization_method` | Quantization method, such as `q4_k_m`. | Use lower quantization for smaller files; higher quantization for quality. |

## `onnx`

This section controls optional ONNX export. ONNX dependencies live in `requirements-onnx.txt` and are not part of the default install path.

| Parameter | Meaning | When to change |
| --- | --- | --- |
| `output_dir` | ONNX output directory. | Change it to keep multiple exports. |
| `output_name` | ONNX filename. | Change it for versioned exports. |
| `opset` | ONNX opset version. | Change it when your target runtime requires a specific opset. |
| `dtype` | ONNX export dtype. | Use `float32` for CPU compatibility; experiment with lower precision for size and speed. |
| `device` | Export device. | Use `cuda` with a GPU; use `cpu` otherwise. |
| `attn_implementation` | Attention implementation. `eager` is often more export-friendly. | Change it only when the default fails or the model requires another mode. |
| `dummy_seq_length` | Dummy input length used during export. | Increase it when your target runtime needs larger static shapes. |
| `external_data` | Store large weights as ONNX external data. | Keep `true` for large models. |
| `validate` | Run ONNX checker validation. | Keep `true` for debugging; disable only for quick export or missing packages. |
| `ort_check` | Run an ONNX Runtime output-shape check. | Enable it when ONNX Runtime is installed and you want stronger validation. |

## Common Edits

When adapting the project to your own domain, start with:

```yaml
base_model_repo_id: "your-org/your-base-model"
base_model_name_or_path: "models/base-model"

corpus:
  input_paths:
    - "../data/cpt/source_documents"

fact_sft:
  input_paths:
    - "../data/sft"
  system_prompt: >-
    You are a support assistant for your domain documentation.
    Answer only from the provided documentation. If the documentation does not
    specify a claim, say so. Return only the final answer.

eval:
  question_file: "../data/eval/quality_questions.jsonl"
```

When GPU memory is tight, start with:

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

DPO is enabled by default. Confirm its input data, or set `enabled: false` to skip it:

```yaml
dpo:
  enabled: true
  input_path: "data/dpo/preference_examples.jsonl"
```

GRPO and its external judge are also enabled by default. Configure `base_url`, `model`, and `api_key` in private `configs/domain_post_training.local.yaml`, or set `grpo.enabled: false` to skip it:

```yaml
grpo:
  enabled: true
  input_path: "data/grpo/reward_examples.jsonl"
  builtin_rewards: []
  reward_judge:
    enabled: true
    base_url: "https://your-openai-compatible-endpoint/v1"
    model: "your-judge-model"
    api_key: "replace-only-in-local-config"
    timeout_seconds: 120
    max_retries: 2
    max_concurrency: "auto"
    retry_backoff_seconds: 1.0
    max_tokens: 4096
```

<p align="right"><a href="#chinese"><strong>返回中文</strong></a></p>
