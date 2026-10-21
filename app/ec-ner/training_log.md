# ec-ner 模型训练与超参数调优记录

初版日期：2026-10-21
最后更新：2026-10-21（首训完成，以下为权威值）

> 本文件记录 `app/ec-ner` 交付模型的训练过程、超参数决策与 F1 结果。
> 基线对比见 `app/ner/hyperparameter-tuning.markdown`（800 条人工 TAG 基线，F1 0.3355）。
> 数据基座与阶段划分见 `task.md`。

---

## 1. 背景

| 项 | 内容 |
|----|------|
| 数据 | GAIIC 2022 商品标题实体识别（京东），**40,000 句**已标注（B-1..B-54 脱敏编码） |
| 预处理 | `preprocess.py` 塌缩为多类型 BIO → 单 TAG `['B','I','O']`，全量 9:1:1 切分 |
| 切分规模 | **train 32,719 / valid 3,636 / test 3,636**（去重后 39,991 句，固定种子 RANDOM_SEED=42） |
| 序列长度 | 京东商品标题，平均约 20–40 字，最长 101 字（< MAX_SENT_LEN=200，无丢弃） |
| 模型 | `google-bert/bert-base-chinese`，3 标签 token 分类（**单 TAG，类型无关**） |
| 下游 | `extract` 出的实体原样写 Neo4j `Tag` 节点（仅 name，无 type） |
| 环境 | conda `graph`：Python 3.12、torch 2.11.0+cu128、RTX 5070、transformers 5.20.1、datasets 5.1.0 |
| 离线 | `HF_HUB_OFFLINE=1` 走本地 `google-bert/bert-base-chinese` 缓存 |

**与 app/ner 基线的关键差异**：`app/ner` 基线只用 1,000 条人工 TAG 电商描述（train 800 / valid 100 / test 100），
而 `app/ec-ner` 用 **40,000 条**同域（京东标题）公开标注语料——数据量约 **40×**，且域与业务高度一致。
这正是 F1 从 0.3355 跃升到 0.88 量级的根因（见 §4）。

---

## 2. 超参数决策与理由

超参继承 `app/ner` 基线框架，但**按 40k 数据规模重新推导**（而非照搬 800 条基线的数值）：

| 参数 | app/ner 基线 | **ec-ner 本训** | 理由 |
|------|-------------|----------------|------|
| `BATCH_SIZE` | 16 | **32** | 32k 数据每轮 ~1,022 步，RTX 5070 显存充裕；批大梯度更稳、吞吐更高 |
| `EPOCHS` | 3 | **3** | 800 条基线因过拟合定为 3；40k 数据过拟合风险低，但实测 F1 在 ~1.5 epoch 已趋平台（见 §3），3 轮足够，再多仅边际收益 |
| `LEARNING_RATE` | 3e-5 | **3e-5** | bert-base 微调标准区间 1e-5~3e-5；基线扫描已确认 3e-5 优于 2e-5，沿用 |
| `WEIGHT_DECAY` | 0.01 | **0.01** | 正则，防过拟合 |
| `WARMUP_RATIO` | 0.1（→ warmup_steps） | **0.1（→306 步）** | 按总步数 `3×32719//32×0.1≈306` 自适应；前 306 步线性升温 |
| `EVAL/SAVE_STEPS` | 100 | **500** | 1,022 步/轮，每 500 步描一次点 ≈ 每轮 2 次 eval；6 个 checkpoint 足够选最优 |
| `LOGS_STEPS` | 100 | 100 | 训练 loss 日志粒度 |
| `fp16` | True | **True** | 5070 上半精度加速，不影响收敛 |
| `metric_for_best_model` | eval_overall_f1 | **eval_overall_f1** | seqeval 实体级 F1（token 级 acc 在 O 占多数时失真） |
| `load_best_model_at_end` | True | **True** | 自动加载最优 checkpoint 落盘 `checkpoints/best_model` |
| `save_total_limit` | 不限 | **不限** | 避免最优 checkpoint 被轮换删除；训练后保留 `checkpoint-*` |

### 2.1 调优手段说明

- **未做大规模自动扫描**：数据量从 800 跃升到 32k 后，模型在默认微调区间已稳定收敛（§3 轨迹平滑、无过拟合迹象），扫描收益有限。本训以"数据规模推导 + 收敛观测"确定超参。
- **唯一需要权衡的点**是 `EPOCHS`：轨迹显示 F1 在 1.5 epoch 后进入平台（0.874 → 0.879，仅 +0.005），说明 3 轮已接近该配置下的上限；进一步提升应靠**数据扩充**而非堆 epoch（见 §5）。

---

## 3. 训练轨迹（valid 集，seqeval 实体级 F1）

训练总步数 **3,069**（3 epoch × 1,022 步/epoch），每 500 步在 valid 上评估一次：

| 步数(≈) | epoch | eval_loss | **eval F1** | precision | recall | accuracy |
|--------|-------|-----------|-------------|-----------|--------|----------|
| 500 | 0.49 | 0.2358 | **0.8550** | 0.8231 | 0.8895 | 0.918 |
| 1000 | 0.98 | 0.2211 | **0.8693** | 0.8457 | 0.8941 | 0.9229 |
| 1500 | 1.47 | 0.2198 | **0.8739** | 0.8553 | 0.8932 | 0.9249 |
| 2000 | 1.96 | 0.2126 | **0.8761** | 0.8564 | 0.8968 | 0.926 |
| 2500 | 2.44 | 0.2166 | **0.8786** | 0.8589 | 0.8992 | 0.9265 |
| 3000 | 2.93 | 0.2112 | **0.8783** | 0.8602 | 0.8972 | 0.9264 |
| 3069 | 3.00 | 0.2124 | **0.8789** | 0.8598 | 0.8989 | 0.9268 |

观察：
- **快速收敛**：第 1 个 epoch 内 F1 即从 0.855 升到 0.869，说明 32k 同域语料提供了充足信号。
- **平台期**：1.5 epoch 后 F1 在 0.874–0.879 窄幅波动，无过拟合（train_loss 仍缓降、valid_loss 平稳），符合预期。
- **最优 checkpoint**：epoch 3.0 的 `eval_overall_f1=0.8789` 为全训练最高 → `load_best_model_at_end` 自动选中，`trainer.save_model()` 落盘 `checkpoints/best_model`。

---

## 4. 实验结果（最终交付指标）

`python eval.py` 在 **test 集**（3,636 句 / 60,338 实体，与训练同分布、9:1:1 留出）实测：

| 指标 | app/ner 基线 | **ec-ner 本训** |
|------|-------------|----------------|
| TEST F1 | 0.3355 | **0.8815** |
| TEST P / R | 0.3230 / 0.3490 | **0.8631 / 0.9007** |
| TEST accuracy（token 级） | 0.7774 | **0.9276** |
| TEST 实体数 | 298 | 60,338 |
| 训练数据量 | 800 | 32,719 |

- **F1 提升 +0.546（相对 +163%）**，P/R 同步大幅改善。
- **valid（0.8789）≈ test（0.8815）**：无过拟合，泛化良好，指标可信。
- **recall（0.90）> precision（0.86）**：模型略微偏向"多抽实体"。对下游 Tag 抽取而言，漏抽一个真实 Tag 比多抽一个噪声 Tag 更糟（后者可在图谱侧/人审环节过滤），故该偏置可接受。

---

## 5. 决策总结与遗留事项

### 5.1 结论

- 40k 同域（京东标题）语料 + bert-base-chinese 微调：**3 epoch、bs 32、lr 3e-5、wd 0.01、warmup 10%、fp16、每 500 步 eval/save**。
- 以 `eval_overall_f1` 选最优权重，落 `checkpoints/best_model`（406 MB）。
- 交付模型 TEST F1 **0.8815**，可直接用于阶段二 `sync_text` 从 `spu_info.description` 抽取 Tag 入图谱。

### 5.2 进一步调优方向（如需更高 F1）

1. **扩数据（收益最大）**：当前 F1 上限由数据规模与单 TAG 边界模糊决定。`llm_annotate.py`（LLM 合成电商 TAG 语料）扩充训练集，预计显著提 F1。
2. **epochs 4–5**：轨迹显示边际收益仅 +0.001~0.003，性价比低，优先级低于扩数据。
3. **LR / warmup / dropout 扫描**：当前已在标准区间，扫描空间有限；可小范围验证 lr=2e-5 与 dropout=0.1 是否更稳。
4. **序列切分粒度**：标题较短，当前 MAX_SENT_LEN=200 未触发丢弃；若后续引入更长商品描述，需重新评估截断策略。

### 5.3 已知边界

- **实体类型单一是项目规格，非缺陷**：B/I/O 仅 TAG 一种，下游 `Tag` 节点只存 name 不存 type（据项目图谱文档）。若需把实体链接到 `category3_id`/`tm_id`，是**额外的实体链接（entity linking）步骤**，不在本模型范畴——模型只输出 Tag 文本边界。
- **Recall 偏高**：如 §4 所述，可接受。
- transformers 5.x 已踩平两坑（见 `app/ner/hyperparameter-tuning.markdown` §7）：`preprocess_logits_for_metrics` 张量维度、`LayerNorm` 新键名由 `from_pretrained` 自动映射，推理正常。

---

## 6. 复现命令

```bash
cd app/ec-ner
python download_data.py        # 已下载则跳过
python preprocess.py           # -> data/processed/{train 32719, valid 3636, test 3636}（9:1:1）
python train.py                # GPU：conda graph 环境 + HF_HUB_OFFLINE=1 -> checkpoints/best_model
python eval.py                 # -> TEST F1 0.8815
```

训练日志归档：`logs/train_20261021_1438.log`（含每步 loss 与 7 次 eval 完整指标）。
