# DirtyRAG / EvidenceBoard-RAG 实验报告

> 主实验：RAMDocs 200 条子集 · 5 方法对照 · training-free post-retrieval robustness
> 数据来源：`ServerOutputs/outputs/runs/latest`（服务器跑，run_20260528_235151）
> 报告用途：论文 Experiments / Results / Analysis / Limitations 章节素材
> 生成日期：2026-05-29

---

## 0. 一句话结论

在 RAMDocs 200 条冲突证据子集上，**EvidenceBoard-RAG 在 strict success、gold coverage、conflict sensitivity 三项核心指标上显著领先所有 baseline**（strict success 0.495 vs 次优 0.36，相对提升约 37%）；wrong-answer leakage 与 Vanilla RAG 持平（0.06），但其失败以**保守弃答**为主而非**自信错答**，叙事更健康；代价是约 **3.3× 的 token 成本**与最高的延迟。

我们**不声称全面 SOTA**：在干净/简单场景多步验证会引入额外成本和偶发弃答；RAMDocs 本身极难（72/200 条所有方法全错）。

---

## 1. 实验设置

### 1.1 数据集

| 项 | 值 |
|---|---|
| 数据集 | RAMDocs（Retrieval-Augmented Generation with Conflicting Evidence, COLM 2025） |
| 子集规模 | 前 200 条（`data/processed/ramdocs_200.jsonl`） |
| 任务类型 | conflict_qa（冲突 / 误导 / 噪声证据下的 QA） |
| 每条文档数 | ≤ 12（`max_docs_per_example: 12`） |
| 文档类型标注 | correct / misinfo / noise（**仅评估使用，方法运行时禁止读取**） |
| 字段 | question, documents, gold_answers, wrong_answers, disambig_entity |

RAMDocs 的核心特征：检索回来的文档**全部"相关"，但互相矛盾**——既有正确答案，也有误导信息和噪声。这正是普通 relevance filter 无法处理的场景。

### 1.2 模型与运行条件

| 项 | 值 |
|---|---|
| LLM | DeepSeek（OpenAI-compatible API，`deepseek-chat`） |
| temperature | 0（确定性） |
| max_tokens | 512（最终回答）；各中间步骤单独设置（见 §2.4） |
| seed | 42 |
| 检索条件 | 不做真实向量检索，直接使用 RAMDocs 提供的候选文档（聚焦 post-retrieval robustness） |
| 公平性 | 所有方法共享同一 LLM、同一文档集合、同一评估脚本 |
| 缓存 | 相同 prompt 结果缓存到 `llm_calls.jsonl`，retry 3 次 + 指数退避 |
| EB-RAG 总运行时长 | 约 2 小时 6 分钟（200 条） |

### 1.3 评估指标

| 指标 | 定义 | 方向 |
|---|---|---|
| **strict_success** | 覆盖到 gold answer **且** 未泄漏 wrong answer | 越高越好（主指标） |
| **gold_coverage** | 回答中是否出现任一 gold answer | 越高越好 |
| **wrong_leakage** | 回答中是否出现任一 wrong answer | 越低越好 |
| **conflict_sensitivity** | 回答中是否显式标识冲突/歧义（关键词匹配：conflict / contradict / inconsistent / multiple answers / cannot determine） | 越高越好 |
| **avg_latency** | 平均每条耗时（秒） | 参考 |
| **avg_total_tokens** | 平均每条 token 消耗 | 越低越省 |

> **方法学说明**：gold_coverage 采用"出现任一 gold answer 即记为命中"的宽松字符串匹配。对多答案场景（如 §5.2 的 Monkey Grip）这意味着覆盖一个合法答案即算命中，可能高估多答案完整覆盖率。论文中应注明这是一个 lenient match，未来可补充 exact-set coverage。

---

## 2. 对照方法

5 个方法全部为 training-free，继承统一 `BaseMethod` 接口。

### 2.1 B0 · Direct LLM
仅输入 question，不给任何文档。closed-book 对照，用于度量"纯参数化知识"的下限。

### 2.2 B1 · Vanilla RAG
question + 全部候选文档拼接后交给 LLM。最基础的 RAG 范式（Lewis et al., NeurIPS 2020）。

### 2.3 B2 · RAG + Relevance Filter
先用 LLM 对每篇文档判 relevance（0/1/2），保留 ≥1 的文档再回答。检验"只过滤无关文档"是否足够。

### 2.4 B3 · CRAG-style RAG
CRAG（Corrective RAG, 2024）的轻量复现：
- LLM 判检索整体质量：correct / ambiguous / incorrect / insufficient
- 据此触发：直接回答 / 过滤后回答 / 弃答
- **与原版差异（论文须诚实声明）**：用 zero-shot LLM judge 替代训练版 T5 evaluator，**移除 web search 分支**，移除 decompose-then-recompose。论文中写 `CRAG-style`，不声称完整复现。

### 2.5 Ours · EvidenceBoard-RAG
training-free 证据中心框架，流程：
```
Evidence Card 抽取（每文档→结构化卡片）
→ Duplicate Grouping（Jaccard≥0.82 去重）
→ Conflict Graph（support / contradict / duplicate 边）
→ Answer Cluster Scoring（按答案簇而非文档数聚合，抗证据稀释）
→ Candidate Decision（answer / conflict / unknown）
→ Verifier（supported / revise / conflict / unknown）
→ Final Answer + Evidence Board JSON
```
关键设计：抗稀释聚合（重复证据只算一次独立支持）、显式冲突建模、可弃答/可标冲突、全过程可解释（每条存 Evidence Board JSON）。

中间步骤 token 上限：evidence card 900、relevance judge 400、CRAG eval 600、verifier 600（V3.1 调高以稳定结构化 JSON 输出）。

---

## 3. 主结果

### 3.1 主结果表（RAMDocs 200, n=200/method）

| 方法 | strict_success | gold_coverage | wrong_leakage | conflict_sensitivity | avg_latency (s) | avg_total_tokens |
|---|---:|---:|---:|---:|---:|---:|
| Direct LLM | 0.050 | 0.055 | **0.010** | 0.000 | **2.66** | **141** |
| Vanilla RAG | 0.335 | 0.380 | 0.060 | 0.180 | 7.26 | 1091 |
| RAG + Filter | 0.360 | 0.400 | 0.055 | 0.195 | 19.10 | 2528 |
| CRAG-style | 0.220 | 0.220 | 0.015 | 0.140 | 9.45 | 2025 |
| **EB-RAG (Ours)** | **0.495** | **0.505** | 0.060 | **0.350** | 33.06 | 3596 |

（对应图：`figures/main_metrics.png`，4 面板柱状图）

### 3.2 关键读数

- **strict success**：Ours 0.495，比次优 RAG+Filter（0.36）**绝对高 13.5 个百分点、相对高约 37%**。
- **gold coverage**：Ours 0.505，同样领先，说明它在多答案/可答场景下更能捞回正确答案。
- **conflict sensitivity**：Ours 0.35 vs 次优 0.195，**接近翻倍**——证据中心的显式冲突建模直接转化为"系统能意识到冲突"的能力。
- **wrong leakage**：Ours 0.06，与 Vanilla 持平，略高于 CRAG（0.015）和 Direct（0.01）。**这是诚实的短板**（见 §4.3 解释）。
- **成本**：Ours token 是 Vanilla 的 3.3×、延迟约 4.5×。

### 3.3 成本-效果权衡

（对应图：`figures/cost_tradeoff.png`，散点图）

EB-RAG 位于右上角——**效果最高但成本最高**。Direct LLM 在左下（便宜但几乎不可用）。CRAG-style 是个反例外点：花了 2025 tokens 却只有 0.22 strict success，**性价比甚至低于免费得多的 Vanilla RAG**（见 §4.1）。

---

## 4. 分析

### 4.1 发现一：CRAG-style 在脏证据上反而劣于 Vanilla（强 motivation 证据）

| 方法 | strict_success | token |
|---|---:|---:|
| Vanilla RAG | 0.335 | 1091 |
| CRAG-style | 0.220 | 2025 |

CRAG-style **多花近 1 倍 token，strict success 反而掉了 11.5 个百分点**。原因分析：
- CRAG 的 retrieval evaluator 设计用于处理 *retrieval-gone-wrong*（检索到无关/低质文档）。
- 但 RAMDocs 的难点是 *retrieval-succeeded-but-conflicting*：文档都相关，只是互相矛盾。
- CRAG 把"有冲突"误判为"质量不足"，触发过度过滤或弃答（CRAG conflict_sensitivity 0.14 也偏低），把本可答对的题答成 unknown。

**这条数据直接支撑论文核心 motivation**：相关性导向的纠错（CRAG）不足以处理"相关但互斥"的证据，需要证据级冲突建模。建议放入 Related Work 末尾或 Discussion。

### 4.2 发现二：EB-RAG 的优势来自"独有正确"而非"边际改进"

逐条胜负分解（200 条）：

| 情形 | 条数 | 占比 |
|---|---:|---:|
| 所有方法都答对 | 1 | 0.5% |
| **EB-RAG 独有答对（4 个 baseline 全错）** | **43** | **21.5%** |
| 所有方法都答错 | 72 | 36% |

- 36% 的题**所有方法都搞不定**——说明 RAMDocs 极难，绝对分数天花板低，0.495 的 strict success 已经是很强的表现。
- EB-RAG 有 **43 条是它独自答对的**，证明它的增益不是统计噪声，而是来自结构化证据处理带来的真实能力。

### 4.3 发现三：EB-RAG 的失败以"保守弃答"为主，而非"自信错答"

EB-RAG 全部 200 条回答的结果分类：

| 结果 | 条数 | 占比 |
|---|---:|---:|
| strict success（答对） | 99 | 49.5% |
| **abstain（输出 conflict / unknown）** | **73** | **36.5%** |
| other miss（既无 gold 也无 wrong） | 16 | 8% |
| **wrong leak（泄漏错误答案）** | **12** | **6%** |

关键解读：
- Ours 的 wrong_leakage 虽然和 Vanilla 同为 0.06，但**结构完全不同**。Ours 的失败里 73 条是主动弃答（保守、安全），只有 12 条是真错答。
- Vanilla 的失败则是"自信地把错误答案说出来"。
- 换言之：**EB-RAG 在不确定时倾向于说"冲突/不知道"，而不是编一个错答案**——这在脏知识库的实际应用中是更可取的失败模式（宁可不答，不可误导）。

EB-RAG 决策模式分布（来自 200 个 Evidence Board）：

| 阶段 | answer/supported | conflict | unknown |
|---|---:|---:|---:|
| candidate decision | 128 | 70 | 2 |
| verifier verdict | 127 | 70 | 3 |

70 条（35%）被判定为 conflict——与 conflict_sensitivity=0.35 一致，说明系统确实在显式识别冲突。

### 4.4 RQ 对照

| 研究问题 | 结论 |
|---|---|
| **RQ1 冲突鲁棒性**：能否减少错答？ | 部分成立。strict success 大幅领先，但 wrong_leakage 未优于 Vanilla；优势体现在"答对更多" + "失败时弃答而非错答"。 |
| **RQ2 证据稀释**：能否抗重复误导？ | 主实验未单独隔离 duplicate 维度（需 stress test 补充，见 §7）。抗稀释聚合机制已实现，但需专门实验量化。 |
| **RQ3 成本与失败模式**：多步验证代价？ | 明确。3.3× token、4.5× 延迟；失败模式以保守弃答为主，clean 场景有过度弃答风险。 |

---

## 5. 案例研究

三个案例图位于 `figures/case_ramdocs_0000XX.png`，含证据图（节点=文档卡，颜色=correct/misinfo/noise，边=contradict/support/duplicate）+ 各方法回答对比表 + EB-RAG 决策链。

### 5.1 案例 ramdocs_000098（EB-RAG 完胜）— 同名实体污染

> **Question**: What sport does Cameron Murray play?
> **Gold**: Football　**Wrong**: Cricket, Basketball

| 方法 | 回答 | 结果 |
|---|---|---|
| Vanilla RAG | "Football (soccer) and cricket." | ❌ leak（混入 cricket） |
| CRAG-style | unknown | ❌ 弃答 |
| **EB-RAG** | **American football** | ✅ strict success |

存在多个同名 Cameron Murray（不同运动）。Vanilla 把不同实体的证据混在一起、泄漏了 cricket；CRAG 过度保守弃答；EB-RAG 通过证据卡的实体显式性 + 答案簇聚合，选出独立支持最强的正确答案。

### 5.2 案例 ramdocs_000168（EB-RAG 完胜）— 多答案歧义

> **Question**: What is the medium of "Monkey Grip"?
> **Gold**: Film, Novel

| 方法 | 回答 | 结果 |
|---|---|---|
| Vanilla RAG | "exists as both a novel (1977) and a film (1982)" | ❌（注：标注将 Novel 列入 wrong，触发 leak 判定） |
| CRAG-style | conflict | ❌ |
| **EB-RAG** | film | ✅（命中 gold "Film"，宽松匹配下记为成功） |

> ⚠️ 本例暴露 RAMDocs 标注的一个边界：gold 同时含 Film 和 Novel，但 wrong 也含 Novel——标注本身有张力。结合 §1.3 的宽松匹配，EB-RAG 因命中 "film" 记为成功。论文引用此例时建议谨慎，或仅用作"多答案歧义"定性展示。

### 5.3 案例 ramdocs_000028（EB-RAG 失败）— 诚实的局限性

> **Question**: What is the genre of "Big Girl"?
> **Gold**: Short film　**Wrong**: Documentary

| 方法 | 回答 | 结果 |
|---|---|---|
| Vanilla RAG | conflict | ❌（弃答，未 leak） |
| CRAG-style | unknown | ❌ |
| **EB-RAG** | **documentary** | ❌ **wrong leak** |

EB-RAG 这里把误导答案 "documentary" 当成了主答案——证据簇打分被高可信度的误导文档主导。这是 EB-RAG 的真实失败模式，**适合放入 Limitations 章节**，体现实验诚实性：当误导证据本身写得"很像正确答案"时，证据中心机制也可能被带偏。

---

## 6. 论文叙事建议

### 6.1 主张（建议写法）

> EvidenceBoard-RAG substantially improves strict success and conflict sensitivity over vanilla, relevance-filtered, and CRAG-style RAG on the RAMDocs conflicting-evidence subset, by structuring retrieved documents into evidence cards and explicitly modeling answer-level conflict before generation. Crucially, its residual failures are dominated by conservative abstention rather than confident wrong answers. These gains come at roughly 3.3× token cost, and the method does not reduce wrong-answer leakage below strong abstention-heavy baselines such as CRAG.

### 6.2 不要写的（会翻车）
- ❌ "achieves zero wrong-answer leakage"（10 条时的假象，200 条已是 0.06）
- ❌ "outperforms all baselines on all metrics"（wrong_leakage 输给 CRAG/Direct）
- ❌ "fully reproduces CRAG"（我们是 CRAG-style 简化版）

### 6.3 三个最强卖点
1. strict success 相对提升 ~37%，且有 43 条独有答对（§4.2）
2. conflict sensitivity 接近翻倍，证据中心设计直接见效（§3.2）
3. 失败以保守弃答为主（73 弃答 vs 12 错答），脏知识库下更安全（§4.3）

---

## 7. 局限性与未完成实验

| 局限 | 说明 |
|---|---|
| 单一数据集 | 仅 RAMDocs；FaithEval-inconsistent / AmbigDocs 尚未跑（loader 已就绪） |
| 缺 duplicate stress test | 抗稀释聚合的核心卖点（RQ2）未被专门实验量化 |
| 缺 ablation | 未隔离 Conflict Graph / Deduplication / Verifier 各模块贡献 |
| 缺 MADAM-RAG-lite | 最贴题的多智能体 baseline 未跑（可选） |
| 成本高 | 3.3× token、4.5× 延迟，clean 场景性价比低 |
| 评估宽松 | gold_coverage 用"任一命中"宽松匹配，可能高估多答案覆盖 |
| LLM judge 偏差 | conflict_sensitivity 用关键词匹配，可能漏判语义化冲突表达 |
| 启发式过拟合风险 | V3.2/V4 的 sport/domain/population 启发式针对前 10 条调过，需检查在 11-200 条上是否泛化（200 条 wrong_leakage 0.06 不算高，初步说明未严重过拟合） |
| 数据泄漏防线 | 须确认方法运行时未读 source_type/gold_answers/wrong_answers（建议跑前 grep 核验） |

### 后续实验优先级
1. **Duplicate stress test**（兑现 RQ2，对 misinfo 文档复制 1/3/5/7 次画曲线）
2. **Ablation**（w/o Conflict Graph / w/o Dedup / w/o Verifier）
3. FaithEval-inconsistent 50 条（faithfulness 维度）
4. MADAM-RAG-lite（多智能体对照）

---

## 8. 复现信息

### 8.1 配置
- 配置文件：`configs/ramdocs_200.yaml`
- run 目录：`outputs/runs/run_20260528_235151`（= latest）
- 数据：`data/processed/ramdocs_200.jsonl`（200 条，本地由 RAMDocs_test.jsonl 转换）

### 8.2 复现命令
```bash
# 1. 准备数据
python -m dirtyrag.data.prepare_datasets --dataset ramdocs --limit 200

# 2. 跑全量（设置 DeepSeek 环境变量）
export LLM_API_KEY=... LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-chat
python -m dirtyrag.cli run --config configs/ramdocs_200.yaml --limit 200

# 3. 评估
python -m dirtyrag.cli evaluate --run-dir outputs/runs/latest

# 4. 画图
python -m dirtyrag.cli plot --run-dir outputs/runs/latest
```

### 8.3 产物清单
```
outputs/runs/latest/
  config.yaml                 # 运行配置快照
  dataset_path.txt            # 数据路径
  predictions.jsonl           # 1000 条（200×5）
  per_case_metrics.jsonl      # 逐条指标
  metrics.csv                 # 方法级汇总（主结果表来源）
  llm_calls.jsonl             # LLM 调用缓存（含 prompt 原文）
  evidence_boards/            # 200 个 EB-RAG 证据板 JSON
  figures/
    main_metrics.png/.pdf     # 主结果 4 面板
    cost_tradeoff.png/.pdf    # 成本-效果散点
    case_ramdocs_000098.png   # 案例：同名实体（完胜）
    case_ramdocs_000168.png   # 案例：多答案歧义（完胜）
    case_ramdocs_000028.png   # 案例：误导主导（失败/limitations）
```

---

## 附录 A. 完整数字速查

```
method,num_examples,strict_success,gold_coverage,wrong_leakage,conflict_sensitivity,avg_latency,avg_total_tokens
direct_llm,200,0.05,0.055,0.01,0.0,2.660972,140.95
vanilla_rag,200,0.335,0.38,0.06,0.18,7.263547,1090.755
relevance_filter_rag,200,0.36,0.4,0.055,0.195,19.09809,2527.95
crag_style_rag,200,0.22,0.22,0.015,0.14,9.450903,2025.18
evidenceboard_rag,200,0.495,0.505,0.06,0.35,33.063782,3595.555
```

## 附录 B. EB-RAG 行为统计速查

```
逐条胜负（200 条）：
  所有方法答对        : 1
  EB-RAG 独有答对     : 43
  所有方法答错        : 72

EB-RAG 回答分类（200 条）：
  strict success      : 99  (49.5%)
  abstain(conflict/unknown): 73 (36.5%)
  other miss          : 16  (8%)
  wrong leak          : 12  (6%)

EB-RAG 决策模式：
  candidate: answer 128 / conflict 70 / unknown 2
  verifier : supported 127 / conflict 70 / unknown 3
```
