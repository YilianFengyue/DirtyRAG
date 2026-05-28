# DirtyRAG 选题调研与论文方案 v2：公开数据集 + 证据中心鲁棒 RAG 方案

> 选题方向：**面向“脏知识库”的 RAG 鲁棒性增强 + 评测基准（冲突 / 过期 / 重复）**  
> 推荐英文题目：**EvidenceBoard-RAG: A Training-Free Evidence-Centric Framework for Robust Retrieval-Augmented Generation over Dirty Knowledge Bases**  
> 备选英文题目：**DirtyRAG-Guard: Conflict-Aware Verification for Robust Retrieval-Augmented Generation under Conflicting, Noisy, and Duplicated Evidence**  
> 版本说明：这是在“数据集自产自销、多智能体过水、预期结果过度完美”等评审意见基础上重新迭代后的方案。核心改动是：**主实验采用公开数据集子集，方法叙事从“多智能体炫技”改成“证据中心的冲突建模与验证”，结果叙事改成可证伪、可失败、可解释。**

---

## 0. 修订后的一句话结论

这个题目仍然可做，而且更应该做，但原方案需要收敛：

> 不要把它包装成“我自造 20 个 case，然后我的多 Agent 必然赢”。  
> 应该把它包装成：**基于公开冲突证据数据集，设计一个 training-free 的证据中心 RAG 验证框架，研究普通 RAG、CRAG-style RAG、多智能体 debate 在冲突/噪声/重复证据下各自的失效模式。**

修订后的论文主张不是“我们全面 SOTA”，而是：

> **在公开 DirtyKB-style 子集上，我们的方法能更稳定地识别证据冲突、减少被重复/错误证据带偏，并在不可判定时触发澄清或弃答；同时我们诚实报告 clean 场景下多步验证可能带来的额外成本与错误。**

这比“全面超过所有 baseline”更可信，也更像认真做过实验。

---

## 1. 原方案的主要隐患与修订策略

### 1.1 隐患一：完全自构造数据集，实验说服力不足

原方案里提出 20 个 case 自己构造，这确实有问题。因为如果问题、脏文档、ground truth 都是自己写的，那么方法设计很容易和数据分布绑定，最后变成“出题人自己答题”。

#### 修订策略

主实验改为使用公开数据集子集：

1. **RAMDocs**：主数据集，直接面向 ambiguity / misinformation / noise 的 conflicting evidence RAG 场景。
2. **FaithEval-inconsistent**：补充数据集，重点看上下文不一致时模型能不能忠实于 evidence。
3. **AmbigDocs**：补充模糊实体场景，检验模型是否能区分同名实体并给出多答案或澄清。
4. **可选：CONFACT / RAGTruth / RARE**：作为 related work 或少量扩展实验，不作为一天内主线依赖。

自构造数据只保留为：

- demo 展示；
- duplicate stress test 的可控扰动；
- 过期页面/时间敏感问题的少量补充分析。

论文写法：

> Our main experiments are conducted on public benchmarks, including RAMDocs, FaithEval-inconsistent, and AmbigDocs. We only use synthetic perturbations for controlled duplicate-stress analysis, and we report them separately from the main benchmark results.

---

### 1.2 隐患二：“多智能体”容易被质疑只是多个 prompt

原方案把 Evidence Profiler、Conflict Detector、Answer Agent、Verifier 都叫 agent，严格讲容易被问：这不就是同一个 LLM 的不同 prompt 吗？真正 multi-agent 往往有独立状态、信息交换、辩论、协作或竞争。

#### 修订策略

主方法叙事从“多智能体框架”改为更稳的：

> **Evidence-centric agentic workflow / role-specialized verification pipeline**

也就是承认它主要是一个“证据中心的 agentic pipeline”，而不是强行吹成复杂 multi-agent system。

如果课程需要多 Agent 元素，就实现一个**轻量但不水的通信机制**：

- 每个 retrieved document 由一个 **Document Agent** 生成结构化 Evidence Card；
- 所有 Evidence Card 写入共享记忆 **Evidence Board**；
- **Conflict Adjudicator** 读取 Evidence Board，生成 Conflict Edges；
- **Answer Composer** 基于冲突图生成答案；
- **Verifier/Critic** 检查答案是否覆盖正确证据、是否被 misinfo 支配、是否应拒答/澄清。

关键不是“agent 名字多”，而是引入明确的通信对象：

```json
{
  "evidence_board": [
    {
      "doc_id": "D1",
      "claim": "Michael Jordan attended the University of North Carolina.",
      "answer_candidate": "University of North Carolina",
      "stance": "supports_answer",
      "doc_type": "correct",
      "confidence": 0.82
    },
    {
      "doc_id": "D2",
      "claim": "Michael Jordan attended the University of Oxford.",
      "answer_candidate": "University of Oxford",
      "stance": "conflicting_answer",
      "doc_type": "misinfo",
      "confidence": 0.76
    }
  ],
  "conflict_edges": [
    {
      "source": "D1",
      "target": "D2",
      "relation": "contradict",
      "reason": "Both documents answer the same education question with different universities."
    }
  ]
}
```

这样即使用的是同一个 LLM API，也不是纯“多 prompt 糊弄”，因为系统确实有：

- 中间状态；
- 共享记忆；
- 消息格式；
- 证据图；
- 纠错与弃答策略。

---

### 1.3 隐患三：预期结果叙事太完美

原方案默认：Vanilla RAG 会输，CRAG 会半输，我们的方法全面赢。这个叙事太“论文味”，但实际跑起来不一定。

可能出现的真实情况：

- clean 场景下，Vanilla RAG 反而更好，因为多步 pipeline 会引入 judge 错误；
- 简单噪声场景下，CRAG-style relevance filter 已经够用；
- 多 Agent debate 会因为 token 成本和随机性变差；
- LLM judge 本身会错判冲突；
- 当 misinfo 数量远多于 correct evidence 时，我们的方法也可能被“证据多数派”干扰。

#### 修订策略

论文结果叙事改成“失效模式研究 + 局部优势”，而不是“全指标第一”。

推荐主张：

> We do not claim universal superiority. Instead, we show that evidence-centric conflict modeling improves robustness in conflict-heavy and duplicate-heavy settings, while introducing extra cost and occasional over-refusal in clean settings.

中文：

> 我们不声称所有场景都更强；我们展示它在冲突证据和重复干扰强的场景中更稳，但也承认它在干净场景下有额外成本和过度拒答风险。

---

## 2. 公开数据集选择：不要“自产自销”

### 2.1 主数据集一：RAMDocs

**论文/项目：** *Retrieval-Augmented Generation with Conflicting Evidence*，COLM 2025。  
**数据集：** RAMDocs, Retrieval with Ambiguity and Misinformation in Documents。  
**为什么适合：** 这是最贴题的数据集，专门模拟 user query 面对 ambiguity、misinformation、noise 的复杂冲突证据场景。

RAMDocs GitHub 仓库公开提供 `RAMDocs_test.jsonl`。仓库 README 明确写到：RAMDocs 模拟真实复杂场景中的 conflicting evidence，包括 ambiguity、misinformation 和 noise；每个样本包含 question、documents、disambig_entity、gold_answers、wrong_answers 等字段。文档字段里还标注了 document type：`correct`、`misinfo`、`noise`。

#### RAMDocs 字段对本课题的价值

| 字段 | 用途 |
|---|---|
| question | 用户问题 |
| documents | 模拟检索得到的多文档上下文 |
| document.text | 具体证据文本 |
| document.type = correct | 正确信息 |
| document.type = misinfo | 错误/误导信息 |
| document.type = noise | 无关噪声 |
| document.answer | 单个文档支持的答案 |
| gold_answers | 正确答案集合 |
| wrong_answers | 错误答案集合 |
| disambig_entity | 模糊实体集合 |

#### 实验用法

取 RAMDocs 20–50 个 case 作为主实验：

- 不自己写 question；
- 不自己写 gold answer；
- 不自己写 misinfo/noise；
- 只实现不同 RAG pipeline 在这些样本上的回答与评估。

#### 适合评估的指标

- Exact Match / F1；
- correct answer coverage；
- wrong answer leakage；
- noise suppression；
- conflict detection rate；
- ambiguity handling rate。

参考：

- HanNight/RAMDocs GitHub: https://github.com/HanNight/RAMDocs
- OpenReview: https://openreview.net/forum?id=z1MHB2m3V9
- arXiv: https://arxiv.org/abs/2504.13079

---

### 2.2 主数据集二：FaithEval-inconsistent

**论文/项目：** *FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows"*。  
**为什么适合：** FaithEval 专门评估 LLM 在给定上下文中是否保持 faithful。它包含三类任务：unanswerable、inconsistent、counterfactual。对于本题最有用的是 **inconsistent subset**。

FaithEval GitHub README 写到：FaithEval 是一个 contextual faithfulness benchmark，包含 unanswerable、inconsistent 和 counterfactual 三类场景；这些任务模拟真实检索中可能出现的 incomplete、contradictory 或 fabricated information。仓库还给出了 HuggingFace 加载方式：

```python
from datasets import load_dataset
inconsistent_dataset = load_dataset("Salesforce/FaithEval-inconsistent-v1.0", split="test")
```

#### 实验用法

取 FaithEval-inconsistent 20–30 个 case 作为补充实验，检验：

- 模型是否会在上下文冲突时编造一个“看似合理”的答案；
- 是否能识别上下文不一致；
- 是否能输出“evidence is inconsistent / cannot determine”；
- 是否能保持 faithfulness。

参考：

- GitHub: https://github.com/SalesforceAIResearch/FaithEval
- arXiv: https://arxiv.org/abs/2410.03727

---

### 2.3 补充数据集：AmbigDocs

**论文/项目：** *AmbigDocs: Reasoning across Documents on Different Entities under the Same Name*，COLM 2024。  
**为什么适合：** AmbigDocs 关注同名实体带来的多文档歧义。例如 “Michael Jordan” 可能指篮球运动员，也可能指其他同名人物。普通 RAG 容易把不同实体的信息混在一起。

OpenReview 摘要写到：AmbigDocs 利用 Wikipedia disambiguation pages 找到同名实体对应的文档，生成包含 ambiguous name 的问题和对应多答案集合；研究发现 SOTA 模型常常给出模糊答案或错误合并不同实体的信息。

#### 实验用法

AmbigDocs 不一定作为主实验，但可以作为“ambiguity handling”子实验：

- 如果一个问题有多个合法答案，系统不能只输出一个；
- 如果证据分属不同实体，系统应分组回答；
- 如果无法判定用户指代哪个实体，应触发 clarification。

参考：

- OpenReview: https://openreview.net/forum?id=mkYCfO822n
- arXiv: https://arxiv.org/abs/2404.12447

---

### 2.4 可选数据集：CONFACT

**论文/项目：** *Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs*，2025。  
**为什么可能有用：** CONFACT 是面向 fact-checking 的 conflicting evidence 数据集，强调不同来源可信度导致的冲突。

它和本课题关系较近，但比 RAMDocs 更偏 fact-checking，而不是通用 RAG QA。若一天内时间有限，不建议主线使用，可作为 related work 或扩展实验。

参考：

- arXiv: https://arxiv.org/abs/2505.17762
- GitHub: https://github.com/zoeyyes/CONFACT

---

### 2.5 可选参考：RARE / RAGTruth / RAGChecker

这三个更适合作为论文动机与评估参考，不一定作为主实验数据。

| 名称 | 作用 | 是否主实验 |
|---|---|---|
| RARE | 检索感知鲁棒性评估，覆盖 query/document perturbation、外部/内部知识冲突、time-sensitive corpus | 可引用，是否用数据看公开可获取性与时间 |
| RAGTruth | RAG 幻觉语料，说明 RAG 仍可能生成 unsupported response | 作为动机/related work |
| RAGChecker | 细粒度诊断 retrieval 和 generation 的评估框架 | 借鉴指标设计 |

---

## 3. 修订后的问题定义

### 3.1 任务定义

给定一个用户问题 `q` 和一组检索文档 `D = {d1, d2, ..., dk}`，其中可能包含：

- 正确证据 correct evidence；
- 错误证据 misinformation；
- 无关噪声 noise；
- 多个合法实体/答案 ambiguity；
- 重复证据 duplicated evidence；
- 过期证据 outdated evidence。

系统需要生成答案 `a`，同时满足：

1. 尽可能覆盖正确答案；
2. 尽可能避免输出 wrong answer；
3. 遇到 ambiguity 时保留多个合法答案或要求澄清；
4. 遇到 conflict 无法解决时允许 abstain；
5. 不被重复错误证据带偏；
6. 能解释答案来自哪些证据。

### 3.2 研究问题

建议论文里写 3 个 research questions：

#### RQ1: Conflict Robustness

> Compared with vanilla RAG and CRAG-style filtering, can evidence-centric verification reduce wrong-answer leakage under conflicting evidence?

#### RQ2: Evidence Dilution

> When misinformation or duplicated evidence increases, can the system avoid being dominated by repeated but wrong evidence?

#### RQ3: Cost and Failure Modes

> What extra cost and new errors are introduced by multi-step verification, especially on clean or simple cases?

这三个问题比“我们全面刷爆 baseline”更稳。

---

## 4. 修订后的方法：EvidenceBoard-RAG

### 4.1 方法定位

不要主打“多智能体 SOTA”，而是主打：

> **training-free evidence-centric RAG verification framework**

中文：

> 一个不训练大模型的、以证据为中心的 RAG 鲁棒验证框架。

核心不是模型参数创新，而是系统机制创新：

1. 把文档拆成结构化 Evidence Cards；
2. 构建 Evidence Board 作为共享记忆；
3. 显式建模证据之间的 support / contradict / duplicate / noise 关系；
4. 生成答案前先做 conflict-aware synthesis；
5. 生成答案后再用 verifier 检查 wrong answer leakage 和 unsupported claims。

---

### 4.2 整体流程

```text
Question q
   ↓
Retriever / Provided Context Loader
   ↓
Document Agents / Evidence Card Builder
   ↓
Evidence Board Memory
   ↓
Conflict Graph Builder
   ↓
Conflict-Aware Answer Composer
   ↓
Verifier / Critic
   ↓
Final Answer / Abstention / Clarification
```

如果使用 RAMDocs，documents 已经给好，可以跳过真实向量检索，直接视为 retrieved documents。这样实验更聚焦于“retrieved context 有冲突后怎么处理”，而不是被检索器质量拖偏。

---

### 4.3 Evidence Card

每篇文档转成一张证据卡：

```json
{
  "doc_id": "D3",
  "claim": "The person named Michael Jordan attended the University of North Carolina.",
  "answer_candidate": "University of North Carolina",
  "entity_or_scope": "Michael Jordan, basketball player",
  "stance": "support",
  "evidence_type": "correct | misinfo | noise | unknown",
  "relevance_score": 0.91,
  "confidence": 0.84,
  "source_time": null,
  "is_duplicate_of": null
}
```

注意：实验时不能把 RAMDocs 的 `type=correct/misinfo/noise` 直接喂给方法，否则是作弊。这个字段只能用于评估。方法只能看到 question 和 document text。

---

### 4.4 Evidence Board：共享记忆

Evidence Board 是方法的关键创新之一。它是所有 agent / module 的共享中间状态，不只是 prompt 调用。

```json
{
  "question": "...",
  "cards": [...],
  "conflict_edges": [...],
  "duplicate_groups": [...],
  "answer_clusters": [...],
  "decision": {
    "mode": "answer | multi_answer | clarify | abstain",
    "reason": "..."
  }
}
```

这可以对应课程要求里的“记忆存储机制”。

---

### 4.5 Conflict Graph

对 Evidence Cards 做两类关系判断：

1. **Answer-level conflict**：不同文档支持不同答案；
2. **Claim-level conflict**：同一事实槽位给出互斥值。

关系类型：

| Relation | 含义 |
|---|---|
| support | 两个文档支持同一答案或相容事实 |
| contradict | 两个文档对同一问题给出互斥答案 |
| duplicate | 两个文档基本重复 |
| noise | 文档与问题无关 |
| ambiguous_scope | 文档可能属于不同实体或不同语境 |

---

### 4.6 Anti-Dilution Aggregation：防证据稀释聚合

这是修订后真正可以称为“创新点”的地方。

普通 RAG 的问题：

> 如果错误答案被重复了 5 次，模型可能以为它更可信。

CRAG-style 过滤的问题：

> 错误证据也可能高度相关，因此 relevance filter 不会删掉它。

EvidenceBoard-RAG 的处理：

1. 对重复文档聚类，降低重复票数权重；
2. 按 answer_candidate 聚类，而不是直接按文档数量投票；
3. 当多个 answer clusters 都有强证据时，不强行选一个；
4. 如果是 ambiguity，则输出多答案或请求澄清；
5. 如果是 misinformation conflict，则输出有证据支持的答案并标注冲突，或 abstain。

示意打分：

```text
cluster_score(answer_i) = unique_support_i - lambda * conflict_i - gamma * noise_i
```

这里不需要真的做复杂数学模型，也可以用规则实现：

- duplicate 只算一次；
- 与多个文档冲突的 candidate 降权；
- 没有足够支持时 abstain；
- 多个 disambiguated entity 同时存在时 multi-answer。

---

### 4.7 Verifier / Critic

Verifier 不负责生成新答案，而是检查：

1. final answer 是否包含 wrong answer；
2. final answer 是否漏掉多个合法答案；
3. answer 中每个 claim 是否能被 Evidence Board 支持；
4. 是否应该澄清而不是强答；
5. 是否被重复噪声主导。

输出格式：

```json
{
  "supported": true,
  "wrong_answer_leakage": false,
  "missing_gold_like_answer": false,
  "needs_clarification": false,
  "revision_required": false,
  "reason": "The answer is supported by D1 and D4; D2 is conflicting but appears to refer to a different entity."
}
```

---

## 5. 为什么这个设计“有可能”超过 baseline？

注意，这里只能说“有可能在特定脏知识场景中更稳”，不能说必然全面超过。

### 5.1 相对 Vanilla RAG 的优势

Vanilla RAG 的流程是：

```text
retrieve top-k → concatenate docs → generate
```

它没有显式区分：

- correct vs misinfo；
- ambiguity vs contradiction；
- duplicate majority vs independent evidence；
- answerable vs unanswerable。

因此在 RAMDocs / FaithEval-inconsistent 这类数据上，Vanilla RAG 容易把错误信息混入答案。

EvidenceBoard-RAG 的改进点是：

> 在生成答案前，把证据先结构化并进行冲突显式建模，减少 LLM 直接在混乱上下文里“自由发挥”。

### 5.2 相对 RAG + Reranker 的优势

Reranker 主要判断“相关性”。但 conflict evidence 的难点在于：

> 错误文档通常也是高度相关的。

例如问题问“某人毕业于哪里”，正确文档和错误文档都在讲这个人和学校，所以 relevance score 都很高。Reranker 可以删掉无关噪声，但删不掉“相关但错误/冲突”的证据。

EvidenceBoard-RAG 多了：

- answer_candidate extraction；
- answer cluster；
- conflict edge；
- duplicate grouping；
- abstention / clarification。

所以它理论上更适合处理“相关但互斥”的文档。

### 5.3 相对 CRAG-style RAG 的优势

CRAG 的核心是 retrieval evaluator，判断检索结果整体质量，然后触发不同检索动作。它非常适合处理 retrieval gone wrong，也就是检索到的文档不相关、质量差。

但在 RAMDocs 这类场景中，问题不一定是 retrieval wrong，而是：

> retrieval succeeded, but the retrieved set itself contains multiple plausible and conflicting answers.

也就是说，所有文档都可能“相关”，但它们互相矛盾。CRAG-style relevance evaluator 可能会保留这些文档，而 EvidenceBoard-RAG 会进一步建模它们之间的冲突关系。

### 5.4 相对 MADAM-RAG 的定位

MADAM-RAG 是同赛道强相关方法，论文中用多轮 LLM agent debate 和 aggregator 处理 ambiguity、misinformation、noise。它是最强相关工作，不应该被轻易当弱 baseline。

我们的定位不要写成“超过 MADAM-RAG”，而是：

> 与 MADAM-RAG 的 debate-heavy 路线不同，我们探索一种更轻量的 evidence-structure-first 路线：先显式构建 Evidence Board 和 Conflict Graph，再进行生成与验证。它更适合课程项目和低 token 成本场景。

如果时间允许，可以实现一个 **MADAM-RAG-lite** 作为可运行 baseline：

```text
每篇文档一个 advocate response → aggregator 汇总 → final answer
```

但不建议承诺超过原版 MADAM-RAG。

---

## 6. Baseline 重新设计

### 6.1 必跑 baseline

| 编号 | Baseline | 对应论文/思想 | 是否好复现 | 作用 |
|---|---|---|---|---|
| B0 | Direct LLM | parametric-only baseline | 极易 | 看不检索时表现 |
| B1 | Vanilla RAG | RAG, NeurIPS 2020 | 易 | 最基础 RAG 对照 |
| B2 | RAG + Relevance Filter | RAGAS/RAG Triad context relevance 思想 | 易 | 检验“只过滤无关文档”够不够 |
| B3 | CRAG-style RAG | CRAG, 2024 | 中 | 强 baseline，检索纠错 |
| B4 | MADAM-RAG-lite | MADAM-RAG, COLM 2025 | 中 | 多 agent debate 风格对照 |

### 6.2 不建议完整复现的 baseline

| 方法 | 原因 | 论文里怎么处理 |
|---|---|---|
| Self-RAG | 需要训练 reflection tokens 或使用专门模型 | Related Work，不作为可运行 baseline |
| 原版 MADAM-RAG | 多轮 debate 成本高，70B 模型复现贵 | 做 lite 版或只引用原论文结果 |
| TruthfulRAG / FaithfulRAG | 2025/2026 较重方法，涉及 KG 或 fact-level modeling | Related Work，作为未来工作 |
| FLARE | 生成过程中动态检索，工程复杂且不直接解决冲突证据 | Related Work |

---

## 7. 实验设计 v2

### 7.1 数据组成

建议一天内最稳版本：

| 数据来源 | 数量 | 目的 |
|---|---:|---|
| RAMDocs | 30 | 主实验：conflict + misinfo + noise |
| FaithEval-inconsistent | 20 | faithfulness / inconsistent context |
| AmbigDocs | 10 | ambiguity / multi-answer |
| Synthetic duplicate perturbation | 10 | 重复证据稀释，仅作为 controlled stress test |

合计 60–70 个 case。若时间太紧，最低版本：

| 数据来源 | 数量 |
|---|---:|
| RAMDocs | 20 |
| FaithEval-inconsistent | 10 |
| Synthetic duplicate perturbation | 5 |

### 7.2 Duplicate perturbation 怎么避免“自产自销”太严重？

duplicate stress test 可以基于公开样本做扰动，不从零编造：

1. 从 RAMDocs 取一个 case；
2. 找到其中一个 `misinfo` 文档；
3. 复制该文档 1/3/5 次，形成不同 duplicate level；
4. 测试方法是否被重复错误证据带偏。

这不是主 benchmark，而是 controlled perturbation。论文里要明确：

> Duplicate experiments are controlled perturbation studies built on public RAMDocs examples and are reported separately.

### 7.3 Outdated evidence 怎么处理？

如果找不到稳定可用的公开 outdated 子集，不要硬编太多。可以写成两种选择：

- 主实验先不强行做 outdated；
- 如果需要覆盖博士生原题“过期页面”，可以从 RARE 或时间敏感 QA 数据中少量抽样；
- 或者基于公开样本加 timestamp perturbation，作为小型 case study，而不是主指标。

不要把“过期”作为主贡献，否则一天内数据可信度会变弱。

---

## 8. 指标设计 v2

### 8.1 主指标

| 指标 | 解释 | 数据适配 |
|---|---|---|
| Exact Match / F1 | 最终答案是否匹配 gold answer | RAMDocs / AmbigDocs |
| Gold Coverage | 多答案场景下覆盖多少 gold answers | RAMDocs / AmbigDocs |
| Wrong Answer Leakage | 是否输出 wrong_answers 中的错误答案 | RAMDocs |
| Noise Suppression Rate | 是否避免引用/采纳 noise 文档 | RAMDocs |
| Faithfulness | 回答是否被上下文支持 | FaithEval / RAGAS-style |
| Abstention Accuracy | 应该拒答/澄清时是否拒答/澄清 | FaithEval / AmbigDocs |
| Average Correction Turns | 平均修正轮数 | 课程要求 |

### 8.2 自定义指标：Conflict Sensitivity

用于回答博士生给出的“冲突敏感度”。

定义思路：

> 当输入文档中存在多个 answer clusters 时，系统是否显式发现并处理这些互斥答案。

简化计算：

```text
Conflict Sensitivity = (# cases where system explicitly flags conflict or ambiguity) / (# cases with annotated conflict or multiple answer candidates)
```

### 8.3 自定义指标：Evidence Dilution Rate

用于衡量重复错误证据是否稀释正确证据。

简化计算：

```text
Evidence Dilution Error = (# cases where duplicated misinfo changes a previously correct answer to wrong) / (# duplicate stress cases)
```

或者做 robustness drop：

```text
Dilution Robustness Drop = Accuracy(original) - Accuracy(duplicated_misinfo)
```

越低越好。

---

## 9. 预期结果叙事：更诚实的写法

### 9.1 不要这样写

> Our method significantly outperforms all baselines in all settings.

这太满，容易翻车。

### 9.2 推荐这样写

> EvidenceBoard-RAG shows clear advantages in conflict-heavy and duplicate-heavy settings by reducing wrong-answer leakage and improving conflict sensitivity. However, on clean or simple cases, the additional verification steps may introduce over-refusal and higher token cost. These results suggest that evidence-centric verification is most useful when retrieved contexts are noisy or mutually inconsistent.

中文：

> EvidenceBoard-RAG 在冲突强、重复干扰强的场景中能减少错误答案泄漏，提高冲突敏感度；但在干净或简单场景中，多步验证可能带来过度拒答和更高 token 成本。这说明证据中心验证机制最适合检索上下文嘈杂或互相矛盾的场景。

---

## 10. 论文结构 v2

### Abstract

- RAG 在真实知识库中会遇到冲突、噪声、重复；
- 现有 RAG / CRAG 主要关注检索相关性，不足以处理相关但互斥的证据；
- 提出 EvidenceBoard-RAG；
- 在 RAMDocs、FaithEval-inconsistent、AmbigDocs 子集上评估；
- 报告优势、成本和失败模式。

### 1. Introduction

重点写：

- RAG 不是万灵药；
- retrieved context 可能彼此冲突；
- relevance filtering 不能解决所有问题；
- 需要 evidence-level conflict modeling；
- 本文贡献：公开数据子集评测 + training-free 证据中心框架 + 新指标。

### 2. Related Work

分为：

1. RAG and Corrective RAG；
2. Faithfulness and RAG Evaluation；
3. Conflicting Evidence in RAG；
4. Agentic / Critic-guided RAG。

### 3. Task and Benchmark

介绍：

- RAMDocs；
- FaithEval-inconsistent；
- AmbigDocs；
- duplicate stress test；
- 指标。

### 4. Method

介绍：

- Evidence Card；
- Evidence Board；
- Conflict Graph；
- Anti-Dilution Aggregation；
- Verifier / Critic。

### 5. Experiments

对比：

- Direct LLM；
- Vanilla RAG；
- Relevance Filter RAG；
- CRAG-style；
- MADAM-RAG-lite；
- Ours。

### 6. Results and Analysis

不要只放一张总表，要放：

- 主结果；
- conflict-heavy 子集；
- duplicate stress；
- clean/simple 子集；
- token cost；
- failure cases。

### 7. Limitations

主动写：

- 不是训练新模型；
- LLM judge 有偏差；
- public subset 数量有限；
- duplicate/outdated 部分仍有 synthetic perturbation；
- 与原版 MADAM-RAG / Self-RAG 不构成完整 SOTA 对比。

---

## 11. 代码实现建议

### 11.1 目录结构

```text
dirtyrag_v2/
  data/
    ramdocs_sample.jsonl
    faitheval_inconsistent_sample.jsonl
    ambigdocs_sample.jsonl
    duplicate_stress.jsonl
  src/
    loaders.py
    baselines.py
    evidence_board.py
    conflict_graph.py
    verifier.py
    metrics.py
    run_experiments.py
  results/
    predictions.csv
    metrics_summary.csv
    failure_cases.md
  app.py
```

### 11.2 最低可运行版本

如果时间极其紧，先实现：

1. RAMDocs loader；
2. Direct LLM；
3. Vanilla RAG / context RAG；
4. Relevance Filter；
5. EvidenceBoard-RAG；
6. EM / wrong answer leakage / conflict sensitivity。

CRAG-style 和 MADAM-RAG-lite 可后补。

---

## 12. 参考论文与引用清单

### 12.1 核心必引

1. **RAG / Vanilla RAG**  
   Patrick Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.  
   https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

2. **CRAG / Corrective RAG**  
   Shi-Qi Yan et al. *Corrective Retrieval Augmented Generation*. 2024.  
   https://openreview.net/forum?id=JnWJbrnaUE

3. **Self-RAG**  
   Akari Asai et al. *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. ICLR 2024.  
   https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html

4. **RAMDocs / MADAM-RAG**  
   Han Wang et al. *Retrieval-Augmented Generation with Conflicting Evidence*. COLM 2025.  
   https://openreview.net/forum?id=z1MHB2m3V9  
   https://github.com/HanNight/RAMDocs

5. **FaithEval**  
   Yifei Ming et al. *FaithEval: Can Your Language Model Stay Faithful to Context, Even If "The Moon is Made of Marshmallows"*. 2024.  
   https://github.com/SalesforceAIResearch/FaithEval

6. **AmbigDocs**  
   Yoonsang Lee et al. *AmbigDocs: Reasoning across Documents on Different Entities under the Same Name*. COLM 2024.  
   https://openreview.net/forum?id=mkYCfO822n

7. **RAGChecker**  
   Dongyu Ru et al. *RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation*. NeurIPS 2024 Datasets and Benchmarks Track.  
   https://papers.nips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract-Datasets_and_Benchmarks_Track.html

### 12.2 可选前沿引用

8. **RARE**  
   Yixiao Zeng et al. *RARE: Retrieval-Aware Robustness Evaluation for Retrieval-Augmented Generation Systems*. 2025.  
   https://arxiv.org/abs/2506.00789

9. **RAG-Critic**  
   Guanting Dong et al. *RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for Retrieval Augmented Generation*. ACL 2025.  
   https://aclanthology.org/2025.acl-long.179/

10. **FaithfulRAG**  
   Qinggang Zhang et al. *FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation*. ACL 2025.  
   https://arxiv.org/abs/2506.08938

11. **TruthfulRAG**  
   Shuyi Liu et al. *TruthfulRAG: Resolving Factual-level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs*. AAAI 2026.  
   https://ojs.aaai.org/index.php/AAAI/article/view/40489

12. **CONFACT**  
   Ziyu Ge et al. *Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs*. 2025.  
   https://arxiv.org/abs/2505.17762

---

## 13. 给另一个 AI 评审的提示词

你可以把下面这段直接丢给另一个 AI：

```text
请你评审下面这个人工智能导论课程论文方案。背景：我们选择“面向脏知识库的 RAG 鲁棒性增强 + 评测基准”方向，目标是在一天左右完成一篇 8–10 页英文 Overleaf 论文和一个可运行 demo。方案不训练新大模型，主打 training-free evidence-centric RAG framework。

请重点评审：
1. 数据集选择是否可信：RAMDocs、FaithEval-inconsistent、AmbigDocs 是否适合；duplicate stress test 是否可以作为补充；是否还需要 RARE / CONFACT / RAGTruth。
2. 方法创新是否成立：Evidence Card、Evidence Board、Conflict Graph、Anti-Dilution Aggregation、Verifier Loop 是否比普通 RAG / CRAG-style baseline 有清晰差异。
3. baseline 是否合理：Direct LLM、Vanilla RAG、RAG + Relevance Filter、CRAG-style RAG、MADAM-RAG-lite 是否足够；Self-RAG / 原版 MADAM-RAG 是否只放 Related Work 合理。
4. 实验指标是否合理：Exact Match、Gold Coverage、Wrong Answer Leakage、Noise Suppression、Conflict Sensitivity、Evidence Dilution Error、Abstention Accuracy 是否能支撑论文主张。
5. 结果叙事是否克制：不要声称 SOTA，只强调在 conflict-heavy / duplicate-heavy 场景中更稳，同时报告 clean 场景下额外成本和过度拒答风险。

请给出严厉评审意见，并指出哪些部分最容易被导师 challenge。
```

---

## 14. 最终建议

修订后方案可以定，但必须遵守三条底线：

1. **主实验不要自构造数据**：优先 RAMDocs + FaithEval-inconsistent + AmbigDocs。
2. **不要硬吹多智能体**：主叙事用 evidence-centric / agentic workflow；如果写 multi-agent，要明确 Evidence Board 通信协议。
3. **不要承诺全面赢**：重点讲在冲突、重复、噪声场景下的鲁棒性提升，同时诚实写失败案例和成本。

最终论文卖点应改成：

> We provide a course-scale but publicly grounded study of robust RAG under dirty knowledge bases, showing how lightweight evidence structuring and conflict-aware verification can reduce wrong-answer leakage under conflicting and duplicated evidence.

这版比原方案更稳、更可信，也更不容易被博士生师兄或老师一眼看穿“自产自销”。
