# 毕设策略选择计划

## 目标
- 在已有情绪识别与原因分析的输出基础上，引入常识库，生成合适的应对策略，并继续以 JSON 形式输出。

## 待办步骤
1) 确定策略输出格式
	- 在现有结果上增加字段：`strategy`（字符串或列表）、`source`（策略依据）、`confidence`（0-1）。

2) 构建/引入常识库
	- 方案 A：手工或少量标注构建 `data/strategy_kb.jsonl`，字段示例：`{"emotion": "afraid", "triggers": ["fear", "scared"], "strategies": ["先确认安全，给予安慰", "建议寻求帮助或医疗"] , "cautions": ["避免轻描淡写"]}`。
	- 方案 B：使用公开心理学/同理心资源（如 EmpatheticDialogues、CARE、COMET 类常识）提取规则，转存为与方案 A 相同格式。
	- 方案 C：用 LLM 生成初稿策略，再人工筛选后固化进库。

3) 检索与匹配逻辑
	- 输入：`pred_emotion`（模型预测）+ `reason`（CoT 解释文本）作为检索与打分的信号。
	- 基线检索与打分流程：
		1. 情绪标签匹配（high weight）：如果 KB 条目 `emotion` 与 `pred_emotion` 精确相同，给予较高基础分。
		2. 触发词/触发短语匹配（medium weight）：将 KB 中的 `triggers` 与 `reason` 进行分词后计算 Jaccard，相交比例用于得分；若 `triggers` 中任一词在 `reason` 中出现，额外加分。
		3. TF-IDF 相似度（medium weight，可选实现）：将 KB 条目（triggers + strategy 描述）与 `reason` 用 `TfidfVectorizer` 编码，计算余弦相似度作为候选分。
		4. 得分合并与归一化：用加权和合并以上分数（可配置权重），输出 `match_score`（0-1），并以此作为 `confidence` 的一部分。
	- 可选增强：
		- 使用 WordNet/synonym 扩展触发词集合以覆盖同义表达（轻量）；
		- 使用句向量检索（`sentence-transformers` + FAISS）做语义召回，先召回 N 个候选再按上面规则重新打分；适用于 KB 较大或需要更语义化匹配的场景。
	- 实施要点：尽量先用轻量 TF-IDF/Jaccard 实现基线，确保可解释性；当样本量与 KB 增长再引入向量索引。

4) 策略生成/重写
	- 候选生成：对每条输入先检索并返回 Top-N 候选策略（含原始 KB 策略文本和 `triggers`/`emotion` 标签）。
	- 排序规则（示例，可配置）：
		- `score = w_emotion * emotion_match + w_jaccard * trigger_overlap + w_tfidf * tfidf_sim`，若 `reason` 标注为 `explicit`，可对包含触发词的候选做额外提升。
		- 同一候选可以根据用户指定偏好（例如更具行动性或更具情感安抚性）做二次排序。
	- 输出格式：在最终样本 JSON 中加入字段：
		- `strategy`: Top-K 策略文本列表（字符串或对象列表）；
		- `strategy_source`: 列表，标明每条策略来自哪条 KB 记录或由何种方法生成（e.g., "kb:ID42", "comet"）；
		- `strategy_confidence`: 与 `match_score` 对齐的 0-1 数值；
		- `strategy_rationale`（可选）：简短文本说明为何该策略被选中（例如匹配到的触发词、情绪一致性）。
	- 可选重写：保留原有项——可调用基座 LLM 对 Top-K 策略做简短重写，提示中明确要求：保持同理心、安全性、可执行性并避免医疗/法律建议；重写为可选步骤并记录 `strategy_source` 为 `llm_rewrite`。
	- 安全与审查：对生成或重写出的策略加入简单规则过滤（禁用暴力、歧视、医疗/法律断言），并记录任何被过滤或降级的候选。


5) 集成到推理流水线
	- 在情绪原因解析后调用策略选择模块，生成 `strategy` 字段，写入输出 JSON。
	- 产物保存在 `/data2/xqchen/Qwen3_test/output`，文件名保持时间戳。

6) 评估与迭代
	- 人工抽样对比：策略相关性、可行性、安全性。
	- 记录失败案例，补充/修正常识库。

## 代码脚手架（放在 /data2/xqchen/Judge）
- 新增文件：`Judge/strategy_selector.py`，提供：
	- `load_kb(path)`: 加载 JSON/JSONL 常识库。
	- `select_strategy(emotion, reason, kb, top_k=3)`: 基于情绪标签与关键词相似度返回候选策略。
	- `attach_strategy(sample, kb)`: 将策略写回样本，便于与现有流水线衔接。
- 示例常识库文件：`Judge/strategy_kb_example.jsonl`（可复制为正式库）。

## 下一步
- 先以方案 A 手工构建 5-10 条示例策略（可先编辑 `Judge/strategy_kb_example.jsonl`），跑一小批样本验证。
- 如需向量检索，再补充 sentence-transformers 依赖并构建索引。
