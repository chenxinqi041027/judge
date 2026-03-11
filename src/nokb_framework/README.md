# COMET-only Strategy Selection (No KB)

轻量级框架：输入对话 → COMET 常识 → 文本编码 → 特征融合 → Cross-Encoder 排序 → 输出策略。

## 组件
- `config.py`：路径、模型名、COMET 关系映射。
- `strategies.py`：固定策略集合（ESConv 对齐）。
- `comet_features.py`：调用 COMET，扁平化常识文本。
- `encoder.py`：BERT 文本编码（可换 Sentence-BERT）。
- `fuse.py`：特征拼接与归一化。
- `ranker.py`：Cross-Encoder 打分（单标签标量）。
- `pipeline.py`：端到端推理脚本，无需 KB。

## 运行
```bash
cd /data2/xqchen/Judge
python -m src.nokb_framework.pipeline \
  --input /data2/xqchen/Qwen3_test/output/emotion_prediction_20260310_220307.jsonl \
  --output_dir /data2/xqchen/Judge/output \
  --log_dir /data2/xqchen/Judge/log \
  --top_k 3 \
  --device cuda
```
CPU 运行可将 `--device cpu`。

## 输入格式
- JSONL，每行包含至少一个 `utterance`/`text`/`reason`，可选 `dialog`（列表，每项含 `speaker` 与 `text`）。

## 输出
- `commonsense`: COMET 生成的事实
- `strategy_candidates`: 打分后的候选（含 category/description/score）
- `strategy`: 最高分类别
- `strategy_confidence`: 最高分

## 训练思路
- 使用 `ranker.StrategyRanker` 对 `[context + commonsense]` 与 `strategy description` 进行二分类或排序微调。
- 标签：正例为人工标注策略类别，其余为负例；或使用 pairwise ranking loss。
