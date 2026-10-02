# 模型裁判一致性实验报告

## 结论

总体平均分的 ICC(A,1) 未达到预设阈值 0.90，当前数据不支持“不同裁判给出高度一致的绝对分数”。

- 完全配对样本数：30
- 总体 ICC(A,1)：0.147 [0.053, 0.238]（poor）
- 总体 ICC(C,1)：0.406 [0.183, 0.571]（poor）
- ICC(A,1) 衡量绝对分数是否一致；ICC(C,1) 仅衡量去除系统性宽严差后的变化一致性。

## 数据覆盖

| 裁判 | 原始样本数 | 未进入共同交集 | 来源 |
|---|---:|---:|---|
| gpt-5-mini | 30 | 0 | `eval/judge_agreement/cross_judge_30_gemini31_deepseekv4/summaries/gpt-5-mini.xlsx` |
| gemini-3.1-p | 30 | 0 | `eval/judge_agreement/cross_judge_30_gemini31_deepseekv4/summaries/gemini-3.1-p.xlsx` |
| ali-deepseek-v4-pro | 30 | 0 | `eval/judge_agreement/cross_judge_30_gemini31_deepseekv4/summaries/ali-deepseek-v4-pro.xlsx` |

## 各维度一致性

| 评分项 | ICC(A,1), 95% CI | ICC(C,1), 95% CI | Kendall W, 95% CI | 绝对一致等级 |
|---|---:|---:|---:|---|
| overall_mean | 0.147 [0.053, 0.238] | 0.406 [0.183, 0.571] | 0.634 [0.444, 0.787] | poor |
| modeling_rigor | 0.055 [0.000, 0.105] | 0.140 [0.000, 0.268] | 0.460 [0.284, 0.624] | poor |
| practicality_scientificity | 0.096 [-0.028, 0.253] | 0.217 [-0.071, 0.499] | 0.515 [0.315, 0.705] | poor |
| problem_analysis | 0.142 [0.013, 0.271] | 0.252 [0.028, 0.438] | 0.548 [0.385, 0.694] | poor |
| result_bias | 0.431 [0.244, 0.571] | 0.513 [0.333, 0.650] | 0.714 [0.566, 0.823] | poor |

## 总体平均分的两两诊断

| 裁判 A | 裁判 B | 均值 A | 均值 B | 满分率 A | 满分率 B | B-A 偏差 | MAE | Pearson | Spearman | Lin CCC | |差|≤1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-5-mini | gemini-3.1-p | 6.408 | 8.671 | 0.0% | 6.7% | 2.263 | 2.263 | 0.465 | 0.450 | 0.091 | 0.067 |
| gpt-5-mini | ali-deepseek-v4-pro | 6.408 | 5.825 | 0.0% | 0.0% | -0.583 | 1.083 | 0.407 | 0.407 | 0.268 | 0.633 |
| gemini-3.1-p | ali-deepseek-v4-pro | 8.671 | 5.825 | 6.7% | 0.0% | -2.846 | 2.846 | 0.545 | 0.497 | 0.137 | 0.033 |

## 判定规则

- 预注册主判据：总体平均分 ICC(A,1) ≥ 0.90。
- 强证据判据：ICC(A,1) 的 95% bootstrap 置信区间下界也 ≥ 0.90。
- 相关系数高但 ICC(A,1) 低，表示排序可能相近，但原始分数不能互换。
- 两位裁判时 Kendall W 与 Spearman 相关存在直接对应关系，不能把 W 的数值直接当成绝对一致性。
- Bootstrap 以 `Task ID` 为聚类单位重采样，保持同一题目的多个模型回答和所有裁判评分一起出现。
