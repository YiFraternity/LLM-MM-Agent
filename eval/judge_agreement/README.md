# Cross-judge agreement experiments

This directory documents the judge-model agreement tooling. Generated
experiment outputs are local-only and intentionally ignored, including score
tables, frozen report copies, rendered prompts, raw API artifacts, and caches.

## Code

- `../src/run_multijudge_experiment.py`: freezes inputs, records hashes, runs
  multiple judge models, validates score JSON, and supports resume/retry.
- `../src/analyze_judge_agreement.py`: computes ICC(A,1), ICC(C,1), Kendall W,
  Pearson/Spearman, Lin CCC, bias, MAE, Bland-Altman limits, and bootstrap CIs.
- `../src/3_eval_report_using_mmagent.py`: correctly reads nested
  `criteria.task_understanding` and fails fast when rubric context is missing.
- `../prompts/mmagent_eval_with_task_understanding.yaml`: evidence-first,
  missing-content score caps for calibrated judging.

## Local result directories

- `cross_judge_30_gemini31_deepseekv4/`: local 30-report baseline experiment using
  the historical GPT score set plus Gemini 3.1 Pro and DeepSeek V4 Pro.
- `cross_judge_calibration_v2_10/`: local paired 10-report before/after
  prompt calibration experiment for Gemini and DeepSeek.

These directories are not committed. The API configuration file is also
local-only and must never be committed.
