# 📊 How Far Are We?

## A Problem-Oriented, Stage-Wise Evaluation Framework for Mathematical Modeling

Official implementation for the ACL 2026 submission:

**“How Far Are We? Systematic Evaluation of LLMs vs. Human Experts in Mathematical Contest in Modeling”**

---

# 📖 Overview

Large Language Models (LLMs) perform strongly on isolated reasoning benchmarks.
However, real-world problem solving requires **full-process modeling**:

* Understanding the problem
* Formulating a mathematical abstraction
* Constructing and solving models
* Implementing executable solutions
* Validating and analyzing results

Mathematical modeling competitions provide a natural testbed for evaluating this capability.

Unlike traditional benchmarks:

* There is **no unique correct answer**
* Multiple modeling approaches may be valid
* Evaluation relies on expert judgment

This repository implements a **problem-oriented, stage-wise evaluation framework** designed to measure modeling competence faithfully and diagnostically.

---

# ⭐ Core Contributions

## 1️⃣ Problem-Oriented Evaluation

Instead of using generic rubrics, we:

* Decompose each modeling problem into **subtasks**
* Align evaluation criteria with **problem semantics**
* Ground scoring in **task-specific necessary conditions**

This avoids rewarding superficially polished but fundamentally misaligned solutions.

---

## 2️⃣ Stage-Wise Evaluation Across the Modeling Pipeline

Each subtask is evaluated along seven canonical modeling stages:

1. Problem Identification
2. Problem Formulation
3. Assumption Development
4. Model Construction
5. Model Solving
6. Code Implementation
7. Result Analysis

This structure enables:

* Fine-grained diagnosis
* Stage-level performance comparison
* Failure pattern tracing

---

## 3️⃣ Expert-Aligned Reliability

We validate our framework against independent domain experts.

Using ICC(2,1) to measure agreement:

| Evaluation Scheme | ICC(2,1) vs Expert |
| ----------------- | ------------------ |
| Baseline Rubric   | 0.012              |
| **Ours**          | **0.673**          |



This demonstrates substantially stronger alignment with expert judgment.

---

## 4️⃣ Comprehension–Execution Gap

Applying the framework to state-of-the-art LLMs reveals:

* Strong performance in early stages (problem identification & formulation)
* Significant degradation in:

  * Model solving
  * Code implementation
  * Result validation

Performance declines monotonically across stages.

Scaling model size improves comprehension,
but **does not close the execution gap** .

---

## 5️⃣ Failure Pattern Analysis

Stage-wise failure analysis shows that:

* Failures are rarely due to completely wrong ideas.
* Most errors arise from:

  * Missing specification
  * Lack of verification
  * Incomplete derivation
  * Non-reproducible implementation
  * Missing validation

Errors propagate across stages without correction .

---

# 🏗 Framework Architecture

The evaluation framework operates in two phases:

---

## Phase I: Subtask Decomposition

Each modeling problem is decomposed into:

* Self-contained modeling requirements
* Each requiring a full modeling pipeline

Experts verify that subtasks faithfully represent the original problem intent .

---

## Phase II: Stage-Wise Criteria Instantiation

For each (Subtask × Stage) pair:

* LLM generates fine-grained evaluation criteria
* Experts refine and verify
* Criteria become atomic scoring units

Evaluation is:

* Problem-conditioned
* Stage-aware
* Uniformly applied across models

---

# 📂 Dataset

We evaluate on:

**97 problems from the China Postgraduate Mathematical Contest in Modeling (PMCM)**

Characteristics:

* Graduate-level difficulty
* Multi-page real-world tasks
* Multi-method modeling requirements
* Only ~1–1.5% gold medal rate

Problems are converted into verified LaTeX format via a structured preprocessing pipeline.

---

# 📈 What This Framework Enables

* Faithful evaluation of open-ended modeling
* Expert-aligned scoring
* Diagnostic stage-wise analysis
* Identification of execution-level weaknesses
* Comparative evaluation across model scales

---

# ▶️ Usage

### 1. Generate Stage-Wise Criteria

The system generates JSON-based evaluation schemas conditioned on:

* Problem background
* Subtask definition
* Pipeline position

### 2. Stage-Wise Evaluation

Given:

* Subtask
* Modeling report
* Generated criteria

The framework performs:

* Criterion-wise scoring
* Evidence-based justification
* Stage-level aggregation

All outputs are in structured JSON format.

---

# 📊 Evaluation Protocol

* All models are evaluated under a fixed generation protocol
* Same subtask decomposition applied to all models
* Only the LLM varies

Agreement with experts measured using ICC(2,1) under a two-way random-effects model .

---

# 🔬 Key Insight

Mathematical modeling competence is not a linear extension of language understanding.

LLMs can:

✔ Generate plausible modeling ideas

But struggle to:

✘ Execute models rigorously
✘ Provide checkable solutions
✘ Produce reproducible code
✘ Validate results

Improvement requires:

* Process-aware reasoning
* Stage-wise self-correction
* Execution-grounded validation

—not just scaling model size .

---

# 📜 Citation

If you find this work useful, please cite:

```
@article{anonymous2026howfar,
  title={How Far Are We? Systematic Evaluation of LLMs vs. Human Experts in Mathematical Contest in Modeling},
  journal={ACL 2026 Submission}
}
```