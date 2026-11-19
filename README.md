# Coding rare failure modes in field reports by agreement between a fine-tuned classifier and a language model

Code, outputs of the coders and result files for the paper of the same title (J. Choi, Department of Industrial
Engineering, Yonsei University; submitted to *Reliability Engineering & System Safety*).

The method places a rare failure code on a report only when the code is the highest-ranked rare code within a fine-tuned
classifier's top five codes and a large language model (LLM), answering without seeing this ranking, names it. An optional
selection model chooses the code of every other report. The experiments use medical device reports (FDA MAUDE) and aircraft service
difficulty reports (FAA SDR) of 2025 and January to October 2026, with Qwen3.8-27B and Gemma 4 31B run locally through llama.cpp.

## Contents

| Folder | What it holds |
|---|---|
| `scripts/` | all code of the project; the scripts of the paper's pipeline are listed below |
| `splits/` | report keys of the samples (random samples, strata, balanced tests) |
| `runs/` | answers of the LLM coders (`llm_cls_*`: zero-shot, `llm_ragfull_*`: retrieval, `llm_cand5_*`: shown the classifier's top five), shortlists, retrieved neighbours, classifier outputs on the samples |
| `results/` | the metric files behind every table and figure (`results/paper_v4`, `rare_year`, `reliability`, `why_it_works`, `compute_time`, ...) |

## Data

The reports are public: MAUDE files of the US Food and Drug Administration
(https://www.fda.gov/medical-devices/mandatory-reporting-requirements-manufacturers-importers-and-device-user-facilities/about-manufacturer-and-user-facility-device-experience-maude)
and the Service Difficulty Reporting System of the Federal Aviation Administration (https://sdrs.faa.gov/). The data scripts
below build the cleaned report pools from the downloaded files. The fine-tuned classifiers and their outputs over every report
of the test years (several gigabytes) are not in this repository; they are available from the author on request.

## Scripts of the paper

**Data**

- `scripts/maude_pool.py`: Build the deduplicated MAUDE 2024 item pool that every later split draws from.
- `scripts/sdr_data.py`: Build the FAA Service Difficulty Report corpus in the MAUDE pool schema (notes/prereg-sdr-2026-10-24.md; corpus search notes/corpus-search-2026-10-23.md).
- `scripts/maude_data.py`: Shared loaders for the MAUDE stage (splits/maude_v1.json over data/maude/pool.jsonl).
- `scripts/maude_split_2025.py`: Fix the out-of-time test set: 2025 reports for the 424 known codes.
- `scripts/maude_split_natural.py`: Fix the natural-distribution 2025 sample (notes/prereg-maude-natural-2026-10-23.md).

**Classifier**

- `scripts/maude_encoder.py`: MAUDE encoder arms: roberta-base, CE or class-balanced CE, at a training size N.
- `scripts/maude_encoder_year.py`: The selector's encoder over every known-code report of a year (notes/selector-vs-rules-2026-11-11.md, section "한 해 전체로 재는 방법"). GPU: lent by the user for 2026-11-12 20:00 to 2026-11-13 09:00.
- `scripts/maude_encoder_probs.py`: Full softmax of the selector's encoder (class-balanced RoBERTa-large, full N, seed 42; the retrained weights in ~/models/maude_encoders/cb-large_full_seed42 that produced every selector input) on the three selector sets,

**Shortlist, retrieval and LLM coders**

- `scripts/rare_year.py`: Rare codes over a whole year: the encoder shortlists, an LLM confirms (notes/selector-vs-rules-2026-11-11.md, section "한 해 전체로 재는 방법").
- `scripts/maude_llm.py`: MAUDE LLM arms: request builders, label normalization and novelty parsing.
- `scripts/openweight_llm.py`: Open-weight LLM arms on a local llama.cpp server.
- `scripts/openweight_fetch.py`: Fetch the open-weight arms' runtime and weights, pinned.

**Selection model and baselines**

- `scripts/candidate_ranker.py`: Candidate ranker (exploration; notes/selector-vs-rules-2026-11-11.md, section "후보 재순위").
- `scripts/maude_selector.py`: Starvation-aware selector (notes/prereg-maude-selector-2026-10-23.md).
- `scripts/selector_vs_rules.py`: Selector against simple rules (notes/selector-vs-rules-2026-11-11.md).
- `scripts/redesign_score.py`: Score the redesign once (notes/prereg-redesign-2026-11-06.md).
- `scripts/selection_ablation.py`: Where the selection model's accuracy comes from, and the encoder corrected for label frequency (review of manuscript v4).
- `scripts/logit_adjusted_exact.py`: Exact whole-year rare placements of the logit-adjusted classifier (tau = 0.5), from one GPU pass of the classifier over every known-code report of a year. 2026-11-19, after the manuscript review: the year files of maude_

**Numbers, tests and figures**

- `scripts/paper_v4_numbers.py`: All numbers of the v4 paper from one fixed configuration (notes/paper-plan-v4-2026-11-16.md).
- `scripts/accuracy_tests.py`: McNemar's exact test for the accuracy differences of the paper (2026-11-18: the user asked for significance marks in the accuracy table in place of bootstrap intervals).
- `scripts/reviewer_checks.py`: Whole-year checks asked for by the review of the v4 manuscript (CPU only, exact counts over every known-code report).
- `scripts/uncertainty_trigger.py`: Shortlist of the agreement rule against an uncertainty trigger, over whole years (paper v4, Section 5.3).
- `scripts/why_it_works.py`: Why the gate works: the analyses of notes/why-it-works-plan-2026-11-16.md (CPU; experiment 3's GPU runs come from run_cand5.ps1 and are scored here once their files exist).
- `scripts/case_study.py`: One rare failure mode followed through a year, and how the gate's gain is spread over the rare codes (notes/why-it-works-2026-11-16.md, section "사례"; user 2026-11-17: "사례 하나는 돌려볼까 일단?").
- `scripts/tail_split.py`: Where do the LLM's correct rare answers sit in the encoder's ranking? (notes/paper-narrative-v4-2026-11-16.md, 11-16)
- `scripts/reliability_year.py`: Failure-mode statistics over a whole year: what the gate changes in the year's counts of rare failure modes (notes/paper-outline-v4-2026-11-16.md; user 2026-11-16: "신뢰성, 그런 분석들을 좀 해야겠는데").
- `scripts/term_check.py`: A check of rare-code placements that needs no rater: does the narrative contain the placed code's own word? (notes/why-it-works-2026-11-16.md, section "글에 그 말이 있는가"; user 2026-11-17: "ㅇㅇ 돌려봐").
- `scripts/withheld_check.py`: Reports filed under the withheld codes, both test years: how many rare codes the classifier and the agreement rule place on them (notes/mock-review-2026-11-18.md, E3; asked for by the user on 2026-11-18 to decide whether
- `scripts/run_withheld.ps1`: Windows driver of the test on reports filed under codes withheld from training (Section 5.7).
- `scripts/compute_time.py`: Time per report of each part of the method on this machine (2026-11-18: a reviewer will ask how much the method adds to the classifier alone). Each phase runs in its own process, so that only one model holds the GPU at a
- `scripts/paper_v5_figures.py`: Data figures of the RESS manuscript, second design (2026-11-18, after the user found the figures too alike).
- `scripts/paper_v4_diagram.py`: Method figure of the v4 paper (notes/paper-plan-v4-2026-11-16.md, figure 1), drawn as PowerPoint shapes with the helpers and colours of paper_diagrams.py (the author's SMART/SARC figure language): rounded boxes, thin dar

## Environment

Python 3.13 with the packages in `requirements.txt`; one NVIDIA RTX 3090 (24 GB) for the classifier and the LLMs; llama.cpp
for the LLMs (Qwen3.8-27B UD-Q4_K_M and Gemma 4 31B QAT q4_0 GGUF files, greedy decoding, context 16,384 tokens).

## License

Code: MIT License (see `LICENSE`). The report data come from public US government sources.
