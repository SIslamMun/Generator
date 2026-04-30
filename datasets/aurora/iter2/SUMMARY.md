# Iter2 — Final Eval Results

**Date:** 2026-04-30
**Eval set:** 53 hand-authored Aurora holdout questions
**Judge:** Sophia gpt-oss-120b (LLM-as-judge over (Q, gold, model_answer) triples, scored 0-5)

## Scorecard (averaged 0-5 across 53 questions)

| Rank | Model | Avg score | n graded | perfect (5) |
|---:|---|---:|---:|---:|
| 1 | `llama_A` | **2.80** | 49 | 6 |
| 2 | `llama_B` | **2.67** | 45 | 3 |
| 3 | `llama_C3` | **2.31** | 49 | 5 |
| 4 | `coord_llama` | **2.08** | 50 | 1 |
| 5 | `llama_C1` | **2.00** | 50 | 2 |
| 6 | `base_llama` | **1.76** | 45 | 2 |
| 7 | `llama_C2` | **1.67** | 49 | 2 |
| 8 | `coord_gemma270m` | **1.57** | 49 | 0 |
| 9 | `coord_gemma1b` | **1.55** | 47 | 0 |

## Training losses (for reference)

| Model | train_loss |
|---|---:|
| `llama_A` | 0.6224 |
| `llama_B` | 0.6338 |
| `llama_C3` | 0.6523 |
| `llama_C2` | 0.6630 |
| `llama_C1` | 0.6851 |
| `gemma1b_C3` | 0.9367 |
| `gemma1b_C2` | 0.9609 |
| `gemma1b_C1` | 1.0268 |
| `gemma270m_C2` | 1.2462 |
| `gemma270m_C1` | 1.3203 |

## Outputs (all under `/lus/flare/projects/gpu_hack/sislam6/work/runs/iter2/`)

- `eval/questions_50.json` — the 53 holdout questions with gold answers
- `eval/answers_baselines.json` — 6 baselines × 53 Qs (base Llama + 5 Llama LoRAs)
- `eval/coord_llama_answers.json` — Llama coord (router → C1/C2/C3 experts → synth)
- `eval/coord_gemma270m_answers.json` — Gemma 270M coord (with gemma1b_C3 substituted for missing slot)
- `eval/coord_gemma1b_answers.json` — Gemma 1B coord
- `eval/scores.json` — judge ratings per (qid, model)
- `artifacts/<name>/{lora,merged_16bit}/` — 10 trained LoRA + merged checkpoints
- `scripts/push_iter2_to_hf.sh` — push to HF (needs HF_TOKEN env)
- `scripts/post_eval_judge.log` — this run's log