# Cross-model / cross-temperature benchmark on real Jarvis MCP

Every provider calls the **real** `jarvis-mcp` stdio server (29 Jarvis-CD tools). Each row below is the mean across 12 held-out cases (6 single, 2 multi, 2 chain_first, 2 error_recovery) × repeats; 95% CIs are bootstrap intervals.

## Environment
| model | family | params | provider | hardware |
|---|---|---|---|---|
| `nemotron-nano-30b` | nemotron-3 | 30B | Ollama | NVIDIA GH200 120GB (Delta-AI, aarch64) |
| `nemotron-super-120b` | nemotron-3 | 120B | Ollama | NVIDIA GH200 120GB (Delta-AI, aarch64) |

## Per-(model, temperature) aggregate
| model | T | task_success | tool_ok | arg_ok | answer_ok | halluc | mcp_err | gen_s | total_s |
|---|---|---|---|---|---|---|---|---|---|
| `nemotron-nano-30b` | 0.0 | 67% [42, 92] | 83% [58, 100] | 83% [58, 100] | 67% [42, 92] | 0.00 | 1.17 | 14.39 | 14.54 |
| **`nemotron-nano-30b`** | 0.3 | 69% [56, 83] | 86% [75, 94] | 86% [75, 94] | 72% [56, 86] | 0.00 | 0.72 | 9.68 | 23.32 |
| `nemotron-nano-30b` | 0.7 | 69% [56, 83] | 81% [67, 92] | 81% [67, 92] | 72% [58, 86] | 0.00 | 0.81 | 11.49 | 15.00 |
| **`nemotron-super-120b`** | 0.0 | 67% [42, 92] | 92% [75, 100] | 83% [58, 100] | 75% [50, 100] | 0.00 | 1.00 | 9.26 | 29.56 |
| `nemotron-super-120b` | 0.3 | 61% [44, 75] | 89% [78, 97] | 83% [69, 94] | 64% [47, 78] | 0.00 | 1.08 | 7.95 | 21.59 |
| `nemotron-super-120b` | 0.7 | 64% [47, 78] | 89% [78, 97] | 83% [72, 94] | 69% [56, 83] | 0.00 | 1.03 | 7.84 | 24.75 |

**Bold** = temperature with the highest `task_success_mean` for that model (tie-broken by lowest `gen_s_mean`).

## Best temperature per model
| model | best_T | task_success@T | family | provider |
|---|---|---|---|---|
| `nemotron-nano-30b` | 0.3 | 69% | nemotron-3 | Ollama |
| `nemotron-super-120b` | 0.0 | 67% | nemotron-3 | Ollama |