# Cross-model / cross-temperature benchmark on real Jarvis MCP

Every provider calls the **real** `jarvis-mcp` stdio server (29 Jarvis-CD tools). Each row below is the mean across 12 held-out cases (6 single, 2 multi, 2 chain_first, 2 error_recovery) × repeats; 95% CIs are bootstrap intervals.

## Environment
| model | family | params | provider | hardware |
|---|---|---|---|---|
| `jarvis-v10` | gemma3 (FT) | 270M | Ollama | NVIDIA GH200 120GB (Delta-AI, aarch64) |

## Per-(model, temperature) aggregate
| model | T | task_success | tool_ok | arg_ok | answer_ok | halluc | mcp_err | gen_s | total_s |
|---|---|---|---|---|---|---|---|---|---|
| `jarvis-v10` | 0.0 | 67% [42, 92] | 92% [75, 100] | 92% [75, 100] | 75% [50, 100] | 0.08 | 0.58 | 1.55 | 1.69 |
| **`jarvis-v10`** | 0.3 | 67% [50, 81] | 94% [86, 100] | 92% [81, 100] | 81% [67, 92] | 0.11 | 0.67 | 1.27 | 8.08 |
| `jarvis-v10` | 0.7 | 67% [50, 81] | 94% [86, 100] | 92% [81, 100] | 78% [64, 92] | 0.11 | 0.78 | 1.66 | 8.49 |

**Bold** = temperature with the highest `task_success_mean` for that model (tie-broken by lowest `gen_s_mean`).

## Best temperature per model
| model | best_T | task_success@T | family | provider |
|---|---|---|---|---|
| `jarvis-v10` | 0.3 | 67% | gemma3 (FT) | Ollama |