# Cross-model / cross-temperature benchmark on real Jarvis MCP

Every provider calls the **real** `jarvis-mcp` stdio server (29 Jarvis-CD tools). Each row below is the mean across 12 held-out cases (6 single, 2 multi, 2 chain_first, 2 error_recovery) × repeats; 95% CIs are bootstrap intervals.

## Environment
| model | family | params | provider | hardware |
|---|---|---|---|---|
| `claude-haiku-4-5` | claude | ? | Claude Code SDK | Anthropic data center (opaque) |
| `claude-sonnet-4-6` | claude | ? | Claude Code SDK | Anthropic data center (opaque) |
| `claude-opus-4-7` | claude | ? | Claude Code SDK | Anthropic data center (opaque) |
| `gemini-2.5-flash` | gemini | ? | Google ADK | Google data center (opaque) |
| `gemini-2.5-pro` | gemini | ? | Google ADK | Google data center (opaque) |

## Per-(model, temperature) aggregate
| model | T | task_success | tool_ok | arg_ok | answer_ok | halluc | mcp_err | gen_s | total_s |
|---|---|---|---|---|---|---|---|---|---|
| **`claude-haiku-4-5`** | 0.0 | 17% [0, 42] | 17% [0, 42] | 17% [0, 42] | 75% [50, 100] | 0.00 | 0.08 | 9.45 | 25.03 |
| **`claude-opus-4-7`** | 0.0 | 17% [0, 42] | 17% [0, 42] | 17% [0, 42] | 92% [75, 100] | 0.00 | 0.17 | 17.90 | 17.90 |
| **`claude-sonnet-4-6`** | 0.0 | 17% [0, 42] | 17% [0, 42] | 17% [0, 42] | 67% [42, 92] | 0.08 | 0.17 | 13.76 | 31.98 |
| **`gemini-2.5-flash`** | 0.0 | 17% [0, 42] | 17% [0, 42] | 17% [0, 42] | 58% [33, 83] | 0.00 | 0.25 | 4.17 | 4.40 |
| `gemini-2.5-flash` | 0.3 | 17% [6, 31] | 19% [8, 33] | 19% [8, 33] | 72% [58, 86] | 0.06 | 0.19 | 4.64 | 4.94 |
| `gemini-2.5-flash` | 0.7 | 17% [6, 31] | 17% [6, 31] | 17% [6, 31] | 75% [61, 89] | 0.03 | 0.22 | 4.65 | 5.50 |
| `gemini-2.5-pro` | 0.0 | 17% [0, 42] | 17% [0, 42] | 17% [0, 42] | 67% [42, 92] | 0.00 | 0.33 | 11.07 | 11.21 |
| **`gemini-2.5-pro`** | 0.3 | 17% [6, 31] | 17% [6, 31] | 17% [6, 31] | 58% [42, 75] | 0.06 | 0.28 | 10.05 | 10.18 |
| `gemini-2.5-pro` | 0.7 | 17% [6, 31] | 17% [6, 31] | 17% [6, 31] | 61% [44, 78] | 0.00 | 0.31 | 10.64 | 10.77 |

**Bold** = temperature with the highest `task_success_mean` for that model (tie-broken by lowest `gen_s_mean`).

## Best temperature per model
| model | best_T | task_success@T | family | provider |
|---|---|---|---|---|
| `claude-haiku-4-5` | 0.0 | 17% | claude | Claude Code SDK |
| `claude-opus-4-7` | 0.0 | 17% | claude | Claude Code SDK |
| `claude-sonnet-4-6` | 0.0 | 17% | claude | Claude Code SDK |
| `gemini-2.5-flash` | 0.0 | 17% | gemini | Google ADK |
| `gemini-2.5-pro` | 0.3 | 17% | gemini | Google ADK |