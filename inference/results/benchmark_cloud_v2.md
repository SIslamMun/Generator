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
| **`claude-haiku-4-5`** | 0.0 | 58% [33, 83] | 75% [50, 100] | 75% [50, 100] | 83% [58, 100] | 0.00 | 0.25 | 8.44 | 12.47 |
| **`claude-opus-4-7`** | 0.0 | 58% [33, 83] | 83% [58, 100] | 83% [58, 100] | 92% [75, 100] | 0.08 | 0.33 | 14.40 | 14.40 |
| **`claude-sonnet-4-6`** | 0.0 | 42% [17, 75] | 67% [42, 92] | 67% [42, 92] | 83% [58, 100] | 0.00 | 0.33 | 15.79 | 19.31 |
| `gemini-2.5-flash` | 0.0 | 75% [50, 100] | 83% [58, 100] | 83% [58, 100] | 83% [58, 100] | 0.08 | 0.33 | 5.01 | 5.26 |
| **`gemini-2.5-flash`** | 0.3 | 75% [61, 89] | 83% [69, 94] | 83% [69, 94] | 81% [67, 92] | 0.08 | 0.31 | 4.73 | 6.11 |
| `gemini-2.5-flash` | 0.7 | 72% [58, 86] | 92% [81, 100] | 92% [81, 100] | 78% [64, 89] | 0.06 | 0.47 | 5.42 | 6.20 |
| **`gemini-2.5-pro`** | 0.0 | 67% [42, 92] | 75% [50, 100] | 75% [50, 100] | 75% [50, 100] | 0.00 | 0.08 | 6.62 | 9.20 |
| `gemini-2.5-pro` | 0.3 | 67% [50, 81] | 78% [64, 92] | 78% [64, 92] | 72% [58, 86] | 0.00 | 0.14 | 7.43 | 10.45 |
| `gemini-2.5-pro` | 0.7 | 17% [6, 31] | 19% [8, 33] | 19% [8, 33] | 31% [17, 47] | 0.03 | 0.22 | 11.04 | 11.07 |

**Bold** = temperature with the highest `task_success_mean` for that model (tie-broken by lowest `gen_s_mean`).

## Best temperature per model
| model | best_T | task_success@T | family | provider |
|---|---|---|---|---|
| `claude-haiku-4-5` | 0.0 | 58% | claude | Claude Code SDK |
| `claude-opus-4-7` | 0.0 | 58% | claude | Claude Code SDK |
| `claude-sonnet-4-6` | 0.0 | 42% | claude | Claude Code SDK |
| `gemini-2.5-flash` | 0.3 | 75% | gemini | Google ADK |
| `gemini-2.5-pro` | 0.0 | 67% | gemini | Google ADK |