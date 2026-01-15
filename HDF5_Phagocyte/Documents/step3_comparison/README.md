# Phase 2, Step 3: Regular vs Agent Comparison

## Overview
This step runs both regular (regex) and agent (Claude) parsing on the same document and generates a comparison report.

## Command Executed

```bash
cd parser && uv run parser parse-refs \
  ../pipeline_output/phase1_research/research/research_report.md \
  -o ../pipeline_output/phase2_parser/step3_comparison \
  --compare \
  --agent claude
```

## Input

| Parameter | Value |
|-----------|-------|
| **Input File** | `phase1_research/research/research_report.md` |
| **Mode** | Comparison (--compare) |
| **Agent** | Claude |

## Output Files

| File | Description |
|------|-------------|
| [regular/regular_references.json](regular/regular_references.json) | Regex-based references |
| [regular/regular_references.md](regular/regular_references.md) | Regex-based markdown |
| [agent/agent_claude_references.json](agent/agent_claude_references.json) | Agent-based references |
| [agent/agent_claude_references.md](agent/agent_claude_references.md) | Agent-based markdown |
| [agent/agent_raw_response.txt](agent/agent_raw_response.txt) | Raw Claude response |
| [agent/agent_result.json](agent/agent_result.json) | Full agent result |
| [comparison_report.md](comparison_report.md) | Detailed comparison report |
| [execution.log](execution.log) | Full execution log |

## Results Summary

### Reference Counts

| Method | Total | DOI | Paper | PDF | Website |
|--------|-------|-----|-------|-----|---------|
| **Regular** | 46 | 4 | 4 | 12 | 26 |
| **Agent** | 43 | 3 | 3 | 12 | 25 |

### Overlap Analysis

| Metric | Count |
|--------|-------|
| **Common references** | 40 |
| **Only in Regular** | 6 |
| **Only in Agent** | 3 |

### References Only Found by Regular Parsing
1. `[doi] 10.1007/s11390-020-9822-9`
2. `[paper] Efficient Asynchronous I/O with Request Merging`
3. `[paper] Enabling Transparent Asynchronous I/O using Background Threads`
4. `[paper] Transparent Asynchronous Parallel I/O Using Background Threads`
5. `[paper] Tuning HDF5 subfiling performance on parallel file systems`
6. `[website] https://jcst.ict.ac.cn/en/10.1007/s11390-020-9822-9`

### References Only Found by Agent Parsing
1. `[paper] 10.1109/IPDPSW59300.2023.00107`
2. `[paper] 10.1109/PDSW49588.2019.00006`
3. `[paper] 10.1109/TPDS.2021.3090322`

## Analysis

### Regular Parsing Strengths
- Better at extracting explicit DOIs from text
- Captures more website URLs
- Faster execution (milliseconds)

### Agent Parsing Strengths
- Better context understanding
- Recognizes DOIs formatted as paper references
- Fewer false positives on websites

### Recommendation
Use **regular parsing** for quick extraction, **agent parsing** when accuracy is critical for complex documents.

## Validation

| Check | Status | Notes |
|-------|--------|-------|
| Regular parsing | ✅ | 46 references |
| Agent parsing | ✅ | 43 references |
| Comparison report | ✅ | Generated |

## Issues / Problems

| Issue | Severity | Description |
|-------|----------|-------------|
| Overlap variance | ℹ️ Info | 6 refs only in regular, 3 only in agent |

## Next Step
→ Step 4: Acquisition (download papers)
