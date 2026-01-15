# Phase 2, Step 2: Agent (Claude) Reference Parsing

## Overview
This step uses the `parser` module's AI agent (Claude) to extract references from the research report with better context understanding.

## Command Executed

```bash
cd parser && uv run parser parse-refs \
  ../pipeline_output/phase1_research/research/research_report.md \
  -o ../pipeline_output/phase2_parser/step2_agent \
  --format both \
  --export-batch \
  --export-dois \
  --agent claude
```

## Input

| Parameter | Value |
|-----------|-------|
| **Input File** | `phase1_research/research/research_report.md` |
| **Agent** | Claude (claude-sonnet-4-20250514) |
| **Format** | both (JSON + Markdown) |
| **Export Batch** | Yes |
| **Export DOIs** | Yes |

## Output Files

| File | Description |
|------|-------------|
| [references.json](references.json) | Structured reference data |
| [references.md](references.md) | Human-readable reference list |
| [batch.json](batch.json) | Export for `parser batch` command |
| [dois.txt](dois.txt) | DOI/arXiv IDs for `parser doi2bib` |
| [agent_raw_response.txt](agent_raw_response.txt) | Raw Claude response |
| [agent_result.json](agent_result.json) | Full agent result with metadata |
| [execution.log](execution.log) | Full execution log |

## Results Summary

| Reference Type | Count |
|----------------|-------|
| **Paper** | 3 |
| **DOI** | 3 |
| **PDF** | 12 |
| **Website** | 24 |
| **Total** | 42 |

### Exports
- **Batch export**: 14 papers ready for download
- **DOI export**: 3 identifiers for BibTeX conversion

## Validation

| Check | Status | Notes |
|-------|--------|-------|
| Command executed | ✅ | Completed without errors |
| Agent response | ✅ | Claude processed successfully |
| JSON generated | ✅ | Structured output |
| Markdown generated | ✅ | Human-readable |
| Batch export | ✅ | 14 papers |
| DOI export | ✅ | 3 identifiers |

## Issues / Problems

| Issue | Severity | Description |
|-------|----------|-------------|
| None | - | Agent parsing completed successfully |

## Comparison with Regular Parsing

| Metric | Regular | Agent |
|--------|---------|-------|
| Total | 46 | 42 |
| DOIs | 4 | 3 |
| Papers | 4 | 3 |
| PDFs | 12 | 12 |
| Websites | 26 | 24 |

## Next Step
→ Step 3: Comparison of regular vs agent outputs
