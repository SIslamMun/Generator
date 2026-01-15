# Phase 2, Step 1: Regular (Regex) Reference Parsing

## Overview
This step uses the `parser` module's regex-based parsing to extract references from the research report.

## Command Executed

```bash
cd parser && uv run parser parse-refs \
  ../pipeline_output/phase1_research/research/research_report.md \
  -o ../pipeline_output/phase2_parser/step1_regular \
  --format both \
  --export-batch \
  --export-dois
```

## Input

| Parameter | Value |
|-----------|-------|
| **Input File** | `phase1_research/research/research_report.md` |
| **Format** | both (JSON + Markdown) |
| **Export Batch** | Yes (for paper downloads) |
| **Export DOIs** | Yes (for doi2bib) |

## Output Files

| File | Description |
|------|-------------|
| [references.json](references.json) | Structured reference data |
| [references.md](references.md) | Human-readable reference list |
| [batch.json](batch.json) | Export for `parser batch` command |
| [dois.txt](dois.txt) | DOI/arXiv IDs for `parser doi2bib` |
| [execution.log](execution.log) | Full execution log |

## Results Summary

| Reference Type | Count |
|----------------|-------|
| **DOI** | 4 |
| **Paper** | 4 |
| **PDF** | 12 |
| **Website** | 26 |
| **Total** | 46 |

### Exports
- **Batch export**: 7 papers ready for download
- **DOI export**: 4 identifiers for BibTeX conversion

## Validation

| Check | Status | Notes |
|-------|--------|-------|
| Command executed | ✅ | Completed without errors |
| JSON generated | ✅ | Structured output |
| Markdown generated | ✅ | Human-readable |
| Batch export | ✅ | 7 papers |
| DOI export | ✅ | 4 identifiers |

## Issues / Problems

| Issue | Severity | Description |
|-------|----------|-------------|
| None | - | Regular parsing completed successfully |

## Next Step
→ Step 2: Agent (Claude) parsing for comparison
