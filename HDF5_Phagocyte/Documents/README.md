# Phase 2: Parser - Reference Extraction & Acquisition

## Overview
This phase uses the `parser` module to extract references from the research report, compare parsing methods, download papers, and convert DOIs to BibTeX.

## Steps

| Step | Name | Status | Description |
|------|------|--------|-------------|
| 1 | [Regular Parsing](step1_regular/README.md) | ✅ | Regex-based reference extraction |
| 2 | [Agent Parsing](step2_agent/README.md) | ✅ | Claude AI-based extraction |
| 3 | [Comparison](step3_comparison/README.md) | ✅ | Regular vs Agent analysis |
| 4 | [Acquisition](step4_acquisition/README.md) | ✅ | Paper downloads |
| 5 | [DOI2BIB](step5_doi2bib/README.md) | ✅ | BibTeX conversion |

## Summary Results

### Reference Extraction

| Method | Total | DOI | Paper | PDF | Website |
|--------|-------|-----|-------|-----|---------|
| Regular (regex) | 46 | 4 | 4 | 12 | 26 |
| Agent (Claude) | 42 | 3 | 3 | 12 | 24 |
| Common | 40 | - | - | - | - |

### Paper Acquisition

| Status | Count |
|--------|-------|
| Downloaded | 5 |
| Skipped (cached) | 1 |
| Failed | 1 |
| **Total** | 7 |

### DOI to BibTeX

| Input | Output |
|-------|--------|
| 4 DOIs | 4 BibTeX entries |

## Key Commands Used

```bash
# Step 1: Regular parsing
parser parse-refs research_report.md -o step1_regular --format both --export-batch --export-dois

# Step 2: Agent parsing
parser parse-refs research_report.md -o step2_agent --agent claude --format both --export-batch --export-dois

# Step 3: Comparison
parser parse-refs research_report.md -o step3_comparison --compare --agent claude

# Step 4: Acquisition
parser batch batch.json -o step4_acquisition/papers -v

# Step 5: DOI to BibTeX
parser doi2bib -i dois.txt -o step5_doi2bib/references.bib
```

## Validation Summary

| Check | Status |
|-------|--------|
| Regular parsing | ✅ |
| Agent parsing | ✅ |
| Comparison report | ✅ |
| Paper downloads | ⚠️ 1 failed |
| BibTeX generation | ✅ |

## Issues Encountered

| Step | Issue | Severity | Resolution |
|------|-------|----------|------------|
| 4 | 1 paper missing DOI | ❌ | Could not download |
| 4 | Institutional access failed | ⚠️ | Used Sci-Hub fallback |
| 4 | Sci-Hub used for 2 papers | ⚠️ | Legal gray area warning |

## Next Phase
→ Phase 3: Ingestor - Convert documents to markdown
