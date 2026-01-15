# Phase 2, Step 5: DOI to BibTeX Conversion

## Overview
This step uses the `parser doi2bib` command to convert DOIs and arXiv IDs to BibTeX citations for use in academic writing.

## Command Executed

```bash
cd parser && uv run parser doi2bib \
  -i ../pipeline_output/phase2_parser/step1_regular/dois.txt \
  -o ../pipeline_output/phase2_parser/step5_doi2bib/references.bib
```

## Input

| Parameter | Value |
|-----------|-------|
| **Input File** | `step1_regular/dois.txt` |
| **Identifiers** | 4 DOIs |
| **Output Format** | BibTeX |

### Input Identifiers
```
10.1109/TPDS.2021.3090322
10.1007/s11390-020-9822-9
10.1109/IPDPSW59300.2023.00107
10.1109/PDSW49588.2019.00006
```

## Output Files

| File | Description |
|------|-------------|
| [references.bib](references.bib) | BibTeX entries for all DOIs |
| [execution.log](execution.log) | Full execution log |

## Results Summary

| Metric | Value |
|--------|-------|
| **Input identifiers** | 4 |
| **BibTeX entries generated** | 4 |
| **Success rate** | 100% |

### Generated BibTeX Entries

| Key | Title | Year | Venue |
|-----|-------|------|-------|
| Tang2022 | Transparent Asynchronous Parallel I/O Using Background Threads | 2022 | IEEE TPDS |
| Byna2020 | ExaHDF5: Delivering Efficient Parallel I/O on Exascale Computing Systems | 2020 | JCST |
| Chowdhury2023 | Efficient Asynchronous I/O with Request Merging | 2023 | IPDPSW |
| Tang2019 | Enabling Transparent Asynchronous I/O using Background Threads | 2019 | PDSW |

## Validation

| Check | Status | Notes |
|-------|--------|-------|
| Command executed | ✅ | Completed without errors |
| All DOIs resolved | ✅ | 4/4 successful |
| BibTeX file generated | ✅ | Valid format |

## Issues / Problems

| Issue | Severity | Description |
|-------|----------|-------------|
| None | - | All DOIs converted successfully |

## Usage

The generated `references.bib` can be used with LaTeX:

```latex
\bibliography{references}
\bibliographystyle{plain}
```

Or with Markdown tools like Pandoc:

```bash
pandoc paper.md --citeproc --bibliography=references.bib -o paper.pdf
```

## Next Step
→ Phase 3: Ingestor (convert documents to markdown)
