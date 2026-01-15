# Phase 2, Step 4: Paper Acquisition

## Overview
This step uses the `parser batch` command to download papers from the extracted references using multiple sources (Unpaywall, arXiv, PMC, Semantic Scholar, institutional access, Sci-Hub, etc.).

## Command Executed

```bash
cd parser && uv run parser batch \
  ../pipeline_output/phase2_parser/step1_regular/batch.json \
  -o ../pipeline_output/phase2_parser/step4_acquisition/papers \
  -v
```

## Input

| Parameter | Value |
|-----------|-------|
| **Input File** | `step1_regular/batch.json` |
| **Papers to retrieve** | 7 |
| **Max concurrent** | 3 |
| **Institutional access** | ✅ Enabled (IIT EZProxy) |

## Output Files

| File | Description | Size |
|------|-------------|------|
| `Byna_2020_ExaHDF5_Delivering_Efficient_Parallel_IO_on_Exasca.pdf` | ExaHDF5 paper | 798 KB |
| `Chowdhury_2023_Efficient_Asynchronous_IO_with_Request_Merging.pdf` | Async I/O paper | 4.1 MB |
| `Tang_2019_Enabling_Transparent_Asynchronous_IO_using_Backgro.pdf` | Background threads paper | 449 KB |
| `Tang_2022_Transparent_Asynchronous_Parallel_IO_Using_Backgro.pdf` | Parallel I/O paper | 4.6 MB |
| `failed/` | Failed download logs | - |
| [execution.log](execution.log) | Full execution log | - |

## Results Summary

| Status | Count | Percentage |
|--------|-------|------------|
| ✅ **Downloaded** | 5 | 71% |
| ⊘ **Skipped (cached)** | 1 | 14% |
| ❌ **Failed** | 1 | 14% |
| **Total** | 7 | 100% |

### Download Sources Used

| Paper | Source | Notes |
|-------|--------|-------|
| ExaHDF5 (Byna 2020) | semantic_scholar | Open access |
| Async I/O Request Merging (Chowdhury 2023) | semantic_scholar | Open access |
| Background Threads (Tang 2019) | scihub | ⚠️ Gray area |
| Parallel I/O (Tang 2022) | scihub | ⚠️ Gray area |
| Duplicate | cached | Already downloaded |

### Failed Downloads

| Paper | Reason |
|-------|--------|
| Tuning HDF5 subfiling performance on parallel file systems (2017) | No DOI, not found in any source |

## Validation

| Check | Status | Notes |
|-------|--------|-------|
| Command executed | ✅ | Completed |
| PDFs downloaded | ✅ | 4 unique papers |
| Metadata in filenames | ✅ | Author_Year_Title format |
| Institutional access | ⚠️ | Failed for these papers |

## Issues / Problems

| Issue | Severity | Description |
|-------|----------|-------------|
| Sci-Hub used | ⚠️ Warning | 2 papers retrieved via Sci-Hub (legal gray area) |
| Missing DOI | ❌ Error | 1 paper missing DOI, couldn't be found |
| Institutional failure | ⚠️ Warning | EZProxy didn't work for IEEE papers |

## Source Priority Used

1. Unpaywall (open access)
2. arXiv (preprints)
3. PubMed Central (biomedical)
4. bioRxiv (biology preprints)
5. Semantic Scholar ✅
6. ACL Anthology (NLP papers)
7. OpenAlex
8. Frontiers
9. Institutional (IIT EZProxy) ⚠️
10. Sci-Hub ⚠️
11. LibGen

## Next Step
→ Step 5: DOI to BibTeX conversion
