# Phagocyte Pipeline Test Run: HDF5 Research

> End-to-end workflow: Research → Parse References → Acquire Documents → Ingest to Knowledge Store

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PHAGOCYTE PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │  RESEARCHER │────▶│   PARSER    │────▶│  INGESTOR   │────▶│ PROCESSOR │ │
│  │             │     │             │     │             │     │           │ │
│  │ • Research  │     │ • Parse refs│     │ • PDF→MD    │     │ • Chunk   │ │
│  │ • Citations │     │ • Acquire   │     │ • Web→MD    │     │ • Embed   │ │
│  │ • Report    │     │ • DOI→BIB   │     │ • Git→MD    │     │ • Store   │ │
│  └─────────────┘     └─────────────┘     └─────────────┘     └───────────┘ │
│                                                                             │
│  Input: Topic        Output: Papers      Output: Markdown   Output: Vector  │
│         + Artifacts          + BibTeX            + Images          Database │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Test Run Configuration

| Parameter | Value |
|-----------|-------|
| **Topic** | HDF5 file format: architecture, best practices, and advanced usage patterns |
| **Primary Artifacts** | https://support.hdfgroup.org/documentation/, https://github.com/HDFGroup/hdf5 |
| **Date** | January 1, 2026 |
| **Duration** | ~15 minutes total |

---

## Phase Summary

| Phase | Module | Status | Duration | Key Output |
|-------|--------|--------|----------|------------|
| 1 | [researcher](phase1_research/README.md) | ✅ | 374s | Research report with 45+ citations |
| 2.1 | [parser](phase2_parser/step1_regular/README.md) | ✅ | <1s | 46 references (regex) |
| 2.2 | [parser](phase2_parser/step2_agent/README.md) | ✅ | ~5s | 42 references (Claude) |
| 2.3 | [parser](phase2_parser/step3_comparison/README.md) | ✅ | ~5s | Comparison report |
| 2.4 | [parser](phase2_parser/step4_acquisition/README.md) | ⚠️ | ~60s | 5/7 papers downloaded |
| 2.5 | [parser](phase2_parser/step5_doi2bib/README.md) | ✅ | <1s | 4 BibTeX entries |
| 3 | [ingestor](phase3_ingestor/README.md) | ✅ | ~210s | 4 PDFs + 1 repo + 40 pages |
| 4 | [processor](phase4_processor/README.md) | ✅ | ~120s | 1,547 chunks in LanceDB |

---

## Phase 1: Research (Researcher Module)

### Command
```bash
cd researcher && unset GOOGLE_API_KEY && uv run researcher research \
  "HDF5 file format: architecture, best practices, and advanced usage patterns" \
  --mode directed \
  -a "https://support.hdfgroup.org/documentation/" \
  -a "https://github.com/HDFGroup/hdf5" \
  -o ../pipeline_output/phase1_research -v
```

### Results
- **Duration**: 374.3 seconds
- **Report**: 17KB comprehensive research report
- **Citations**: 45+ structured references (DOIs, arXiv, websites)
- **Topics covered**: Architecture, best practices, SWMR, VDS, Subfiling, Async I/O

### Output Files
- `research/research_report.md` - Main research report
- `research/research_metadata.json` - Metadata
- `research/thinking_steps.md` - Agent reasoning

---

## Phase 2: Parser (Parser Module)

### Step 1: Regular Parsing
```bash
parser parse-refs research_report.md -o step1_regular --format both --export-batch --export-dois
```
**Result**: 46 references (4 DOIs, 4 papers, 12 PDFs, 26 websites)

### Step 2: Agent Parsing (Claude)
```bash
parser parse-refs research_report.md -o step2_agent --agent claude --format both --export-batch --export-dois
```
**Result**: 42 references (3 DOIs, 3 papers, 12 PDFs, 24 websites)

### Step 3: Comparison
```bash
parser parse-refs research_report.md -o step3_comparison --compare --agent claude
```
**Result**: 40 common, 6 only in regular, 3 only in agent

### Step 4: Acquisition
```bash
parser batch batch.json -o step4_acquisition/papers -v
```
**Result**: 5 downloaded, 1 cached, 1 failed

| Paper | Source |
|-------|--------|
| Byna 2020 (ExaHDF5) | Semantic Scholar |
| Chowdhury 2023 (Async I/O) | Semantic Scholar |
| Tang 2019 (Background Threads) | Sci-Hub ⚠️ |
| Tang 2022 (Parallel I/O) | Sci-Hub ⚠️ |
| (Failed) Subfiling 2017 | No DOI |

### Step 5: DOI to BibTeX
```bash
parser doi2bib -i dois.txt -o references.bib
```
**Result**: 4 BibTeX entries generated and verified

---

## Phase 3: Ingestor (Ingestor Module)

### PDF Ingestion
```bash
ingestor batch papers/ -o pdfs --recursive -v
```
**Result**: 4 PDFs converted to markdown with figures

### GitHub Repository
```bash
ingestor ingest "https://github.com/HDFGroup/hdf5" -o github -v
```
**Result**: Repository content extracted

### Web Documentation
```bash
ingestor crawl "https://support.hdfgroup.org/documentation/" -o web --max-depth 2 --max-pages 20 -v
```
**Result**: 40 pages crawled (37 successful, 3 timeouts)

---

## Phase 4: Processor (Processor Module)

### Document Processing
```bash
processor process ./input -o ./lancedb --text-profile low --code-profile low --table-mode separate
```
**Result**: 53 files processed, 1,547 chunks created

### Database Statistics
| Table | Rows | Description |
|-------|------|-------------|
| text_chunks | 385 | Research, papers, websites |
| code_chunks | 1162 | HDF5 codebase |

### Sample Search Results

| Query | Best Match | Distance | Source |
|-------|------------|----------|--------|
| "asynchronous IO HDF5" | HDF5 Library and Tools 2.0.0 | 0.42 | websites |
| "virtual dataset VDS" | VDS allows dataset composition... | 0.61 | research |
| "SWMR single writer multiple reader" | SWMR enables concurrent read/write... | 0.59 | research |
| "background thread asynchronous write" | Transparent Async Parallel I/O... | 0.55 | papers |
| "parallel MPI IO collective" | ExaHDF5: Delivering Efficient... | 0.67 | papers |
| "H5Fcreate H5Dwrite" (code) | dcpl_id = H5I_INVALID_HID()... | 22510 | codebases |

### Search Commands
```bash
# Text search
processor search ./lancedb "SWMR single writer" --table text_chunks -k 5

# Code search  
processor search ./lancedb "H5Fcreate" --table code_chunks -k 5

# Hybrid search (vector + BM25)
processor search ./lancedb "parallel IO" --table text_chunks --hybrid
```

---

## Validation Summary

### ✅ Successful

| Component | Validation |
|-----------|------------|
| Research report | Generated with 45+ citations |
| Regular parsing | 46 references extracted |
| Agent parsing | 42 references extracted |
| Comparison | Report generated with overlap analysis |
| Paper downloads | 5/7 papers retrieved |
| BibTeX generation | 4 entries created |
| BibTeX verification | 4/4 verified via CrossRef |
| PDF ingestion | 4/4 papers converted |
| GitHub ingestion | Repository extracted |
| Web crawling | 37/40 pages successful |
| RAG processing | 1,547 chunks embedded |
| Vector search | 6/6 queries return relevant results |

### ⚠️ Issues Encountered

| Phase | Issue | Severity | Details |
|-------|-------|----------|---------|
| 2.4 | Missing DOI | ❌ Error | "Tuning HDF5 subfiling" paper couldn't be found |
| 2.4 | Institutional access | ⚠️ Warning | EZProxy failed, fell back to Sci-Hub |
| 2.4 | Sci-Hub used | ⚠️ Warning | 2 papers retrieved via gray-area source |
| 3 | Web timeouts | ⚠️ Warning | 3 pages timed out (forum, main site) |
| 3 | OCR warnings | ℹ️ Info | Empty OCR results for text-based PDFs (normal) |
| 4 | Chunk truncation | ℹ️ Info | 1 oversized chunk truncated to 8000 chars |

---

## Output Directory Structure

```
pipeline_output/
├── README.md                          # This file
│
├── phase1_research/                   # RESEARCHER OUTPUT
│   ├── README.md
│   ├── execution.log
│   ├── research_stream.md
│   └── research/
│       ├── research_report.md         # Main research report
│       ├── research_metadata.json
│       └── thinking_steps.md
│
├── phase2_parser/                     # PARSER OUTPUT
│   ├── README.md
│   ├── step1_regular/                 # Regex parsing
│   │   ├── README.md
│   │   ├── references.json
│   │   ├── references.md
│   │   ├── batch.json
│   │   └── dois.txt
│   ├── step2_agent/                   # Claude parsing
│   │   ├── README.md
│   │   ├── references.json
│   │   ├── agent_raw_response.txt
│   │   └── ...
│   ├── step3_comparison/              # Comparison
│   │   ├── README.md
│   │   ├── comparison_report.md
│   │   ├── regular/
│   │   └── agent/
│   ├── step4_acquisition/             # Paper downloads
│   │   ├── README.md
│   │   └── papers/
│   │       ├── *.pdf                  # Downloaded papers
│   │       └── failed/
│   └── step5_doi2bib/                 # BibTeX
│       ├── README.md
│       ├── references.bib
│       └── verified/
│
└── phase3_ingestor/                   # INGESTOR OUTPUT
    ├── README.md
    ├── pdfs/                          # PDF → Markdown
    │   ├── Byna_2020_.../
    │   │   ├── *.md
    │   │   └── img/
    │   └── ...
    ├── github/                        # GitHub → Markdown
    │   └── github_com_HDFGroup_hdf5/
    └── web/                           # Web → Markdown
        └── (40 page directories)

└── phase4_processor/                  # PROCESSOR OUTPUT
    ├── README.md
    ├── process_execution.log
    ├── input/                         # Organized input files
    │   ├── research/                  # Research reports
    │   ├── papers/                    # Converted papers
    │   ├── websites/                  # Web content
    │   └── codebases/                 # GitHub repos
    └── lancedb/                       # Vector Database
        ├── text_chunks.lance/         # Text embeddings
        ├── code_chunks.lance/         # Code embeddings
        └── _metadata.lance/           # DB metadata
```

---

## Full Command Sequence

```bash
# ============================================
# PHASE 1: RESEARCH
# ============================================
cd researcher
unset GOOGLE_API_KEY
uv run researcher research \
  "HDF5 file format: architecture, best practices, and advanced usage patterns" \
  --mode directed \
  -a "https://support.hdfgroup.org/documentation/" \
  -a "https://github.com/HDFGroup/hdf5" \
  -o ../pipeline_output/phase1_research -v

# ============================================
# PHASE 2: PARSER
# ============================================
cd ../parser

# Authenticate for institutional access
uv run parser auth

# Step 1: Regular parsing
uv run parser parse-refs \
  ../pipeline_output/phase1_research/research/research_report.md \
  -o ../pipeline_output/phase2_parser/step1_regular \
  --format both --export-batch --export-dois

# Step 2: Agent parsing
uv run parser parse-refs \
  ../pipeline_output/phase1_research/research/research_report.md \
  -o ../pipeline_output/phase2_parser/step2_agent \
  --agent claude --format both --export-batch --export-dois

# Step 3: Comparison
uv run parser parse-refs \
  ../pipeline_output/phase1_research/research/research_report.md \
  -o ../pipeline_output/phase2_parser/step3_comparison \
  --compare --agent claude

# Step 4: Acquisition
uv run parser batch \
  ../pipeline_output/phase2_parser/step1_regular/batch.json \
  -o ../pipeline_output/phase2_parser/step4_acquisition/papers -v

# Step 5: DOI to BibTeX
uv run parser doi2bib \
  -i ../pipeline_output/phase2_parser/step1_regular/dois.txt \
  -o ../pipeline_output/phase2_parser/step5_doi2bib/references.bib

# Verify BibTeX
uv run parser verify \
  ../pipeline_output/phase2_parser/step5_doi2bib/references.bib \
  -o ../pipeline_output/phase2_parser/step5_doi2bib/verified -v

# ============================================
# PHASE 3: INGESTOR
# ============================================
cd ../ingestor

# PDF ingestion
uv run ingestor batch \
  ../pipeline_output/phase2_parser/step4_acquisition/papers/ \
  -o ../pipeline_output/phase3_ingestor/pdfs --recursive -v

# GitHub ingestion
uv run ingestor ingest \
  "https://github.com/HDFGroup/hdf5" \
  -o ../pipeline_output/phase3_ingestor/github -v

# Web crawling
uv run ingestor crawl \
  "https://support.hdfgroup.org/documentation/" \
  -o ../pipeline_output/phase3_ingestor/web \
  --max-depth 2 --max-pages 20 -v

# ============================================
# PHASE 4: PROCESSOR
# ============================================
cd ../processor

# Organize input documents
mkdir -p ../pipeline_output/phase4_processor/input/{research,papers,websites,codebases}
cp ../pipeline_output/phase1_research/research/*.md ../pipeline_output/phase4_processor/input/research/
cp ../pipeline_output/phase3_ingestor/pdfs/*/*.md ../pipeline_output/phase4_processor/input/papers/
cp ../pipeline_output/phase3_ingestor/web/*/*.md ../pipeline_output/phase4_processor/input/websites/
cp -r ../pipeline_output/phase3_ingestor/github/* ../pipeline_output/phase4_processor/input/codebases/

# Check Ollama is running
uv run processor check

# Process documents into vector database
uv run processor process \
  ../pipeline_output/phase4_processor/input \
  -o ../pipeline_output/phase4_processor/lancedb \
  --text-profile low \
  --code-profile low \
  --table-mode separate \
  --full

# View database stats
uv run processor stats ../pipeline_output/phase4_processor/lancedb

# Test search
uv run processor search \
  ../pipeline_output/phase4_processor/lancedb \
  "asynchronous IO HDF5" \
  --table text_chunks -k 5
```

---

## Recommendations for Production

1. **API Keys**: Set `SEMANTIC_SCHOLAR_API_KEY` for higher rate limits
2. **Institutional Access**: Ensure VPN connection for paywalled papers
3. **Parallel Processing**: Increase `--concurrent` for batch downloads
4. **Web Crawling**: Increase timeout for slower sites
5. **GPU Acceleration**: Use CUDA for faster PDF processing

---

## License

MIT - Phagocyte Pipeline
