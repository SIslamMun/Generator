# Jarvis QA Dataset - Claude Generated

## Overview

High-quality question-answer pairs generated from JARVIS I/O framework documentation using Claude Sonnet 4.

## Dataset Statistics

- **Total QA Pairs**: 300
- **Model**: Claude Sonnet 4 (claude-sonnet-4-20250514)
- **Generated**: January 5, 2026
- **Method**: Instruction Backtranslation
- **QA Pairs per Chunk**: 3
- **Valid Chunks Used**: 100 (≥200 chars)

## Source Documents

The QA pairs were generated from JARVIS documentation:

1. **research_report.md** - JARVIS research findings and technical analysis
2. **github_com_iowarp_runtime-deployment.md** - Runtime deployment documentation
3. **README.md** - JARVIS project overview and usage

## Files

### Full Datasets
- `full_claude.json` - Complete dataset in JSON array format (300 pairs)
- `full_claude.jsonl` - Line-delimited JSON format (one pair per line)
- `full_claude_summary.json` - Metadata and statistics

### Intermediate Files
- `full_claude_intermediate.json` - Checkpoint during generation (auto-saved)

## Sample QA Pair

```json
{
  "question": "What is the primary purpose of the JARVIS I/O framework?",
  "answer": "JARVIS is designed to provide high-performance I/O capabilities for HPC applications by intercepting and optimizing I/O operations at the system level.",
  "chunk_id": "...",
  "source": "README.md"
}
```

## Generation Details

**Command:**
```bash
uv run generator generate \
  /path/to/jarvis/lancedb \
  -o output/jarvis_qa/claude/full_claude.json \
  --provider claude \
  --n-pairs 3 \
  --batch-size 50
```

**Configuration:**
- Provider: Claude (Anthropic)
- Model: claude-sonnet-4-20250514
- Temperature: 0.7
- Rate limiting: 6 seconds between chunks
- Retry logic: Exponential backoff (60s, 120s, 240s)

## Quality Notes

- All pairs generated from chunks ≥200 characters
- Questions focus on important technical concepts
- Answers are directly supported by source text
- Diverse question types (what, how, why, when)

## Next Steps

1. **Curation**: Filter by quality using LLM-as-Judge
   ```bash
   uv run generator curate full_claude.json -o curated_claude.json --threshold 7.0
   ```

2. **Export**: Convert to training format
   ```bash
   uv run generator export curated_claude.json -o training.jsonl -f chatml
   ```

3. **Comparison**: Compare with Gemini-generated pairs for diversity analysis

## Related Datasets

- **HDF5 Dataset (Gemini)**: `output/hdf5_qa/gemini/` (876 pairs, Gemini 2.0 Flash)
- **Jarvis Dataset (Gemini)**: `output/jarvis_qa/full_gemini_intermediate.json` (150 pairs, partial, Gemini 2.0 Flash)

---

**Paper Reference**: Self-Alignment with Instruction Backtranslation (Meta AI, ICLR 2024)  
**GitHub**: https://github.com/your-repo/Generator
