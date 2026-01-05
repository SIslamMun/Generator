# Jarvis QA Datasets

QA pairs generated from the Jarvis runtime deployment codebase and documentation using instruction backtranslation.

## Datasets

### Claude Dataset (Complete)
- **Location:** [claude/](claude/)
- **QA Pairs:** 300
- **Model:** Claude Sonnet 4 (claude-sonnet-4-20250514)
- **Status:** ✅ Complete
- **Generated:** 2026-01-05
- **Sources:** research_report.md, runtime-deployment GitHub repo, Jarvis README
- **Files:**
  - [full_claude.json](claude/full_claude.json) - JSON array format
  - [full_claude.jsonl](claude/full_claude.jsonl) - Line-delimited format
  - [full_claude_summary.json](claude/full_claude_summary.json) - Metadata
  - [README.md](claude/README.md) - Full documentation

### Gemini Dataset (Partial)
- **Location:** [full_gemini_intermediate.json](full_gemini_intermediate.json)
- **QA Pairs:** 150 (50% complete)
- **Model:** Gemini 2.0 Flash (gemini-2.0-flash-exp)
- **Status:** ⚠️ Incomplete - Daily quota exceeded at 150/300 pairs
- **Generated:** 2026-01-05
- **Note:** Checkpoint file from interrupted generation run

## Source Data

**LanceDB Path:** `/home/shazzadul/Illinois_Tech/Spring26/RA/Github/new/Phagocyte/pipeline_output/test_pipeline_jarvis/phase5_processor/lancedb`

**Statistics:**
- Total chunks: 144
- Valid chunks (≥200 chars): 100
- Expected QA pairs: 300 (3 per chunk)

## Generation Method

Uses instruction backtranslation technique where the LLM:
1. Reads code/documentation chunks from LanceDB
2. Generates diverse questions that could be answered by the chunk
3. Generates comprehensive answers using the chunk content
4. Creates instruction-response pairs suitable for fine-tuning

See [configs/prompts.yaml](../../configs/prompts.yaml) for the generation prompt template.
