# Generator: Design and Implementation Overview

**Author:** Shazzadul  
**TL;DR:** Comprehensive synthetic training data generator based on 13 research papers (Dec 2022 - Dec 2025), with full documentation of design decisions and extraction methodology.

---

## What Was Built

A production-ready system for generating high-quality synthetic training data for LLM fine-tuning, with two main pipelines:

1. **QA Pipeline** - Teaches domain knowledge from documents
2. **Tool-Use Pipeline** - Teaches agentic capabilities (API calling)

**Key Updates (Jan 2026):** Added 6 new paper implementations for advanced data selection and tool-use generation.

---

## Research Foundation: Which Papers & What Was Extracted

### 1. **LIMA** (Meta AI, NeurIPS 2023)
**What:** 1K carefully curated examples > 100K low-quality examples

**What We Extracted:**
- Strict quality thresholds (≥7/10 default)
- Multi-dimensional rating (Clarity, Accuracy, Usefulness, Difficulty)
- LLM-as-Judge methodology
- Quality-over-quantity philosophy throughout

**Where:** `src/generator/qa/curate.py`, `configs/prompts/qa_rating.yaml`

---

### 2. **Instruction Backtranslation** (Meta AI, ICLR 2024)
**What:** Treat documents as "answers," generate "questions" for them

**What We Extracted:**
- Core QA generation methodology
- Documents → Chunks → Questions (treat chunks as answers)
- Preserves expert knowledge without hallucination
- Target-based generation (specify total pairs, auto-calculate per-chunk)

**Where:** `src/generator/qa/qa_generator.py`, `configs/prompts/qa_generation.yaml`

**Why This Approach:**
- Documents already contain expert knowledge
- Easier to generate good questions than good answers
- Grounded in real documents (no fabrication)

---

### 3. **Distilling Step-by-Step** (Google, 2023)
**What:** Extract Chain-of-Thought (CoT) rationales to enable small models to match or exceed large model reasoning

**Key Insight:** Training with intermediate reasoning steps allows small models to outperform much larger models with 12.5× less data

**What We Extracted:**
- **Two CoT approaches:**
  1. **Direct generation** - Create QA with built-in reasoning from scratch (`cot_generator.py`)
  2. **Enhancement** - Add reasoning to existing curated QA pairs (`cot_enhancer.py`)
- **Reasoning structure:** 2-5 logical steps that build progressively toward answer
- **Quality validation:** Ensures reasoning is coherent, non-circular, and supports the answer
- **Training efficiency:** Small models (7-13B) can perform complex reasoning comparable to 70B+ models

**Why This Matters:**
- Traditional QA teaches models to memorize patterns (Q→A)
- CoT teaches models *how* to reason (Q→Steps→A)
- Results in better generalization, interpretability, and performance on complex tasks

**Implementation Details:**
- **Direct Generation:** Generates reasoning-focused questions from documents
  - Use when starting fresh or need questions designed for multi-step thinking
  - CLI: `uv run generator generate-cot <lancedb> -o output.json`
  
- **Enhancement:** Adds reasoning to existing high-quality QA pairs
  - Use when you already have curated QA and want to add reasoning efficiently
  - More cost-effective than regenerating everything
  - CLI: `uv run generator enhance-cot qa_pairs.json -o cot_pairs.json`

**Where:** 
- Files: `src/generator/cot/cot_generator.py`, `src/generator/cot/cot_enhancer.py`
- Prompts: `configs/prompts/cot_generation.yaml`, `configs/prompts/cot_enhancement.yaml`
- Tests: `tests/test_cot_generator.py`, `tests/test_cot_enhancer.py`

**Example Output:**
```json
{
  "question": "Why would HDF5 be preferred over CSV for TB-scale datasets?",
  "reasoning": [
    "CSV requires loading entire file into memory, impossible for TB-scale data",
    "HDF5 uses chunked storage allowing partial reads of specific regions",
    "HDF5 includes built-in compression reducing storage by 5-10×",
    "HDF5 supports hierarchical organization for complex multi-dimensional data"
  ],
  "answer": "HDF5 is preferred because it enables efficient partial I/O through chunking, provides compression, and supports complex data organization—capabilities CSV lacks."
}

---

### 4. **AlpaGasus** (UMD, ICLR 2024)
**What:** LLM-as-Judge with detailed criteria beats automated filtering

**What We Extracted:**
- Per-criteria breakdown scoring
- Reasoning explanations for each rating
- Format detection (QA vs CoT) with preservation

**Where:** Integrated into `src/generator/qa/curate.py`

---

### 5. **Toolformer** (Meta AI, NeurIPS 2023)
**What:** Self-supervised tool learning with single-step calls

**What We Extracted:**
- Single-step tool calling mode
- Simple instruction → single tool → result workflow

**Where:** `src/generator/tool/tool_generator.py` (single-step mode)

---

### 6. **Gorilla** (UC Berkeley, NeurIPS 2024)
**What:** Always include API documentation to reduce hallucination <7%

**What We Extracted:**
- Every tool example includes full API documentation
- Documentation grounding prevents invented APIs
- `api_documentation` field in all tool examples

**Where:** `src/generator/tool/tool_generator.py` (auto-included in all examples)

---

### 7. **ToolLLM** (Tsinghua, ICLR 2024)
**What:** DFSDT (Depth-First Search Decision Tree) for multi-step reasoning

**What We Extracted:**
- Multi-step reasoning path structure
- Step-by-step: thought → action → result → next step
- Tool chaining with 2-4 tools in sequence

**Where:** `src/generator/tool/tool_generator.py` (multi-step mode)

---

## How Data Extraction Works

### QA Pipeline (Knowledge)

```
Documents (PDFs, GitHub, Docs)
    ↓ [Phagocyte pre-processing]
Clean Markdown
    ↓ [Semantic chunking]
LanceDB Chunks (512-1024 tokens each)
    ↓ [Instruction Backtranslation]
QA Pairs (questions generated from chunks)
    ↓ [Response rewriting]
Enriched QA (better formatting)
    ↓ [LLM-as-Judge curation]
Filtered QA (rating ≥7/10)
    ↓ [Optional: Add CoT]
QA with Reasoning
    ↓ [Format conversion]
Training File (ChatML/Alpaca/ShareGPT)
```

**Key Design Decisions:**
1. **Target-based generation** - User specifies total pairs (e.g., 300), system calculates per-chunk
2. **Batch processing with checkpoints** - Prevents data loss, enables resume
3. **Code-specific prompts** - Auto-detects `code_chunks` table and uses optimized prompt
4. **Topic filtering** - Available in `curate` command for semantic relevance filtering
5. **Multiple formats** - Export to ChatML, Alpaca, ShareGPT, JSONL

### Tool-Use Pipeline (Agentic)

```
API Specs (OpenAPI, JSON)
    ↓ [Parse tools]
Tool Definitions
    ↓ [Generate examples]
    ├─ Single-step (Toolformer)
    ├─ Multi-step (ToolLLM)
    └─ Auto (balanced mix)
Tool Examples (with API docs - Gorilla)
    ↓ [Execution verification]
Verified Examples (3 layers: format, execution, semantic)
    ↓ [Curation]
Filtered Examples
    ↓ [Export]
Training File
```

**Key Innovation:** Every example includes full API documentation (Gorilla) to prevent hallucination

---

## Implementation Highlights

### Multi-Provider LLM Support

**Supported:**
- Ollama (local, free) ⭐
- Claude (cloud, paid)
- Gemini (cloud, free tier 10 req/min) ⭐
- vLLM (local server, free)
- OpenAI (cloud, paid)
- Anthropic (cloud, paid)

**Easy switching:**
```yaml
# configs/config.yaml
llm:
  provider: gemini  # Change this line
```

Or CLI override:
```bash
uv run generator generate db/ -o qa.json --provider claude
```

### Rate Limiting & Retry Logic

- Exponential backoff for rate limits
- Provider-specific handling (e.g., Gemini 10 req/min)
- Automatic retry on transient errors

### Quality Assurance

**Three-Stage Filtering:**
1. **Pre-filtering** - Basic validation during generation
2. **LLM-as-Judge** - Detailed rating with criteria
3. **Format validation** - Ensure proper structure

**For Tool-Use (APIGen's Triple Verification):**
1. **Format** - Valid JSON, required fields present
2. **Execution** - Tools actually work (simulated or real)
3. **Semantic** - Solution addresses instruction

---

## Data Generated So Far

**HDF5 Dataset:**
- Source: HDF5 documentation and papers
- Generated: 876 QA pairs (Gemini 2.0 Flash)
- Location: `output/hdf5_qa/gemini/`

**Jarvis Dataset:**
- Source: Phagocyte project documentation
- Generated: 300 QA pairs (Claude Sonnet 4)
- Location: `output/jarvis_qa/claude/`

---

## Documentation Structure

### Main Docs (Newly Created)

1. **[DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md)** (12,000+ words)
   - Complete design rationale for every major decision
   - Research foundation (all 7 papers explained)
   - Architecture patterns and design philosophy
   - Why each approach was chosen

2. **[EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md)** (10,000+ words)
   - Step-by-step explanation of data flow
   - How extraction works from documents to training data
   - Code examples for each stage
   - Format conversion and export process

3. **[PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md)** (5,000+ words)
   - Quick reference: Paper → Code mapping
   - What was extracted from each paper
   - Code examples showing implementation
   - CLI commands for each feature

4. **[README.md](README.md)** (Updated)
   - Quick start guide
   - Command reference
   - Links to comprehensive docs

---

## Example Usage

### Generate Training Data (Full Pipeline)

```bash
# QA Pipeline (knowledge)
uv run generator pipeline /path/to/lancedb -o training.jsonl

# Tool-Use Pipeline (agentic)
uv run generator tool-pipeline configs/hdf5_tools.json -o tool_training.json
```

### Or Run Individual Steps

```bash
# 1. Generate QA pairs
uv run generator generate /path/to/lancedb -o qa.json --target-pairs 300

# 2. Enrich responses
uv run generator enrich qa.json -o enriched.json

# 3. Curate by quality
uv run generator curate enriched.json -o curated.json --threshold 7.0

# 4. Optional: Add CoT
uv run generator enhance-cot curated.json -o cot.json

# 5. Export to training format
uv run generator export cot.json -o training.jsonl -f chatml
```

---

## Key Design Principles (from Papers)

### Quality Over Quantity (LIMA)
- Strict thresholds (≥7/10)
- Detailed rating criteria
- Only keep best examples

### Ground Truth Preservation (Instruction Backtranslation)
- Documents as source of truth
- Generate questions, not answers
- No hallucination

### Reasoning Support (Distilling Step-by-Step)
- Step-by-step explanations
- Enable small models to reason
- Progressive learning

### Documentation Grounding (Gorilla)
- Include API docs in every example
- Prevent tool hallucination
- <7% hallucination rate

### Progressive Complexity (ToolLLM)
- Single-step for simple operations
- Multi-step for complex workflows
- Auto-balance based on tool complexity

---

## Testing & Quality

**Comprehensive Test Suite:**
- Unit tests for all modules: `tests/test_*.py`
- Integration tests for pipelines
- Client tests for all providers
- Coverage reports in `htmlcov/`

**Run tests:**
```bash
pytest tests/
```

---

## Next Steps

**Immediate:**
1. ✅ Comprehensive documentation (DONE)
2. ⏳ Fine-tune a model with generated data
3. ⏳ Evaluate: Compare fine-tuned vs base model + RAG

**Future Enhancements:**
1. Add iterative refinement (Self-Instruct)
2. Implement active learning for selective generation
3. Add evaluation metrics (BLEU, ROUGE, BERTScore)
4. Data deduplication and diversity analysis

---

## Quick File Reference

### Core Modules
- `src/generator/qa_generator.py` - QA generation (Instruction Backtranslation)
- `src/generator/cot_generator.py` - CoT generation (Distilling Step-by-Step)
- `src/generator/cot_enhancer.py` - Add CoT to existing QA
- `src/generator/curate.py` - LLM-as-Judge filtering (LIMA + AlpaGasus)
- `src/generator/enrich.py` - Response improvement
- `src/generator/tool_generator.py` - Tool-use examples (Toolformer + ToolLLM + Gorilla + ToolGrad)
- `src/generator/tool_executor.py` - Verification (APIGen)
- `src/generator/tool_curator.py` - Tool curation + turn-level filtering (ToolMind)
- `src/generator/multi_scorer.py` - Multi-dimensional scoring (DEITA) ⭐ NEW
- `src/generator/coverage_selector.py` - Semantic deduplication (TOUCAN) ⭐ NEW
- `src/generator/dependency_graph.py` - Parameter dependency graphs (In-N-Out) ⭐ NEW
- `src/generator/outcome_evaluator.py` - Outcome-oriented evaluation (MCP-AgentBench) ⭐ NEW

### Configuration
- `configs/config.yaml` - LLM provider settings
- `configs/prompts/*.yaml` - Prompt templates

### Documentation
- `DESIGN_DOCUMENTATION.md` - Complete design doc
- `EXTRACTION_METHODOLOGY.md` - How extraction works
- `PAPER_IMPLEMENTATIONS.md` - Paper → code mapping

---

## Summary

**What:** Production-ready synthetic training data generator

**Based On:** 13 research papers (LIMA, Instruction Backtranslation, Distilling Step-by-Step, AlpaGasus, Toolformer, Gorilla, ToolLLM, DEITA, TOUCAN, ToolMind, ToolGrad, In-N-Out, MCP-AgentBench)

**Capabilities:**
- Generate QA pairs from documents (knowledge)
- Generate tool-use examples from APIs (agentic)
- Multi-dimensional scoring for optimal selection (DEITA)
- Semantic deduplication for diversity (TOUCAN)
- Turn-level filtering for quality (ToolMind)
- Chain-first generation for coherent tool chains (ToolGrad)
- Parameter dependency graphs for tool orchestration (In-N-Out)
- Outcome-oriented evaluation (MCP-AgentBench)
- Multi-provider LLM support (6 providers)
- Quality assurance (LLM-as-Judge with detailed criteria)
- Multiple export formats (ChatML, Alpaca, ShareGPT, JSONL)

**Documentation:** 27,000+ words across 3 comprehensive docs

**Status:** Fully functional, all 13 papers implemented, 91 new tests added (Jan 2026)

---

**Questions?** See comprehensive docs:
- Design decisions → `DESIGN_DOCUMENTATION.md`
- How extraction works → `EXTRACTION_METHODOLOGY.md`
- Paper implementations → `PAPER_IMPLEMENTATIONS.md`
