# Generator - Synthetic QA Pair Generator

Generate high-quality QA pairs from LanceDB chunks for LLM fine-tuning. Implements **Instruction Backtranslation** (Meta AI, ICLR 2024) and **Response Rewriting** ([arxiv:2408.04614](https://arxiv.org/abs/2408.04614)).

## ✨ Features

**Pipeline**: Generate → Enrich → Curate → Export  
**CoT Support**: Generate or enhance with Chain-of-Thought reasoning  
**Providers**: Ollama, Claude, Gemini, vLLM, OpenAI, Anthropic  
**Formats**: ChatML, Alpaca, ShareGPT, JSONL  
**Rating**: LLM-as-Judge with detailed criteria (clarity, accuracy, usefulness, difficulty)

## 📦 Installation

```bash
uv pip install -e .              # Local only (Ollama, vLLM)
uv pip install -e ".[cloud]"     # + Cloud providers
uv pip install -e ".[all]"       # All providers
```

## 🚀 Quick Start

**1. Configure** `configs/config.yaml`:
```yaml
llm:
  provider: ollama
  model: mistral:latest
  base_url: http://localhost:11434
  temperature: 0.7
```

**2. Run Pipeline** (recommended):
```bash
uv run generator pipeline /path/to/lancedb -o training.jsonl
```

**3. Or Run Steps**:
```bash
# QA pairs
uv run generator generate /path/to/lancedb -o qa.json --target-pairs 300
uv run generator enrich qa.json -o enriched.json
uv run generator curate enriched.json -o curated.json --threshold 7.0
uv run generator export curated.json -o training.jsonl -f chatml

# Or generate CoT pairs
uv run generator generate-cot /path/to/lancedb -o cot.json --target-pairs 100
uv run generator curate cot.json -o cot_curated.json --threshold 7.0
uv run generator export cot_curated.json -o cot_training.jsonl -f chatml

# Or enhance existing QA with CoT reasoning
uv run generator enhance-cot qa.json -o cot_enhanced.json
```

## 📝 CLI Commands

### `list-providers`
List available LLM providers and setup instructions.

### `generate` - Generate QA pairs (Instruction Backtranslation)

```bash
uv run generator generate LANCEDB_PATH -o OUTPUT.json [OPTIONS]
```

**Options:**
- `--config PATH` - Config file (default: `configs/config.yaml`)
- `--table TEXT` - LanceDB table (default: `text_chunks`)
- `--n-pairs INT` - Fixed pairs per chunk
- `--target-pairs INT` - Total target pairs (auto-calculates per chunk) ⭐
- `--batch-size INT` - Chunks per batch (default: 50)
- `--max-chunks INT` - Limit chunks (for testing)
- `--provider TEXT` - Override provider from config
- `--model TEXT` - Override model from config

**Examples:**
```bash
# Recommended: Target-based generation
uv run generator generate /path/to/lancedb -o qa.json --target-pairs 300

# Test with limited data
uv run generator generate /path/to/lancedb -o qa.json --max-chunks 10 --target-pairs 50

# Override provider
uv run generator generate /path/to/lancedb -o qa.json --provider gemini --model gemini-2.0-flash-exp
```

**Output:** `[{"question": "...", "answer": "...", "chunk_id": "...", "source": "..."}]`

---

### `enrich` - Rewrite answers (Response Rewriting)

```bash
uv run generator enrich INPUT.json -o OUTPUT.json [OPTIONS]
```

**Options:**
- `--config PATH` - Config file
- `--provider TEXT` - Override provider
- `--model TEXT` - Override model
- `--batch-size INT` - Pairs per batch (default: 5)
- `--no-preserve-original` - Don't keep original answer

**Examples:**
```bash
uv run generator enrich qa.json -o enriched.json
uv run generator enrich qa.json -o enriched.json --batch-size 10 --provider claude
```

**Output:** Adds `enrichment_changes` and optionally `original_answer` fields

---

### `generate-cot` - Generate CoT pairs (Chain-of-Thought)

```bash
uv run generator generate-cot LANCEDB_PATH -o OUTPUT.json [OPTIONS]
```

**Options:**
- `--config PATH` - Config file
- `--table TEXT` - LanceDB table (default: `text_chunks`)
- `--n-pairs INT` - Fixed CoT pairs per chunk
- `--target-pairs INT` - Total target pairs (auto-calculates per chunk) ⭐
- `--batch-size INT` - Chunks per batch (default: 50)
- `--max-chunks INT` - Limit chunks (for testing)
- `--provider TEXT` - Override provider
- `--model TEXT` - Override model

**Examples:**
```bash
uv run generator generate-cot /path/to/lancedb -o cot.json --target-pairs 100
uv run generator generate-cot /path/to/lancedb -o cot.json --max-chunks 10
```

**Output:** `[{"question": "...", "reasoning": "Step 1: ...\nStep 2: ...", "answer": "...", "chunk_id": "...", "source": "..."}]`

---

### `enhance-cot` - Add reasoning to existing QA pairs

```bash
uv run generator enhance-cot INPUT.json -o OUTPUT.json [OPTIONS]
```

**Options:**
- `--config PATH` - Config file
- `--provider TEXT` - Override provider
- `--model TEXT` - Override model
- `--batch-size INT` - Pairs per batch (default: 5)

**Examples:**
```bash
uv run generator enhance-cot qa.json -o cot_enhanced.json
uv run generator enhance-cot qa.json -o cot_enhanced.json --batch-size 10 --provider claude
```

**Output:** Converts QA pairs to CoT format with reasoning field

---

### `curate` - Filter by quality (LLM-as-Judge)

```bash
uv run generator curate INPUT.json -o OUTPUT.json [OPTIONS]
```

**Options:**
- `--config PATH` - Config file
- `--threshold FLOAT` - Minimum rating 1-10 (default: 7.0)
- `--batch-size INT` - Pairs rated per call (default: 5)
- `--provider TEXT` - Override provider
- `--model TEXT` - Override model

**Format Support:** ✅ QA pairs, ✅ CoT examples, 🔄 Auto-detection

**Rating Criteria:** Clarity (0-3), Accuracy (0-3), Usefulness (0-2), Difficulty (0-2), Total (0-10)

**Examples:**
```bash
uv run generator curate qa.json -o curated.json
uv run generator curate qa.json -o curated.json --threshold 8.0  # High quality only
uv run generator curate cot.json -o cot_curated.json  # Works with CoT format too
```

**Output:** Adds `rating`, `clarity`, `accuracy`, `usefulness`, `difficulty`, `reasoning` fields

---

### `compare` - Compare multiple QA datasets (LLM Judge)

```bash
uv run generator compare DATASET1.json DATASET2.json ... -o REPORT.json [OPTIONS]
```

**Options:**
- `--config PATH` - Config file
- `--sample-size INT` - Samples to judge per dataset (default: 10)
- `--provider TEXT` - Override provider
- `--model TEXT` - Override model

**What it does:**
- Computes metrics for each dataset (count, avg rating, source diversity, question types)
- Samples random pairs from each dataset for LLM evaluation
- LLM judges quality (score 1-10, strengths, weaknesses)
- Recommends best dataset based on quality + diversity + size

**Examples:**
```bash
# Compare two QA datasets
uv run generator compare qa_v1.json qa_v2.json -o comparison.json

# Compare all curated outputs
uv run generator compare phase4_curate/*.json -o winner.json --sample-size 15
```

**Output:** JSON report with:
- Metrics for each dataset
- LLM quality judgments
- Recommended winner with reasoning
- Alternative suggestions (merge/hybrid)

---

### `export` - Convert to training format

```bash
uv run generator export INPUT.json -o OUTPUT [OPTIONS]
```

**Options:**
- `-f, --format CHOICE` - Format: `chatml`, `alpaca`, `sharegpt`, `jsonl` (default: `chatml`)
- `--system-prompt TEXT` - Custom system prompt

**Examples:**
```bash
uv run generator export curated.json -o training.jsonl -f chatml
uv run generator export curated.json -o training.json -f alpaca
uv run generator export curated.json -o training.jsonl -f chatml --system-prompt "You are an expert in HDF5."
```

**Output Formats:**
- **ChatML**: `{"messages": [{"role": "system/user/assistant", "content": "..."}]}`
- **Alpaca**: `{"instruction": "...", "input": "", "output": "..."}`
- **ShareGPT**: `{"conversations": [{"from": "human/gpt", "value": "..."}]}`

---

### `pipeline` - Run full pipeline

```bash
uv run generator pipeline LANCEDB_PATH -o OUTPUT [OPTIONS]
```

**Options:**
- `--config PATH` - Config file
- `--threshold FLOAT` - Curation threshold (default: 7.0)
- `-f, --format CHOICE` - Output format (default: `chatml`)
- `--max-chunks INT` - Limit chunks (for testing)
- `--skip-enrichment` - Skip enrichment step (faster, slightly lower quality)

**Examples:**
```bash
uv run generator pipeline /path/to/lancedb -o training.jsonl
uv run generator pipeline /path/to/lancedb -o training.jsonl --skip-enrichment
uv run generator pipeline /path/to/lancedb -o test.jsonl --max-chunks 10
uv run generator pipeline /path/to/lancedb -o training.json -f alpaca --threshold 8.0
```

**Pipeline Steps:** Generate (1/4) → Enrich (2/4) → Curate (3/4) → Export (4/4)
    "original_answer": "Use `uv run phagocyte ingest file <path>`",
    "chunk_id": "chunk_123"
  }
]
```

---

### curate

Filter QA pairs or CoT examples by quality using **LLM-as-Judge** with detailed rating criteria.

**Format Support:**
- ✅ QA pairs: `{"question": "...", "answer": "..."}`
- ✅ CoT examples: `{"question": "...", "reasoning": "...", "answer": "..."}`
- 🔄 Automatic format detection and restoration

```bash
uv run generator curate INPUT.json -o OUTPUT.json [OPTIONS]

Options:
## ⚙️ Provider Setup

<details>
<summary><b>Ollama (Local)</b></summary>

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:latest
```
</details>

<details>
<summary><b>Claude / Gemini / OpenAI</b></summary>

```bash
uv pip install ".[cloud]"

# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="your-key"
export OPENAI_API_KEY="sk-..."
```
Get keys: [Claude](https://console.anthropic.com/) | [Gemini](https://aistudio.google.com/apikey) | [OpenAI](https://platform.openai.com/api-keys)
</details>

## 🔬 Methodology

**Instruction Backtranslation** ([arxiv:2308.06259](https://arxiv.org/abs/2308.06259))  
Treats documents as answers, generates questions. More scalable than manual annotation, better long-tail coverage.

**Response Rewriting** ([arxiv:2408.04614](https://arxiv.org/abs/2408.04614))  
Rewrites answers for better clarity/structure while preserving all information. Improves alignment without adding hallucinated facts.

**Chain-of-Thought** ([arxiv:2305.02301](https://arxiv.org/abs/2305.02301))  
Generates or adds step-by-step reasoning to QA pairs. Enables smaller models to learn complex reasoning patterns from larger models.

**LLM-as-Judge**  
Multi-criteria rating (clarity, accuracy, usefulness, difficulty) with reasoning explanations. Threshold filtering for high-quality training data.

## 📊 Example Results

**Jarvis Dataset** ([output/jarvis_qa/](output/jarvis_qa/))  
300 raw → 226 curated (75.3% retention) | Avg rating: 6.06/10 | Model: Claude Sonnet 4

**HDF5 Dataset** ([output/hdf5_qa/gemini/](output/hdf5_qa/gemini/))  
876 QA pairs | Model: Gemini 2.0 Flash | Source: HDF5 research papers

## 🧪 Testing

```bash
uv run pytest tests/ -v                    # 11/11 tests passing ✅
uv run ruff check src/                     # All checks passed ✅
```

## 📚 Documentation

- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) - Detailed status & planned features
- [CHANGES.md](CHANGES.md) - Recent updates & implementation notes
- [configs/prompts/](configs/prompts/) - Prompt templates

## 🛣️ Roadmap

**✅ Phase 1** - Core QA pipeline (generate, enrich, curate, export)  
**✅ Phase 2** - CoT generation & enhancement, format detection/restoration  
**🔄 Phase 3** - Tool use support, multi-turn conversations  
**🔮 Phase 4** - Parallel processing, caching, web UI

---

**Research-backed synthetic data generation for LLM fine-tuning** | MIT License
