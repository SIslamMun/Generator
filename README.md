# Generator - Synthetic Training Data Generator

Generate high-quality training data for LLM fine-tuning: QA pairs from documents and tool-use examples from API definitions.


## ✨ Features

### QA Pipeline (Knowledge Training)
**Pipeline**: Generate → Enrich → Curate → Export  
**CoT Support**: Generate or enhance with Chain-of-Thought reasoning  
**Formats**: ChatML, Alpaca, ShareGPT, JSONL  

### Tool-Use Pipeline (Agentic Training)
**Pipeline**: Parse → Generate → Execute → Curate  
**Modes**: Single-step, Multi-step, or Auto (complexity-based)  
**Output**: Instruction → Reasoning → Tool Calls with API documentation  

### Common
**Providers**: Ollama, Claude, Gemini, vLLM, OpenAI, Anthropic  
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

**2. Run QA Pipeline** (for domain knowledge):
```bash
uv run generator pipeline /path/to/lancedb -o training.jsonl
```

**3. Run Tool-Use Pipeline** (for agentic capabilities):
```bash
uv run generator tool-pipeline configs/hdf5_tools.json -o tool_training.json
```

**4. Or Run Steps Individually**:
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

## 📝 QA Pipeline Commands

### `list-providers`
List available LLM providers and setup instructions.

### `generate` - Generate QA pairs (Instruction Backtranslation)

```bash
uv run generator generate LANCEDB_PATH -o OUTPUT.json [OPTIONS]
```

**Options:**
- `--config PATH` - Config file (default: `configs/config.yaml`)
- `--table TEXT` - LanceDB table (default: `text_chunks`) - **can specify multiple times for unified output** ⭐
- `--n-pairs INT` - Fixed pairs per chunk
- `--target-pairs INT` - Total target pairs (auto-calculates per chunk) ⭐
- `--batch-size INT` - Chunks per batch (default: 50)
- `--max-chunks INT` - Limit chunks (for testing)
- `--topic TEXT` - Topic filter (e.g., 'HDF5') - removes off-topic pairs after generation ⭐ NEW
- `--provider TEXT` - Override provider from config
- `--model TEXT` - Override model from config

**Examples:**
```bash
# Recommended: Target-based generation
uv run generator generate /path/to/lancedb -o qa.json --target-pairs 300

# Generate from multiple tables (unified output) - text + code chunks ⭐ NEW
uv run generator generate /path/to/lancedb --table text_chunks --table code_chunks -o qa_unified.json --target-pairs 1500

# Test with limited data
uv run generator generate /path/to/lancedb -o qa.json --max-chunks 10 --target-pairs 50

# Generate with topic filtering (removes off-topic pairs)
uv run generator generate /path/to/lancedb -o qa.json --target-pairs 300 --topic "HDF5"

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
- `--topic TEXT` - Topic filter (e.g., 'HDF5') - removes off-topic pairs after generation ⭐ NEW
- `--provider TEXT` - Override provider
- `--model TEXT` - Override model

**Examples:**
```bash
uv run generator generate-cot /path/to/lancedb -o cot.json --target-pairs 100
uv run generator generate-cot /path/to/lancedb -o cot.json --max-chunks 10

# Generate with topic filtering
uv run generator generate-cot /path/to/lancedb -o cot.json --target-pairs 100 --topic "HDF5"
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
- `--topic TEXT` - Topic filter (e.g., 'HDF5') - removes off-topic pairs ⭐ NEW
- `--provider TEXT` - Override provider
- `--model TEXT` - Override model

**Format Support:** ✅ QA pairs, ✅ CoT examples, 🔄 Auto-detection

**Rating Criteria:** Clarity (0-3), Accuracy (0-3), Usefulness (0-2), Difficulty (0-2), Total (0-10)

**Topic Filtering:** When `--topic` is specified, the LLM judge evaluates whether each QA pair is directly related to the given topic and filters out irrelevant pairs. This is useful for removing off-topic content from your training data.

**Examples:**
```bash
uv run generator curate qa.json -o curated.json
uv run generator curate qa.json -o curated.json --threshold 8.0  # High quality only
uv run generator curate cot.json -o cot_curated.json  # Works with CoT format too

# Filter by topic - removes off-topic pairs
uv run generator curate qa.json -o hdf5_curated.json --topic "HDF5"
uv run generator curate qa.json -o python_curated.json --topic "Python programming" --threshold 7.5
```

**Output:** Adds `rating`, `clarity`, `accuracy`, `usefulness`, `difficulty`, `topic_relevant`, `reasoning` fields

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

---

## 🔧 Tool-Use Pipeline (Agentic Training)

Generate training data for function-calling and tool-use capabilities. Teaches models to select and invoke APIs based on user instructions.

### Tool Definition Format

Create a JSON file with your tools (see [configs/hdf5_tools.json](configs/hdf5_tools.json) for a complete example):

```json
{
  "name": "HDF5 MCP Tools",
  "tools": [
    {
      "tool_id": "hdf5_open_file",
      "name": "open_file",
      "category": "file_operations",
      "description": "Open an HDF5 file with lazy loading.",
      "parameters": [
        {"name": "path", "type": "string", "required": true, "description": "Path to the HDF5 file"},
        {"name": "mode", "type": "string", "required": false, "default": "r", "description": "Access mode"}
      ],
      "returns": {"type": "string", "description": "Success message"},
      "examples": ["open_file(path='/data/sim.h5', mode='r')"],
      "complexity": "simple"
    }
  ]
}
```

### `tool-generate` - Generate tool-use training examples

```bash
uv run generator tool-generate TOOLS.json -o OUTPUT.json [OPTIONS]
```

**Options:**
- `--config PATH` - Config file
- `--single-step` - Generate only single-tool examples
- `--multi-step` - Generate only multi-tool examples  
- `--target-pairs INT` - Total examples to generate
- `--max-steps INT` - Max steps for multi-step (default: 5)
- `--provider TEXT` - Override provider
- `--model TEXT` - Override model

**Modes:**
- **Auto (default)**: Balanced mix based on instruction complexity
- **Single-step**: One tool call per instruction
- **Multi-step**: Multiple coordinated tool calls with reasoning

**Examples:**
```bash
# Auto mode - balanced mix
uv run generator tool-generate configs/hdf5_tools.json -o examples.json

# Single-step only (simpler tasks)
uv run generator tool-generate configs/hdf5_tools.json -o simple.json --single-step

# Multi-step only (complex workflows)
uv run generator tool-generate configs/hdf5_tools.json -o complex.json --multi-step

# Target specific count
uv run generator tool-generate configs/hdf5_tools.json -o examples.json --target-pairs 500
```

**Output:**
```json
{
  "instruction": "Read the temperature data from my simulation file",
  "solution": {
    "reasoning_path": [
      {
        "step": 1,
        "thought": "First, I need to open the HDF5 file...",
        "tool": "open_file",
        "args": {"path": "simulation.h5", "mode": "r"}
      },
      {
        "step": 2,
        "thought": "Now I can read the temperature dataset...",
        "tool": "read_full_dataset",
        "args": {"path": "/results/temperature"}
      }
    ],
    "api_documentation": "open_file(path: string, mode: string = r)..."
  },
  "metadata": {
    "difficulty": "medium",
    "mode": "multi"
  }
}
```

---

### `tool-curate` - Filter tool-use examples by quality

```bash
uv run generator tool-curate INPUT.json -o OUTPUT.json [OPTIONS]
```

**Options:**
- `--threshold FLOAT` - Minimum rating (default: 7.0)
- `--config PATH` - Config file
- `--provider TEXT` - Override provider
- `--model TEXT` - Override model

**Examples:**
```bash
uv run generator tool-curate examples.json -o curated.json
uv run generator tool-curate examples.json -o high_quality.json --threshold 8.0
```

---

### `tool-pipeline` - Run full tool-use pipeline

```bash
uv run generator tool-pipeline TOOLS.json -o OUTPUT.json [OPTIONS]
```

**Options:**
- `--config PATH` - Config file
- `--single-step` - Single-tool examples only
- `--multi-step` - Multi-tool examples only
- `--target-pairs INT` - Total examples
- `--threshold FLOAT` - Curation threshold (default: 7.0)
- `--skip-curation` - Skip quality filtering

**Examples:**
```bash
# Full pipeline
uv run generator tool-pipeline configs/hdf5_tools.json -o training.json

# Quick generation without curation
uv run generator tool-pipeline configs/hdf5_tools.json -o draft.json --skip-curation

# High-quality multi-step only
uv run generator tool-pipeline configs/hdf5_tools.json -o complex.json --multi-step --threshold 8.0
```

---

### `tool-parse` - Validate tool definitions

```bash
uv run generator tool-parse TOOLS.json
```

Validates JSON format and shows tool summary.

---

### Included Tool Definitions

| File | Description | Tools |
|------|-------------|-------|
| [configs/hdf5_tools.json](configs/hdf5_tools.json) | HDF5 MCP Server tools | 25 tools (file, navigation, dataset, attributes, performance, discovery) |

**HDF5 Tools Categories:**
- **File Operations**: `open_file`, `close_file`, `get_filename`, `get_mode`
- **Navigation**: `get_by_path`, `list_keys`, `visit`
- **Dataset Operations**: `read_full_dataset`, `read_partial_dataset`, `get_shape`, `get_dtype`, `get_size`, `get_chunks`
- **Attribute Operations**: `read_attribute`, `list_attributes`
- **Performance**: `hdf5_parallel_scan`, `hdf5_batch_read`, `hdf5_stream_data`, `hdf5_aggregate_stats`
- **Discovery**: `analyze_dataset_structure`, `find_similar_datasets`, `suggest_next_exploration`, `identify_io_bottlenecks`, `optimize_access_pattern`

---

## 📝 QA Pipeline Commands
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

### QA Training (Domain Knowledge)

**Instruction Backtranslation** ([arxiv:2308.06259](https://arxiv.org/abs/2308.06259))  
Treats documents as answers, generates questions. More scalable than manual annotation, better long-tail coverage.

**Response Rewriting** ([arxiv:2408.04614](https://arxiv.org/abs/2408.04614))  
Rewrites answers for better clarity/structure while preserving all information. Improves alignment without adding hallucinated facts.

**Chain-of-Thought** ([arxiv:2305.02301](https://arxiv.org/abs/2305.02301))  
Generates or adds step-by-step reasoning to QA pairs. Enables smaller models to learn complex reasoning patterns from larger models.

### Tool-Use Training (Agentic Capabilities)

**Unified Tool Learning** (inspired by Toolformer, Gorilla, ToolLLM)  
- Generates realistic user instructions for tool usage
- Annotates solutions with step-by-step reasoning and tool calls
- Always includes API documentation (Gorilla insight) for better generalization
- Supports single-step (simple tasks) and multi-step (complex workflows)

### Common

**LLM-as-Judge**  
Multi-criteria rating (clarity, accuracy, usefulness, difficulty) with reasoning explanations. Threshold filtering for high-quality training data.

## 📊 Example Results

Coming soon...

## 🧪 Testing

```bash
uv run pytest tests/ -v                    # 19/19 tests passing ✅
uv run ruff check src/                     # All checks passed ✅
```

## 📚 Documentation

### Comprehensive Guides
- **[docs/OVERVIEW.md](docs/OVERVIEW.md)** - Quick overview and key points
- **[docs/DESIGN_DOCUMENTATION.md](docs/DESIGN_DOCUMENTATION.md)** - Complete design rationale (12,000+ words)
- **[docs/EXTRACTION_METHODOLOGY.md](docs/EXTRACTION_METHODOLOGY.md)** - Data extraction explained (10,000+ words)
- **[docs/PAPER_IMPLEMENTATIONS.md](docs/PAPER_IMPLEMENTATIONS.md)** - Paper → code mapping (5,000+ words)
- **[docs/ARCHITECTURE_DIAGRAMS.md](docs/ARCHITECTURE_DIAGRAMS.md)** - Visual diagrams and flows
- **[docs/DOCUMENTATION_INDEX.md](docs/DOCUMENTATION_INDEX.md)** - Navigation guide


### Research Papers
- [Instruction Backtranslation (Meta AI, ICLR 2024)](https://arxiv.org/abs/2308.06259)
- [LIMA (Meta AI, NeurIPS 2023)](https://arxiv.org/abs/2305.11206)
- [Distilling Step-by-Step (Google, 2023)](https://arxiv.org/abs/2305.02301)
- [AlpaGasus (UMD, ICLR 2024)](https://arxiv.org/abs/2307.08701)
- [Toolformer (Meta AI, NeurIPS 2023)](https://arxiv.org/abs/2302.04761)
- [Gorilla (UC Berkeley, NeurIPS 2024)](https://arxiv.org/abs/2305.15334)
- [ToolLLM (Tsinghua, ICLR 2024)](https://arxiv.org/abs/2307.16789)
- [CHANGES.md](CHANGES.md) - Recent updates & implementation notes
- [configs/prompts/](configs/prompts/) - Prompt templates
- [configs/hdf5_tools.json](configs/hdf5_tools.json) - HDF5 MCP tool definitions

---

**Research-backed synthetic data generation for LLM fine-tuning** | MIT License
