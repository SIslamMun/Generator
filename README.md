# Generator - Synthetic QA Pair Generator

Generate high-quality question-answer pairs from LanceDB chunks for LLM fine-tuning.

## 🎯 Features

- **Multi-provider LLM support**: Ollama, Claude, Google Gemini, vLLM, OpenAI, Anthropic
- **Modular client architecture**: Each LLM provider in separate, maintainable modules
- **Instruction Backtranslation**: Treat documents as "answers", generate "questions"
- **Rate limiting & retry logic**: Automatic handling of API quotas with exponential backoff
- **Progress tracking**: Rich progress bars and status messages
- **Batch processing**: Efficient chunked processing with intermediate saves

## 📦 Installation

```bash
# Using uv (recommended)
uv pip install -e .

# With cloud providers
uv pip install -e ".[cloud]"

# With all providers
uv pip install -e ".[all]"
```

## � Generated Datasets

### HDF5 Dataset (876 QA pairs)
- **Location**: `output/hdf5_qa/gemini/`
- **Sources**: HDF5, parallel I/O research papers + documentation
- **Model**: Google Gemini 2.0 Flash
- **Format**: JSON, JSONL
- **Status**: ✅ Complete

### Jarvis Dataset (300 QA pairs)
- **Location**: `output/jarvis_qa/claude/`
- **Sources**: JARVIS I/O framework documentation
- **Model**: Claude Sonnet 4
- **Format**: JSON, JSONL
- **Status**: ✅ Complete

### Jarvis Dataset - Gemini (150 QA pairs)
- **Location**: `output/jarvis_qa/` (partial, intermediate file)
- **Sources**: JARVIS I/O framework documentation
- **Model**: Google Gemini 2.0 Flash
- **Format**: JSON
- **Status**: ⚠️ Partial (50% - quota exhausted)

## 🚀 Quick Start

### 1. Configure LLM Provider

Edit `configs/config.yaml`:

```yaml
llm:
  provider: ollama
  model: mistral:latest
  base_url: http://localhost:11434
  temperature: 0.7
```

### 2. Generate QA Pairs

```bash
# Generate from LanceDB
uv run generator generate \
  /path/to/lancedb \
  -o output/qa_raw.json \
  --n-pairs 3 \
  --batch-size 50
```

## 📝 CLI Commands

### list-providers

List all available LLM providers and their setup instructions.

```bash
uv run generator list-providers
```

### generate

Generate QA pairs from LanceDB chunks.

```bash
uv run generator generate LANCEDB_PATH -o OUTPUT.json [OPTIONS]

Options:
  --config PATH         Config YAML file (default: configs/config.yaml)
  --table TEXT          LanceDB table name (default: text_chunks)
  --n-pairs INT         QA pairs per chunk (default: 5)
  --batch-size INT      Chunks per batch (default: 50)
  --max-chunks INT      Max chunks to process (for testing)
  --provider TEXT       Override LLM provider from config
  --model TEXT          Override LLM model from config
```

**Examples:**

```bash
# Basic usage
uv run generator generate /path/to/lancedb -o output/qa.json

# Test with limited chunks
uv run generator generate /path/to/lancedb -o output/test.json --max-chunks 10

# Override provider
uv run generator generate /path/to/lancedb -o output/qa.json --provider gemini --model gemini-2.0-flash-exp

# Custom configuration
uv run generator generate /path/to/lancedb -o output/qa.json --config my_config.yaml
```

### curate

Filter QA pairs by quality using LLM-as-Judge.

```bash
uv run generator curate INPUT.json -o OUTPUT.json [OPTIONS]

Options:
  --config PATH         Config YAML file (default: configs/config.yaml)
  --threshold FLOAT     Minimum rating (1-10, default: 7.0)
  --batch-size INT      Pairs rated per LLM call (default: 5)
  --provider TEXT       Override LLM provider from config
  --model TEXT          Override LLM model from config
```

**Examples:**

```bash
# Basic curation
uv run generator curate output/qa_raw.json -o output/qa_curated.json

# High quality threshold
uv run generator curate output/qa_raw.json -o output/qa_high.json --threshold 8.5

# Use specific provider for rating
uv run generator curate output/qa_raw.json -o output/qa_curated.json --provider claude
```

### export

Export QA pairs to training format.

```bash
uv run generator export INPUT.json -o OUTPUT [OPTIONS]

Options:
  -f, --format CHOICE   Output format: chatml, alpaca, sharegpt, jsonl (default: chatml)
  --system-prompt TEXT  System prompt for conversation formats
```

**Examples:**

```bash
# Export to ChatML format
uv run generator export output/qa_curated.json -o output/training.jsonl -f chatml

# Export to Alpaca format
uv run generator export output/qa_curated.json -o output/training.json -f alpaca

# Export with custom system prompt
uv run generator export output/qa_curated.json -o output/training.jsonl -f chatml --system-prompt "You are an HDF5 expert."
```

### pipeline

Run full pipeline: generate → curate → export.

```bash
uv run generator pipeline LANCEDB_PATH -o OUTPUT [OPTIONS]

Options:
  --config PATH         Config YAML file (default: configs/config.yaml)
  --threshold FLOAT     Curation threshold (1-10, default: 7.0)
  -f, --format CHOICE   Output format: chatml, alpaca, sharegpt, jsonl (default: chatml)
  --max-chunks INT      Max chunks to process (for testing)
```

**Examples:**

```bash
# Full pipeline with defaults
uv run generator pipeline /path/to/lancedb -o output/training.jsonl

# Test pipeline with limited chunks
uv run generator pipeline /path/to/lancedb -o output/test.jsonl --max-chunks 10

# High quality training data in Alpaca format
uv run generator pipeline /path/to/lancedb -o output/training.json -f alpaca --threshold 8.0
```

## 🎨 LLM Provider Setup

### Ollama (Local)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull mistral:latest

# Verify
ollama list
```

### Claude (API)

```bash
# Install packages
uv pip install ".[cloud]"

# Get API key at: https://console.anthropic.com/
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Google Gemini (API)

```bash
# Install packages
uv pip install ".[cloud]"

# Get API key at: https://aistudio.google.com/apikey
export GOOGLE_API_KEY="your-key-here"
```

### OpenAI (API)

```bash
# Install packages
uv pip install ".[cloud]"

# Set API key
export OPENAI_API_KEY="sk-..."
```

## ⚙️ Configuration Files

### configs/config.yaml

```yaml
llm:
  provider: ollama
  model: mistral:latest
  temperature: 0.7
  max_tokens: 4096

generation:
  n_pairs_per_chunk: 5
  batch_size: 50
  max_retries: 3
```

### configs/prompts.yaml

Customize prompts for QA generation:

```yaml
qa_generation: |
  Given the following text, generate {n_pairs} diverse question-answer pairs.
  The text serves as the answer, generate relevant questions.
  
  Text: {text}
  
  Return a JSON array of objects with "question" and "answer" fields.
```

## 🏗️ Architecture

```
src/generator/
├── clients/          # Modular LLM clients
│   ├── base.py      # Abstract base class
│   ├── ollama.py    # Ollama implementation
│   ├── claude.py    # Claude API
│   ├── google_adk.py # Google Gemini API
│   ├── vllm.py      # vLLM
│   ├── openai.py    # OpenAI API
│   └── anthropic.py # Anthropic API
├── qa_generator.py  # QA generation with rate limiting
├── cli.py           # Command-line interface
└── __init__.py      # Package exports
```

### Using Clients in Code

```python
from generator.clients import get_client

# Create client using factory
client = get_client("ollama", {
    "model": "mistral:latest",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "max_tokens": 4096
})

# Generate text
response = client.generate("What is machine learning?")
```

## 🔬 Methodology

Based on **Instruction Backtranslation** (Meta AI, ICLR 2024):
- Treat document chunks as "answers"
- LLM generates relevant "questions"
- Creates natural instruction-response pairs

## 📄 License

MIT
