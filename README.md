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
- Location: `output/hdf_qa/`
- Sources: HDF5, parallel I/O research papers + documentation
- Model: Google Gemini 2.0 Flash
- Format: JSON, JSONL

### Jarvis Dataset (~300 QA pairs)
- Location: `output/jarvis_qa/`
- Sources: JARVIS I/O framework documentation
- Model: Google Gemini 2.0 Flash
- Format: JSON, JSONL

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
uv run python -m generator.cli generate \
  /path/to/lancedb \
  -o output/qa_raw.json \
  --n-pairs 3 \
  --batch-size 50
```

## 📝 CLI Commands

### generate

Generate QA pairs from LanceDB chunks.

```bash
generator generate LANCEDB_PATH -o OUTPUT.json [OPTIONS]

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
uv run python -m generator.cli generate /path/to/lancedb -o output/qa.json

# Test with limited chunks
uv run python -m generator.cli generate /path/to/lancedb -o output/test.json --max-chunks 10

# Override provider
uv run python -m generator.cli generate /path/to/lancedb -o output/qa.json --provider gemini --model gemini-2.0-flash-exp

# Custom configuration
uv run python -m generator.cli generate /path/to/lancedb -o output/qa.json --config my_config.yaml
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
