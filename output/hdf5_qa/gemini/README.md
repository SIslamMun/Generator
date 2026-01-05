# HDF5 QA Dataset

Generated QA pairs from HDF5/Parallel I/O research papers and documentation.

## Dataset Statistics

- **Total QA Pairs**: 876
- **Model**: Google Gemini 2.0 Flash Exp
- **Generated**: January 5, 2026
- **Methodology**: Instruction Backtranslation

## Source Distribution

### Research Papers (50 sources)
- ExaHDF5 papers
- Efficient Asynchronous I/O research
- HDF5 subfiling performance studies
- Transparent Asynchronous Parallel I/O papers

### Documentation (50 sources)
- HDF5 official documentation
- HDFView documentation
- HDF5-JSON specifications
- HDF Group websites

## Files

- `full_gemini.json` - Array format, 876 QA pairs
- `full_gemini.jsonl` - JSONL format (one QA per line)
- `full_gemini_summary.json` - Generation metadata
- `README.md` - This file

## Format

Each QA pair contains:
```json
{
  "question": "What is HDF5?",
  "answer": "HDF5 is a file format and library...",
  "chunk_id": "...",
  "source_file": "papers/...",
  "generated_at": "2026-01-05T...",
  "model": "gemini-2.0-flash-exp"
}
```

## Generation Process

1. **Chunk Filtering**: 292 valid chunks (≥200 chars) from 385 total
2. **Rate Limiting**: 6 seconds between requests (10/min API limit)
3. **QA Generation**: 3 questions per chunk using instruction backtranslation
4. **Quality**: JSON5 parsing with markdown extraction fallback

## Usage

Fine-tune LLMs on HDF5 and parallel I/O knowledge:

```bash
# Load for training
import json
qa_pairs = json.load(open('full_gemini.json'))

# Or use JSONL format
with open('full_gemini.jsonl') as f:
    for line in f:
        qa = json.loads(line)
        # Process each QA pair
```

## Citation

Based on research from Phagocyte pipeline (HDF Group):
- Papers from https://github.com/grc-iit/Phagocyte
- LanceDB chunks processed through QA generation pipeline
