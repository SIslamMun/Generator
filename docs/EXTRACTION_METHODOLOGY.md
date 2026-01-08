# Data Extraction and Generation Methodology

**Author:** Shazzadul  
**Purpose:** Detailed explanation of how data flows through the Generator system, from raw documents to training-ready datasets.

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Processing: Document to LanceDB](#pre-processing-document-to-lancedb)
3. [Pipeline 1: QA Pipeline (Knowledge Extraction)](#qa-extraction-process)
4. [Pipeline 2: CoT Pipeline (Reasoning Extraction)](#cot-extraction-process)
5. [Pipeline 3: Tool-Use Pipeline (Agentic Skills)](#tool-use-extraction-process)
6. [Quality Control Pipeline](#quality-control-pipeline)
7. [Format Conversion and Export](#format-conversion-and-export)

---

## Overview

### The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SOURCE DOCUMENTS                              │
│  Research Papers │ GitHub Repos │ Documentation │ API Specs         │
└──────────────┬──────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    PHAGOCYTE PRE-PROCESSING                          │
│  1. Research: AI-assisted research and citation extraction           │
│  2. Parse: Extract DOIs, references, download PDFs                   │
│  3. Ingest: Convert to clean markdown with images                    │
│  4. Process: Chunk semantically and vectorize to LanceDB             │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      LANCEDB VECTOR STORE                            │
│  ┌────────────┬─────────────┬────────────┬──────────────────┐       │
│  │ chunk_id   │ text        │ vector     │ metadata         │       │
│  ├────────────┼─────────────┼────────────┼──────────────────┤       │
│  │ chunk_001  │ "HDF5 is..."│ [0.1, ...] │ {source, type}   │       │
│  │ chunk_002  │ "Parallel..."│ [0.2, ...] │ {source, type}   │       │
│  └────────────┴─────────────┴────────────┴──────────────────┘       │
└──────────────┬───────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    GENERATOR MODULE (THIS PROJECT)                   │
│                      Three Independent Pipelines                     │
└────────┬──────────────────────┬──────────────────────┬───────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  PIPELINE 1:    │    │  PIPELINE 2:    │    │  PIPELINE 3:    │
│  QA Pipeline    │    │  CoT Pipeline   │    │  Tool-Use       │
│  (Knowledge)    │    │  (Reasoning)    │    │  (Agentic)      │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         ▼                      ▼                      ▼
  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
  │ GENERATE    │       │ GENERATE    │       │ PARSE       │
  │ QA Pairs    │       │ CoT Pairs   │       │ Tools       │
  │             │       │             │       │             │
  │ Instruction │       │ Direct Gen  │       │ OpenAPI/    │
  │ Backtrans.  │       │ OR Enhance  │       │ JSON        │
  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
         │                      │                      │
         │ qa_raw.json          │ cot_raw.json         │ tools.json
         │                      │                      │
         ▼                      ▼                      ▼
  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
  │ ENRICH      │       │ VALIDATE    │       │ GENERATE    │
  │ Answers     │       │ Reasoning   │       │ Examples    │
  │             │       │             │       │             │
  │ Response    │       │ Check steps │       │ Single/     │
  │ Rewriting   │       │ Coherence   │       │ Multi-step  │
  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
         │                      │                      │
         │ qa_enriched.json     │ cot_validated.json   │ examples.json
         │                      │                      │
         ▼                      ▼                      ▼
  ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
  │ CURATE      │       │ CURATE      │       │ EXECUTE     │
  │ by Quality  │       │ by Quality  │       │ & Verify    │
  │             │       │             │       │             │
  │ LLM-as-     │       │ LLM-as-     │       │ Triple      │
  │ Judge ≥7/10 │       │ Judge       │       │ Verification│
  └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
         │                      │                      │
         │ qa_curated.json      │ cot_curated.json     │ verified.json
         │                      │                      │
         │                      │                      ▼
         │                      │              ┌─────────────┐
         │                      │              │ CURATE      │
         │                      │              │ by Quality  │
         │                      │              │             │
         │                      │              │ Filter best │
         │                      │              │ examples    │
         │                      │              └──────┬──────┘
         │                      │                      │
         │                      │              tool_curated.json
         │                      │                      │
         └──────────────────────┴──────────────────────┘
                                │
                                ▼
                        ┌───────────────┐
                        │ EXPORT        │
                        │ (All Formats) │
                        │               │
                        │ ChatML        │
                        │ Alpaca        │
                        │ ShareGPT      │
                        │ JSONL         │
                        └───────┬───────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    TRAINING-READY DATASETS                           │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ training.jsonl   │  │ cot_training.    │  │ tool_training.   │  │
│  │                  │  │ jsonl            │  │ json             │  │
│  │ Simple Q&A       │  │                  │  │                  │  │
│  │ Knowledge pairs  │  │ Q&A + Reasoning  │  │ Tool-use         │  │
│  │                  │  │ Step-by-step     │  │ examples         │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Pre-Processing: Document to LanceDB

### Phase 1: Document Acquisition (Pre-Generator)

**Input:** Raw documents in various formats
- PDFs (research papers, technical reports)
- Web pages (documentation, tutorials)
- GitHub repositories (code + README)
- API specifications (OpenAPI, Swagger)

**Tools Used:** (From Phagocyte's existing pipeline)

```bash
# 1. AI Research Assistant
phagocyte research "HDF5 parallel I/O" -o research_report.md
# Output: Markdown with citations and DOIs

# 2. Reference Parser
phagocyte parse refs research_report.md --export-batch
# Output: batch.json with DOIs and URLs

# 3. Document Downloader
phagocyte parse batch batch.json -o ./papers
# Output: PDFs and web pages

# 4. Document Ingestor
phagocyte ingest batch ./papers -o ./markdown
# Output: Clean markdown + extracted images
```

**Ingestor Features:**
- PDF → Markdown conversion with formatting preservation
- Code extraction from GitHub repositories
- Image extraction and OCR if needed
- Metadata preservation (source, date, authors)

### Phase 2: Semantic Chunking (Processor)

**Why Chunking Matters:**

Traditional naive chunking (e.g., every 512 tokens) breaks semantic units:

```
❌ BAD: Naive chunking
Chunk 1: "...and this algorithm works by"
Chunk 2: "splitting the data into blocks..."
```

Semantic chunking preserves meaning:

```
✓ GOOD: Semantic chunking
Chunk 1: "HDF5 Parallel I/O uses MPI collective operations. 
          The algorithm works by splitting the data into blocks 
          and distributing them across processes..."
```

**Chunking Strategy in Processor:**

```python
# From Phagocyte's processor (conceptual)
def chunk_document(markdown_text):
    chunks = []
    
    # 1. Split by headers (H1, H2, H3)
    sections = split_by_headers(markdown_text)
    
    for section in sections:
        # 2. If section too large, split by paragraphs
        if len(section) > MAX_CHUNK_SIZE:
            paragraphs = split_by_paragraphs(section)
            
            # 3. Group paragraphs into chunks
            current_chunk = []
            for para in paragraphs:
                if len(current_chunk) + len(para) < MAX_CHUNK_SIZE:
                    current_chunk.append(para)
                else:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [para]
            
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
        else:
            chunks.append(section)
    
    return chunks
```

**Chunk Size Guidelines:**
- Target: 512-1024 tokens per chunk
- Minimum: 256 tokens (avoid tiny fragments)
- Maximum: 2048 tokens (avoid oversized chunks)

### Phase 3: Vectorization and LanceDB Storage

**LanceDB Schema:**

```python
# Table: text_chunks
{
    "id": str,              # Unique chunk identifier
    "text": str,            # The actual chunk text
    "vector": List[float],  # Embedding (e.g., 768-dim)
    "source_file": str,     # Original document path
    "doc_type": str,        # "research_paper", "code", "docs"
    "chunk_index": int,     # Position in original document
    "metadata": Dict        # Additional context
}

# Table: code_chunks (similar structure but for code)
{
    "id": str,
    "code": str,            # Code snippet
    "language": str,        # Programming language
    "vector": List[float],
    "source_file": str,
    "function_name": str,   # If applicable
    "metadata": Dict
}
```

**Example LanceDB Entry:**

```json
{
  "id": "hdf5_intro_chunk_003",
  "text": "HDF5 supports parallel I/O through MPI. Applications can use collective operations to coordinate writes across multiple processes. This enables efficient data storage for large-scale simulations...",
  "vector": [0.123, 0.456, 0.789, ...],
  "source_file": "papers/hdf5_parallel_io.md",
  "doc_type": "research_paper",
  "chunk_index": 3,
  "metadata": {
    "title": "Parallel HDF5: A Portable Programming Interface",
    "authors": ["Smith et al."],
    "year": 2020,
    "section": "3.2 Collective Operations"
  }
}
```

**Why LanceDB?**
1. **Columnar storage:** Fast scanning of specific fields
2. **Vector support:** Can do similarity-based sampling
3. **SQL-like queries:** Easy filtering
4. **Python integration:** Direct pandas/polars support

---

## QA Extraction Process

### Step-by-Step: From Chunks to QA Pairs

#### **Step 1: Load Chunks from LanceDB**

```python
# In qa_generator.py
def load_chunks(db_path, table_name, filters=None):
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    
    # Optional: Filter by document type
    if filters:
        chunks = table.search().where(filters).to_pandas()
    else:
        chunks = table.to_pandas()
    
    console.print(f"[cyan]Loaded {len(chunks)} chunks from {table_name}[/cyan]")
    return chunks
```

**Example Query:**
```python
# Load only research papers
chunks = load_chunks(
    "lancedb/",
    "text_chunks",
    filters="doc_type = 'research_paper'"
)
```

#### **Step 2: Calculate Generation Targets**

```python
def calculate_targets(chunks, target_pairs=None, n_pairs_per_chunk=None):
    if target_pairs:
        # User wants specific total
        n_pairs_per_chunk = max(1, target_pairs // len(chunks))
        estimated_total = n_pairs_per_chunk * len(chunks)
        
        console.print(f"""
        [cyan]Target-based generation:[/cyan]
        - Desired: {target_pairs} pairs
        - Per chunk: {n_pairs_per_chunk}
        - Estimated: {estimated_total} pairs
        """)
    
    elif n_pairs_per_chunk:
        # Fixed per chunk
        estimated_total = n_pairs_per_chunk * len(chunks)
        console.print(f"[cyan]Generating {estimated_total} pairs ({n_pairs_per_chunk} per chunk)[/cyan]")
    
    else:
        raise ValueError("Specify either target_pairs or n_pairs_per_chunk")
    
    return n_pairs_per_chunk, estimated_total
```

#### **Step 3: Generate Questions (Instruction Backtranslation)**

**Prompt Template (from `configs/prompts/qa_generation.yaml`):**

```yaml
qa_generation: |
  You are an expert at generating high-quality question-answer pairs for training language models.
  
  Given the following document chunk, generate {n_pairs} question-answer pairs.
  
  REQUIREMENTS:
  1. Questions should be clear, specific, and answerable from the text
  2. Answers should come directly from or be grounded in the text
  3. Cover different aspects and difficulty levels
  4. Use varied question types (what, how, why, when, where)
  5. Avoid yes/no questions unless particularly meaningful
  
  DOCUMENT CHUNK:
  {chunk_text}
  
  Generate exactly {n_pairs} high-quality question-answer pairs.
  Return as valid JSON array:
  [
    {{
      "question": "Clear, specific question?",
      "answer": "Comprehensive answer grounded in the text."
    }},
    ...
  ]
  
  JSON output:
```

**Generation Process:**

```python
def generate_pairs_for_chunk(chunk, n_pairs, llm, prompt_template):
    # Format prompt with chunk text
    prompt = prompt_template.format(
        n_pairs=n_pairs,
        chunk_text=chunk["text"]
    )
    
    # Call LLM
    response = llm.generate(prompt, temperature=0.7)
    
    # Parse JSON (use json5 for forgiveness)
    try:
        pairs = json5.loads(response)
    except json5.JSONDecodeError as e:
        console.print(f"[red]Failed to parse JSON: {e}[/red]")
        return []
    
    # Add metadata to each pair
    for pair in pairs:
        pair["source_chunk_id"] = chunk["id"]
        pair["source_file"] = chunk["source_file"]
        pair["generated_at"] = datetime.now().isoformat()
    
    return pairs
```

**Example LLM Response:**

```json
[
  {
    "question": "What protocol does HDF5 use for parallel I/O operations?",
    "answer": "HDF5 uses MPI (Message Passing Interface) for parallel I/O operations, enabling applications to use collective operations to coordinate writes across multiple processes."
  },
  {
    "question": "How does parallel HDF5 enable efficient data storage for large-scale simulations?",
    "answer": "Parallel HDF5 enables efficient data storage by allowing collective operations that coordinate writes across multiple processes using MPI, which is particularly beneficial for large-scale simulations requiring distributed data handling."
  },
  {
    "question": "What type of operations can applications use with HDF5 parallel I/O?",
    "answer": "Applications can use collective operations with HDF5 parallel I/O to coordinate writes across multiple processes."
  }
]
```

#### **Step 4: Batch Processing with Checkpoints**

```python
def generate_qa_from_lancedb(db_path, output_path, ...):
    chunks = load_chunks(db_path, table_name)
    n_pairs_per_chunk, estimated = calculate_targets(chunks, target_pairs, n_pairs_per_chunk)
    
    all_pairs = []
    checkpoint_file = Path(output_path).with_suffix(".checkpoint.json")
    
    # Resume from checkpoint if exists
    if checkpoint_file.exists():
        all_pairs = json.loads(checkpoint_file.read_text())
        processed_chunks = len(all_pairs) // n_pairs_per_chunk
        chunks = chunks[processed_chunks:]
        console.print(f"[yellow]Resuming from checkpoint: {len(all_pairs)} pairs[/yellow]")
    
    # Process in batches
    with Progress() as progress:
        task = progress.add_task("Generating QA pairs...", total=len(chunks))
        
        for batch in batched(chunks, batch_size):
            batch_pairs = []
            
            for chunk in batch:
                pairs = generate_pairs_for_chunk(chunk, n_pairs_per_chunk, llm, prompt)
                batch_pairs.extend(pairs)
            
            all_pairs.extend(batch_pairs)
            
            # Save checkpoint
            checkpoint_file.write_text(json.dumps(all_pairs, indent=2))
            
            progress.update(task, advance=len(batch))
    
    # Save final output
    Path(output_path).write_text(json.dumps(all_pairs, indent=2))
    
    # Clean up checkpoint
    checkpoint_file.unlink()
    
    return all_pairs
```

#### **Step 5: Optional Topic Filtering**

**Why Filter?** Sometimes chunks contain tangential content

```python
def filter_by_topic(qa_pairs, topic, llm):
    """Remove QA pairs not relevant to specified topic."""
    
    filtered = []
    
    for pair in qa_pairs:
        # Ask LLM if pair is on-topic
        check = llm.generate(f"""
        Is this question-answer pair relevant to the topic "{topic}"?
        
        Question: {pair['question']}
        Answer: {pair['answer']}
        
        Answer: yes or no
        """, temperature=0.1)
        
        if "yes" in check.lower():
            filtered.append(pair)
        else:
            console.print(f"[dim]Filtered: {pair['question'][:50]}...[/dim]")
    
    removed = len(qa_pairs) - len(filtered)
    console.print(f"[cyan]Topic filter: removed {removed} off-topic pairs[/cyan]")
    
    return filtered
```

**Example Usage:**
```bash
# Generate QA pairs but only keep HDF5-related ones
uv run generator generate /db -o qa.json --target-pairs 300 --topic "HDF5"
```

### Step 6: Enrichment (Response Rewriting)

**Why Enrich?** Raw document chunks are information-rich but may not be formatted as good assistant responses.

**Enrichment Prompt (from `configs/prompts/qa_enrichment.yaml`):**

```yaml
qa_enrichment: |
  Rewrite the following answer to be a high-quality assistant response.
  
  GOALS:
  - Maintain ALL information from original answer
  - Improve structure and clarity
  - Use proper formatting (lists, bold, etc.)
  - Sound like a helpful AI assistant
  - Be concise but comprehensive
  
  ORIGINAL QUESTION:
  {question}
  
  ORIGINAL ANSWER:
  {original_answer}
  
  IMPROVED ANSWER:
```

**Enrichment Process:**

```python
def enrich_qa_pairs(qa_pairs, llm, prompts_dir):
    prompts = load_prompts(prompts_dir)
    enriched = []
    
    for batch in batched(qa_pairs, batch_size=5):
        for pair in batch:
            prompt = prompts["qa_enrichment"].format(
                question=pair["question"],
                original_answer=pair["answer"]
            )
            
            # Get improved answer
            improved = llm.generate(prompt, temperature=0.3)
            
            # Store both
            enriched_pair = {
                **pair,
                "answer": improved,
                "original_answer": pair["answer"],  # Keep for comparison
                "enriched": True
            }
            enriched.append(enriched_pair)
    
    return enriched
```

**Example Transformation:**

**Before Enrichment:**
```json
{
  "question": "What is HDF5 used for?",
  "answer": "HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited variety of datatypes and is designed for flexible and efficient I/O and for high volume and complex data."
}
```

**After Enrichment:**
```json
{
  "question": "What is HDF5 used for?",
  "answer": "HDF5 (Hierarchical Data Format 5) is used for storing and managing large, complex datasets. Here are its main uses:\n\n**Core capabilities:**\n- Stores unlimited variety of datatypes\n- Designed for flexible and efficient I/O operations\n- Handles high-volume data with complex structure\n\n**Common applications:**\n- Scientific computing and simulations\n- Big data analytics\n- Machine learning model storage\n- Satellite and sensor data\n\nHDF5 provides both a data model for organizing data and a library/file format for accessing it efficiently.",
  "original_answer": "HDF5 is a data model, library, and file format...",
  "enriched": true
}
```

---

## CoT Extraction Process

### Method 1: Direct CoT Generation

**Goal:** Generate QA pairs with step-by-step reasoning from scratch

**CoT Prompt (from `configs/prompts/cot_generation.yaml`):**

```yaml
cot_generation: |
  You are an expert at creating educational question-answer pairs with reasoning.
  
  Given this document chunk, generate {n_pairs} question-answer pairs that include step-by-step reasoning.
  
  Each pair should have:
  1. A question that requires reasoning to answer
  2. Step-by-step reasoning (2-5 clear steps)
  3. A final answer that follows from the reasoning
  
  DOCUMENT CHUNK:
  {chunk_text}
  
  Generate {n_pairs} pairs in this exact JSON format:
  [
    {{
      "question": "Question requiring reasoning?",
      "reasoning": [
        "Step 1: First we need to...",
        "Step 2: Then we can...",
        "Step 3: Finally..."
      ],
      "answer": "Final answer based on reasoning."
    }}
  ]
  
  JSON output:
```

**Generation Process:**

```python
def generate_cot_pairs(lancedb_path, output_path, llm_config, ...):
    chunks = load_chunks(lancedb_path, table_name)
    llm = get_client(llm_config["provider"], llm_config)
    prompts = load_prompts("configs/prompts")
    
    cot_pairs = []
    
    for chunk in chunks:
        prompt = prompts["cot_generation"].format(
            n_pairs=n_pairs_per_chunk,
            chunk_text=chunk["text"]
        )
        
        response = llm.generate(prompt, temperature=0.7)
        pairs = json5.loads(response)
        
        # Add metadata
        for pair in pairs:
            pair["source_chunk_id"] = chunk["id"]
            pair["cot_generated"] = True
            pair["reasoning_steps"] = len(pair["reasoning"])
        
        cot_pairs.extend(pairs)
    
    return cot_pairs
```

**Example CoT Output:**

```json
{
  "question": "Why is HDF5's chunking strategy important for parallel I/O performance?",
  "reasoning": [
    "HDF5 stores data in chunks, which are fixed-size blocks that can be read/written independently",
    "In parallel I/O, multiple processes need to access different parts of the dataset simultaneously",
    "When chunks align with how data is distributed across processes, each process can read/write its own chunks without conflicts",
    "This eliminates the need for processes to coordinate access to the same chunk, reducing contention",
    "Proper chunk alignment allows processes to perform independent I/O operations in parallel, maximizing bandwidth"
  ],
  "answer": "HDF5's chunking strategy is important for parallel I/O performance because it allows data to be divided into independently accessible blocks. When chunks align with data distribution across processes, each process can perform I/O on its own chunks without conflicts or coordination overhead, enabling truly parallel access and maximizing I/O bandwidth.",
  "source_chunk_id": "hdf5_parallel_chunk_005",
  "cot_generated": true,
  "reasoning_steps": 5
}
```

### Method 2: Enhance Existing QA with CoT

**Goal:** Add chain-of-thought reasoning to already-generated high-quality QA pairs

**Why This Approach?**
- Already have good QA pairs from generation phase
- Want to add reasoning without re-generating entire Q&A
- More cost-effective than full CoT generation
- Preserves specific question phrasings and answer quality
- Enables hybrid workflow: generate → curate → enhance

**Research Motivation:**
The "Distilling Step-by-Step" paper showed that adding reasoning steps to existing data significantly improves small model performance. By taking curated QA pairs and adding reasoning, we get the best of both worlds: quality questions AND reasoning transparency.

**Enhancement Prompt (from `configs/prompts/cot_enhancement.yaml`):**

```yaml
cot_enhancement: |
  Given this question-answer pair, add step-by-step reasoning that explains how to arrive at the answer.
  
  QUESTION:
  {question}
  
  ANSWER:
  {answer}
  
  Provide 2-5 clear reasoning steps that logically lead to this answer.
  The reasoning should:
  - Break down the problem systematically
  - Show intermediate conclusions or observations
  - Build progressively toward the final answer
  - Be coherent and consistent with the given answer
  
  Return JSON:
  {{
    "reasoning": [
      "Step 1: [First logical step or observation]",
      "Step 2: [Build on step 1 with deeper insight]",
      "Step 3: [Continue progression toward answer]",
      ...
    ]
  }}
```

**Enhancement Process:**

```python
def enhance_with_cot(qa_pairs, llm, prompts):
    """Add CoT reasoning to existing QA pairs."""
    enhanced = []
    stats = {"skipped": 0, "enhanced": 0, "failed": 0}
    
    for pair in qa_pairs:
        # Skip if already has reasoning
        if "reasoning" in pair and pair["reasoning"]:
            enhanced.append(pair)
            stats["skipped"] += 1
            continue
        
        try:
            # Generate reasoning steps
            prompt = prompts["cot_enhancement"].format(
                question=pair["question"],
                answer=pair["answer"]
            )
            
            response = llm.generate(prompt, temperature=0.5)
            reasoning_data = json5.loads(response)
            
            # Validate reasoning quality
            if not reasoning_data.get("reasoning") or len(reasoning_data["reasoning"]) < 2:
                logger.warning(f"Insufficient reasoning for: {pair['question'][:50]}")
                stats["failed"] += 1
                continue
            
            # Add reasoning to original pair
            enhanced_pair = {
                **pair,
                "reasoning": reasoning_data["reasoning"],
                "cot_enhanced": True,
            "reasoning_steps": len(reasoning_data["reasoning"])
        }
        enhanced.append(enhanced_pair)
    
    return enhanced
```

**Before → After Example:**

**Before (Simple QA):**
```json
{
  "question": "What is the default chunk size in HDF5?",
  "answer": "HDF5 doesn't have a fixed default chunk size. The optimal chunk size depends on the dataset dimensions, access patterns, and I/O system characteristics."
}
```

**After (With CoT):**
```json
{
  "question": "What is the default chunk size in HDF5?",
  "reasoning": [
    "HDF5 chunking is configurable, not fixed by default",
    "Optimal chunk size varies based on dataset properties (dimensions, data type)",
    "Access patterns matter - sequential vs random, single vs parallel",
    "I/O system characteristics (disk block size, network transfer size) also influence optimal chunking"
  ],
  "answer": "HDF5 doesn't have a fixed default chunk size. The optimal chunk size depends on the dataset dimensions, access patterns, and I/O system characteristics.",
  "cot_enhanced": true,
  "reasoning_steps": 4
}
```

---

## Tool-Use Extraction Process

### Step 1: Parse API Definitions

**Input Formats Supported:**
- OpenAPI/Swagger JSON/YAML
- Custom JSON tool definitions
- Python function signatures (via introspection)

**Parsing OpenAPI Specs:**

```python
def parse_openapi_spec(spec_path):
    with open(spec_path) as f:
        spec = json.load(f) if spec_path.endswith('.json') else yaml.safe_load(f)
    
    tools = []
    
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            tool = Tool(
                tool_id=f"{method}_{path.replace('/', '_')}",
                name=details.get("operationId", path),
                description=details.get("summary", ""),
                parameters=parse_parameters(details.get("parameters", [])),
                returns=parse_response(details.get("responses", {})),
                examples=details.get("examples", []),
                category=details.get("tags", ["general"])[0],
                complexity=classify_complexity(details)
            )
            tools.append(tool)
    
    return tools
```

**Custom JSON Format:**

```json
{
  "tools": [
    {
      "name": "open_file",
      "description": "Open an HDF5 file for reading or writing",
      "parameters": [
        {
          "name": "filename",
          "type": "string",
          "description": "Path to HDF5 file",
          "required": true
        },
        {
          "name": "mode",
          "type": "string",
          "description": "File access mode",
          "required": true,
          "enum": ["r", "r+", "w", "w-", "a"]
        }
      ],
      "returns": {
        "type": "File",
        "description": "HDF5 file object"
      },
      "examples": [
        {
          "arguments": {"filename": "data.h5", "mode": "r"},
          "result": "<HDF5 file 'data.h5' (mode r)>"
        }
      ]
    }
  ]
}
```

**Complexity Classification:**

```python
def classify_complexity(tool_details):
    params = tool_details.get("parameters", [])
    required = [p for p in params if p.get("required", False)]
    has_nested = any(
        p.get("schema", {}).get("type") == "object" 
        for p in params
    )
    
    # Simple: Few required params, no nesting
    if len(required) <= 2 and not has_nested:
        return "simple"
    
    # Complex: Many params or nested structures
    elif len(required) > 5 or has_nested:
        return "complex"
    
    # Medium: Everything else
    return "medium"
```

### Step 2: Generate Tool-Use Examples

#### **Mode 1: Single-Step (Toolformer Approach)**

**Goal:** Simple, single-tool calls for basic operations

```python
def generate_single_step_example(tool, llm, prompts):
    # Generate realistic instruction
    instruction = llm.generate(prompts["single_instruction"].format(
        tool_name=tool.name,
        tool_description=tool.description,
        parameters=[p.name for p in tool.parameters]
    ))
    
    # Generate realistic arguments
    arguments = {}
    for param in tool.parameters:
        if param.required:
            arguments[param.name] = generate_realistic_value(param)
    
    # Format with API documentation (Gorilla)
    example = {
        "instruction": instruction.strip(),
        "solution": {
            "reasoning": f"To {instruction}, I need to call {tool.name}",
            "tool_calls": [{
                "tool": tool.name,
                "arguments": arguments
            }]
        },
        "api_documentation": format_api_docs(tool),  # Key: Gorilla
        "mode": "single-step"
    }
    
    return example
```

**Example Single-Step Output:**

```json
{
  "instruction": "Open the file 'experiment_data.h5' for reading",
  "solution": {
    "reasoning": "To open the file 'experiment_data.h5' for reading, I need to call open_file with the filename and mode='r'",
    "tool_calls": [
      {
        "tool": "open_file",
        "arguments": {
          "filename": "experiment_data.h5",
          "mode": "r"
        }
      }
    ]
  },
  "api_documentation": "open_file(filename: str, mode: str) -> File\n\nOpen an HDF5 file for reading or writing.\n\nParameters:\n- filename: Path to HDF5 file\n- mode: File access mode ('r', 'r+', 'w', 'w-', 'a')\n\nReturns: HDF5 file object\n\nExample:\nfile = open_file('data.h5', 'r')",
  "mode": "single-step"
}
```

#### **Mode 2: Multi-Step (ToolLLM DFSDT Approach)**

**Goal:** Complex workflows with multiple tool calls in sequence

```python
def generate_multi_step_example(tools, llm, prompts):
    # Select 2-4 related tools
    tool_chain = select_tool_chain(tools, min_tools=2, max_tools=4)
    
    # Generate complex instruction
    instruction = llm.generate(prompts["multi_instruction"].format(
        tools=[t.name for t in tool_chain]
    ))
    
    # Generate reasoning chain
    reasoning_steps = []
    for i, tool in enumerate(tool_chain, 1):
        step = llm.generate(prompts["reasoning_step"].format(
            step_number=i,
            previous_steps=reasoning_steps,
            current_tool=tool.name,
            goal=instruction
        ))
        
        reasoning_step = {
            "step": i,
            "thought": step["thought"],
            "tool": tool.name,
            "arguments": generate_realistic_args(tool),
            "expected_result": step["expected_result"]
        }
        reasoning_steps.append(reasoning_step)
    
    example = {
        "instruction": instruction,
        "solution": {
            "reasoning_path": reasoning_steps,
            "final_answer": generate_final_answer(reasoning_steps)
        },
        "api_documentation": format_multi_tool_docs(tool_chain),
        "mode": "multi-step"
    }
    
    return example
```

**Example Multi-Step Output:**

```json
{
  "instruction": "Calculate the mean temperature from the 'climate_sim.h5' dataset for the year 2023",
  "solution": {
    "reasoning_path": [
      {
        "step": 1,
        "thought": "First, I need to open the HDF5 file containing the climate simulation data",
        "tool": "open_file",
        "arguments": {
          "filename": "climate_sim.h5",
          "mode": "r"
        },
        "expected_result": "File object for climate_sim.h5"
      },
      {
        "step": 2,
        "thought": "Next, I need to read the temperature dataset for 2023",
        "tool": "read_dataset",
        "arguments": {
          "file": "<file from step 1>",
          "dataset_path": "/2023/temperature"
        },
        "expected_result": "NumPy array of temperature values"
      },
      {
        "step": 3,
        "thought": "Finally, I'll calculate the mean of the temperature values",
        "tool": "calculate_mean",
        "arguments": {
          "data": "<array from step 2>"
        },
        "expected_result": "Mean temperature value"
      }
    ],
    "final_answer": "The mean temperature from climate_sim.h5 for 2023 is calculated by: (1) opening the file, (2) reading the temperature dataset, and (3) computing the mean of those values."
  },
  "api_documentation": "--- open_file(filename, mode) ---\n...\n\n--- read_dataset(file, dataset_path) ---\n...\n\n--- calculate_mean(data) ---\n...",
  "mode": "multi-step"
}
```

#### **Mode 3: Auto (Balanced Mix)**

**Goal:** Automatically generate appropriate mix based on tool complexity

```python
def generate_auto_mode(tools, target_examples, llm, prompts):
    examples = []
    
    # Classify tools by complexity
    simple_tools = [t for t in tools if t.complexity == "simple"]
    medium_tools = [t for t in tools if t.complexity == "medium"]
    complex_tools = [t for t in tools if t.complexity == "complex"]
    
    # Distribution strategy
    n_single = int(target_examples * 0.4)   # 40% simple
    n_medium = int(target_examples * 0.35)  # 35% medium
    n_multi = target_examples - n_single - n_medium  # 25% complex
    
    # Generate single-step examples
    for tool in random.sample(simple_tools, min(n_single, len(simple_tools))):
        examples.append(generate_single_step_example(tool, llm, prompts))
    
    # Generate medium examples (single or paired)
    for i in range(n_medium):
        tools_subset = random.sample(medium_tools, k=random.choice([1, 2]))
        if len(tools_subset) == 1:
            examples.append(generate_single_step_example(tools_subset[0], llm, prompts))
        else:
            examples.append(generate_multi_step_example(tools_subset, llm, prompts))
    
    # Generate multi-step examples
    for i in range(n_multi):
        chain_length = random.randint(2, 4)
        tools_subset = select_tool_chain(tools, min_tools=2, max_tools=chain_length)
        examples.append(generate_multi_step_example(tools_subset, llm, prompts))
    
    return examples
```

### Step 3: Execution Verification

**Verification Strategy (APIGen's Triple Verification):**

```python
def verify_tool_example(example, mode="simulated"):
    verification = {
        "format_valid": False,
        "execution_valid": False,
        "semantic_valid": False,
        "errors": []
    }
    
    # 1. FORMAT VERIFICATION
    try:
        validate_schema(example)
        verification["format_valid"] = True
    except ValidationError as e:
        verification["errors"].append(f"Format error: {e}")
        return verification
    
    # 2. EXECUTION VERIFICATION
    if mode == "simulated":
        # Generate plausible outputs without real execution
        for step in example["solution"]["reasoning_path"]:
            step["actual_result"] = simulate_execution(step)
            step["status"] = "success"
        verification["execution_valid"] = True
        
    elif mode == "real":
        # Actually execute tool calls
        try:
            context = {}  # Store results between steps
            for step in example["solution"]["reasoning_path"]:
                result = execute_tool(step["tool"], step["arguments"], context)
                step["actual_result"] = result
                step["status"] = "success"
                context[f"step_{step['step']}"] = result
            verification["execution_valid"] = True
        except Exception as e:
            step["status"] = "failed"
            step["error"] = str(e)
            verification["errors"].append(f"Execution error: {e}")
            return verification
    
    # 3. SEMANTIC VERIFICATION
    semantic_check = llm.generate(f"""
    Verify if this solution correctly addresses the instruction:
    
    Instruction: {example["instruction"]}
    Solution steps: {example["solution"]["reasoning_path"]}
    
    Does each step logically follow from the previous?
    Does the final result address the original instruction?
    
    Answer: yes or no, with brief reasoning
    """)
    
    verification["semantic_valid"] = "yes" in semantic_check.lower()
    if not verification["semantic_valid"]:
        verification["errors"].append(f"Semantic check failed: {semantic_check}")
    
    return verification
```

**Simulated Execution (for testing without real tools):**

```python
def simulate_execution(step):
    """Generate plausible output based on tool and arguments."""
    
    tool_name = step["tool"]
    arguments = step["arguments"]
    
    # Template responses for common operations
    simulators = {
        "open_file": lambda args: f"<HDF5 file '{args['filename']}' (mode {args['mode']})>",
        "read_dataset": lambda args: f"<NumPy array shape={args.get('shape', '(1000,)')} dtype=float64>",
        "list_datasets": lambda args: ["dataset1", "dataset2", "dataset3"],
        "get_attributes": lambda args: {"created": "2024-01-01", "version": "1.0"},
        "calculate_mean": lambda args: 23.456,
        "calculate_std": lambda args: 1.234,
    }
    
    if tool_name in simulators:
        return simulators[tool_name](arguments)
    else:
        return f"<{tool_name} result>"
```

---

## Quality Control Pipeline

### Stage 1: Pre-Filtering (During Generation)

**Automatic Filters:**

```python
def validate_qa_pair(pair):
    """Basic quality checks during generation."""
    issues = []
    
    # Check for empty fields
    if not pair.get("question", "").strip():
        issues.append("Empty question")
    if not pair.get("answer", "").strip():
        issues.append("Empty answer")
    
    # Check minimum length
    if len(pair.get("question", "")) < 10:
        issues.append("Question too short (< 10 chars)")
    if len(pair.get("answer", "")) < 20:
        issues.append("Answer too short (< 20 chars)")
    
    # Check for common problems
    question = pair.get("question", "").lower()
    if not any(q in question for q in ["what", "how", "why", "when", "where", "which", "who"]):
        issues.append("Question doesn't start with question word")
    
    # Check for yes/no questions (discouraged)
    if question.startswith(("is ", "are ", "do ", "does ", "can ", "will ")):
        issues.append("Yes/no question (low information)")
    
    return len(issues) == 0, issues
```

### Stage 2: LLM-as-Judge Curation

**Rating Process (from `curate.py`):**

```python
def curate_qa_pairs(qa_pairs, llm_config, threshold=7.0):
    llm = get_client(llm_config["provider"], llm_config)
    prompts = load_prompts("configs/prompts")
    
    rated_pairs = []
    
    for batch in batched(qa_pairs, batch_size=10):
        # Format for rating
        rating_prompt = prompts["qa_rating"].format(
            pairs=json.dumps(batch, indent=2)
        )
        
        # Get ratings
        response = llm.generate(rating_prompt, temperature=0.3)
        ratings = json5.loads(response)
        
        # Apply ratings and filter
        for pair, rating in zip(batch, ratings):
            pair.update(rating)  # Add rating fields
            
            if pair["rating"] >= threshold:
                rated_pairs.append(pair)
            else:
                console.print(
                    f"[dim]Filtered (rating={pair['rating']}): "
                    f"{pair['question'][:50]}...[/dim]"
                )
    
    pass_rate = len(rated_pairs) / len(qa_pairs) * 100
    console.print(f"""
    [cyan]Curation complete:[/cyan]
    - Threshold: {threshold}/10
    - Pass rate: {pass_rate:.1f}%
    - Kept: {len(rated_pairs)} / {len(qa_pairs)} pairs
    """)
    
    return rated_pairs
```

**Rating Criteria (from `configs/prompts/qa_rating.yaml`):**

```yaml
qa_rating: |
  Rate these question-answer pairs on a scale of 1-10 using these criteria:
  
  CLARITY (0-3 points):
  - Is the question clear and unambiguous?
  - Is the answer easy to understand?
  - 0: Confusing, 1: Somewhat clear, 2: Clear, 3: Very clear
  
  ACCURACY (0-3 points):
  - Is the answer factually correct?
  - Is it supported by evidence in the text?
  - 0: Wrong, 1: Partially correct, 2: Correct, 3: Completely accurate
  
  USEFULNESS (0-2 points):
  - Would this help train a language model?
  - Does it teach something valuable?
  - 0: Not useful, 1: Somewhat useful, 2: Very useful
  
  DIFFICULTY (0-2 points):
  - Is the complexity appropriate?
  - 0: Too simple/trivial, 1: Appropriate, 2: Good complexity
  
  TOTAL RATING = Clarity + Accuracy + Usefulness + Difficulty (max 10)
  
  Rate STRICTLY. Most pairs should score 5-7. Only exceptional pairs score 8-10.
  
  For each pair, provide:
  - Individual criteria scores
  - Total rating (sum)
  - Brief reasoning explaining the rating
  
  Return valid JSON:
  [
    {
      "question": "...",
      "answer": "...",
      "rating": 7,
      "clarity": 2,
      "accuracy": 3,
      "usefulness": 1,
      "difficulty": 1,
      "reasoning": "Clear question with accurate answer. Good for training but basic difficulty."
    },
    ...
  ]
```

### Stage 3: Format-Specific Validation

**CoT Validation:**

```python
def validate_cot_pair(pair):
    """Validate Chain-of-Thought specific requirements."""
    issues = []
    
    # Check reasoning field exists
    if "reasoning" not in pair:
        issues.append("Missing reasoning field")
        return False, issues
    
    reasoning = pair["reasoning"]
    
    # Check reasoning is list
    if not isinstance(reasoning, list):
        issues.append("Reasoning should be a list of steps")
    
    # Check number of steps
    if len(reasoning) < 2:
        issues.append("Too few reasoning steps (minimum 2)")
    elif len(reasoning) > 7:
        issues.append("Too many reasoning steps (maximum 7)")
    
    # Check each step
    for i, step in enumerate(reasoning, 1):
        if not step.strip():
            issues.append(f"Step {i} is empty")
        if len(step) < 15:
            issues.append(f"Step {i} too short")
    
    return len(issues) == 0, issues
```

**Tool-Use Validation:**

```python
def validate_tool_example(example):
    """Validate tool-use specific requirements."""
    issues = []
    
    # Check required fields
    required = ["instruction", "solution", "api_documentation"]
    for field in required:
        if field not in example:
            issues.append(f"Missing required field: {field}")
    
    solution = example.get("solution", {})
    
    # Check reasoning path exists
    if "reasoning_path" not in solution:
        issues.append("Missing reasoning_path in solution")
        return False, issues
    
    # Validate each step
    for step in solution["reasoning_path"]:
        if "tool" not in step:
            issues.append(f"Step {step.get('step')} missing tool")
        if "arguments" not in step:
            issues.append(f"Step {step.get('step')} missing arguments")
        if "thought" not in step:
            issues.append(f"Step {step.get('step')} missing thought")
    
    return len(issues) == 0, issues
```

---

## Format Conversion and Export

### Supported Formats

**1. ChatML (Hugging Face, Axolotl)**

```python
def to_chatml(qa_pair):
    messages = [
        {"role": "user", "content": qa_pair["question"]},
        {"role": "assistant", "content": qa_pair["answer"]}
    ]
    
    # Add system message if available
    if "system" in qa_pair:
        messages.insert(0, {"role": "system", "content": qa_pair["system"]})
    
    # Add reasoning as assistant thoughts (for CoT)
    if "reasoning" in qa_pair:
        reasoning_text = "\n".join([
            f"Step {i}: {step}"
            for i, step in enumerate(qa_pair["reasoning"], 1)
        ])
        messages.insert(1, {
            "role": "assistant",
            "content": f"Let me think through this step by step:\n\n{reasoning_text}"
        })
    
    return {"messages": messages}
```

**Example ChatML Output:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": "How does HDF5 handle parallel I/O?"
    },
    {
      "role": "assistant",
      "content": "Let me think through this step by step:\n\nStep 1: HDF5 uses MPI for coordinating parallel access\nStep 2: Collective operations allow multiple processes to write simultaneously\nStep 3: Chunking strategy prevents conflicts between processes"
    },
    {
      "role": "assistant",
      "content": "HDF5 handles parallel I/O through MPI collective operations..."
    }
  ]
}
```

**2. Alpaca (Stanford Alpaca, llama.cpp)**

```python
def to_alpaca(qa_pair):
    result = {
        "instruction": qa_pair["question"],
        "input": "",  # Usually empty for QA
        "output": qa_pair["answer"]
    }
    
    # Add reasoning as part of output (for CoT)
    if "reasoning" in qa_pair:
        reasoning_text = "\n".join(qa_pair["reasoning"])
        result["output"] = f"Reasoning:\n{reasoning_text}\n\nAnswer:\n{qa_pair['answer']}"
    
    return result
```

**3. ShareGPT (FastChat, Vicuna)**

```python
def to_sharegpt(qa_pair):
    conversations = [
        {"from": "human", "value": qa_pair["question"]},
        {"from": "gpt", "value": qa_pair["answer"]}
    ]
    
    # Add reasoning conversation (for CoT)
    if "reasoning" in qa_pair:
        reasoning_text = "\n".join(qa_pair["reasoning"])
        conversations.insert(1, {
            "from": "gpt",
            "value": f"Let me reason through this:\n{reasoning_text}"
        })
    
    return {"conversations": conversations}
```

**4. Tool-Use Format (Custom)**

```python
def to_tool_use_format(tool_example):
    """Format specifically for tool-calling fine-tuning."""
    
    messages = [
        {"role": "user", "content": tool_example["instruction"]}
    ]
    
    # Add API documentation as system message
    messages.insert(0, {
        "role": "system",
        "content": f"You have access to these tools:\n\n{tool_example['api_documentation']}"
    })
    
    # Format reasoning path
    reasoning_path = tool_example["solution"]["reasoning_path"]
    for step in reasoning_path:
        # Thought
        messages.append({
            "role": "assistant",
            "content": f"Thought: {step['thought']}"
        })
        
        # Tool call
        tool_call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": step["tool"],
                    "arguments": json.dumps(step["arguments"])
                }
            }]
        }
        messages.append(tool_call)
        
        # Tool result
        if "actual_result" in step:
            messages.append({
                "role": "tool",
                "content": json.dumps(step["actual_result"]),
                "tool_call_id": f"call_{step['step']}"
            })
    
    # Final answer
    messages.append({
        "role": "assistant",
        "content": tool_example["solution"].get("final_answer", "")
    })
    
    return {"messages": messages}
```

### Export Process

```python
def export_to_format(data, output_path, format_name):
    """Export data to specified format."""
    
    formatters = {
        "chatml": to_chatml,
        "alpaca": to_alpaca,
        "sharegpt": to_sharegpt,
        "tool_use": to_tool_use_format,
        "jsonl": lambda x: x  # No transformation
    }
    
    if format_name not in formatters:
        raise ValueError(f"Unknown format: {format_name}")
    
    formatter = formatters[format_name]
    
    # Convert each item
    formatted_data = [formatter(item) for item in data]
    
    # Save based on format
    if format_name == "jsonl":
        # One JSON object per line
        with open(output_path, 'w') as f:
            for item in formatted_data:
                f.write(json.dumps(item) + '\n')
    else:
        # Regular JSON array
        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2)
    
    console.print(f"[green]✓ Exported {len(formatted_data)} items to {output_path} ({format_name})[/green]")
```

---

## Summary: Complete Data Flow Example

**Starting Point: Research Papers on HDF5**

```bash
# 1. Pre-processing (Phagocyte pipeline)
phagocyte ingest batch ./hdf5_papers -o ./markdown
phagocyte process run ./markdown -o ./lancedb_hdf5

# Result: 250 document chunks in LanceDB
```

**Generator Processing:**

```bash
# 2. Generate QA pairs (Instruction Backtranslation)
uv run generator generate ./lancedb_hdf5 -o qa_raw.json --target-pairs 300 --topic "HDF5"
# Result: 287 QA pairs generated

# 3. Enrich responses
uv run generator enrich qa_raw.json -o qa_enriched.json
# Result: Improved formatting and clarity

# 4. Curate by quality (LLM-as-Judge)
uv run generator curate qa_enriched.json -o qa_curated.json --threshold 7.0
# Result: 218 high-quality pairs (76% pass rate)

# 5. Add CoT reasoning
uv run generator enhance-cot qa_curated.json -o qa_cot.json
# Result: 218 pairs with step-by-step reasoning

# 6. Export to training format
uv run generator export qa_cot.json -o hdf5_training.jsonl -f chatml
# Result: Ready for fine-tuning
```

**Final Output Statistics:**

- **Started with:** 250 document chunks
- **Generated:** 287 QA pairs
- **After curation:** 218 high-quality pairs (rating ≥ 7/10)
- **With CoT:** 218 reasoning-enhanced pairs
- **Pass rate:** 76%
- **Avg rating:** 7.8/10
- **Avg reasoning steps:** 3.4 steps/pair

**Training Dataset Ready:** `hdf5_training.jsonl` (ChatML format, 218 examples)

---

## Best Practices

### 1. Start Small, Scale Up

```bash
# Test with limited chunks first
uv run generator generate ./lancedb -o test.json --max-chunks 10 --target-pairs 20

# Once satisfied, scale up
uv run generator generate ./lancedb -o full.json --target-pairs 500
```

### 2. Monitor Quality Continuously

```bash
# Check intermediate outputs
jq '.[] | {q: .question, r: .rating}' qa_curated.json | head -20

# Analyze rating distribution
jq '[.[] | .rating] | add / length' qa_curated.json
```

### 3. Use Appropriate Providers

- **Testing/Development:** Ollama (free, local)
- **Production/Quality:** Claude or Gemini (higher quality)
- **Cost-Conscious:** Gemini free tier (10 req/min)
- **High Volume:** vLLM (self-hosted, unlimited)

### 4. Checkpoint Management

```bash
# If generation fails, it auto-resumes
uv run generator generate ./lancedb -o qa.json --target-pairs 500
# (Interrupted after 300 pairs)

# Re-run same command - resumes from checkpoint
uv run generator generate ./lancedb -o qa.json --target-pairs 500
# Starts from pair 301
```

---

**Document Version:** 1.0  
**Last Updated:** January 8, 2026  
**Maintained By:** Shazzadul
