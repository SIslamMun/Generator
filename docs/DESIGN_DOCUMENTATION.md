# Generator Design Documentation

**Author:** Shazzadul  
**Date:** January 8, 2026  
**Purpose:** Comprehensive documentation of design decisions, research foundations, and implementation methodology

---

## Table of Contents

1. [Research Foundation](#1-research-foundation)
2. [Design Philosophy](#2-design-philosophy)
3. [Architecture Overview](#3-architecture-overview)
4. [Data Extraction Methodology](#4-data-extraction-methodology)
5. [Generation Pipelines](#5-generation-pipelines)
6. [Quality Assurance](#6-quality-assurance)
7. [Multi-Provider LLM Support](#7-multi-provider-llm-support)
8. [Implementation Decisions](#8-implementation-decisions)

---

## 1. Research Foundation

### 1.1 Core Papers and Their Contributions

The Generator is designed based on a curated progression of research papers, each addressing specific challenges in synthetic training data generation:

#### **Paper 1: LIMA - Less Is More for Alignment**
- **Reference:** Zhou et al., Meta AI, NeurIPS 2023
- **ArXiv:** https://arxiv.org/abs/2305.11206
- **Key Insight:** 1,000 carefully curated examples can rival 100K+ low-quality examples
- **Application in Generator:**
  - Strict quality thresholds (default: ≥7.0/10) in `curate.py`
  - Detailed rating criteria (Clarity, Accuracy, Usefulness, Difficulty)
  - Focus on quality over quantity in all pipelines
  - Rationale field to explain each rating decision

#### **Paper 2: Self-Alignment with Instruction Backtranslation**
- **Reference:** Li et al., Meta AI, ICLR 2024
- **ArXiv:** https://arxiv.org/abs/2308.06259
- **Key Insight:** Treat high-quality documents as "answers," generate "questions" for them
- **Application in Generator:**
  - Primary methodology in `qa_generator.py`
  - Documents from LanceDB → Extract chunks → Generate questions
  - Preserves expert knowledge from original documents
  - Implementation in `generate` command with target-based generation

**Why Instruction Backtranslation?**
- Documents already contain expert knowledge (research papers, documentation, code)
- Generating questions is easier than generating high-quality answers
- Maintains factual accuracy (answers are grounded in real documents)
- Efficient scaling: can generate multiple questions per chunk

#### **Paper 3: Distilling Step-by-Step**
- **Reference:** Hsieh et al., Google Research, 2023
- **ArXiv:** https://arxiv.org/abs/2305.02301
- **Key Insight:** Extract chain-of-thought rationales to enable small models to match large model reasoning
- **Application in Generator:**
  - `cot_generator.py` - Generates QA pairs with step-by-step reasoning
  - `cot_enhancer.py` - Adds CoT reasoning to existing QA pairs
  - Reasoning fields: `reasoning`, `thought_process`, `step_by_step`
  - Enables 7-13B models to perform complex reasoning

#### **Paper 4: AlpaGasus - Filtering and Reranking**
- **Reference:** Chen et al., UMD, ICLR 2024
- **ArXiv:** https://arxiv.org/abs/2307.08701
- **Key Insight:** LLM-as-Judge with detailed criteria consistently beats automated filtering
- **Application in Generator:**
  - Multi-dimensional rating system in `curate.py`
  - Per-criteria scoring breakdown (Clarity 0-3, Accuracy 0-3, etc.)
  - Threshold-based filtering with reasoning explanations
  - Supports both QA and CoT formats

#### **Paper 5: Toolformer - Self-Supervised API Learning**
- **Reference:** Schick et al., Meta AI, NeurIPS 2023
- **ArXiv:** https://arxiv.org/abs/2302.04761
- **Key Insight:** Models can learn to use tools through self-supervised filtering
- **Application in Generator:**
  - Foundation for `tool_generator.py` single-step mode
  - Simple tool calls with immediate verification
  - Format: Instruction → Single Tool → Result

#### **Paper 6: Gorilla - Retrieval-Aware API Training**
- **Reference:** Patil et al., UC Berkeley, NeurIPS 2024
- **ArXiv:** https://arxiv.org/abs/2305.15334
- **Key Insight:** Always include API documentation in training reduces hallucination by <7%
- **Application in Generator:**
  - Every tool-use example includes API documentation
  - `api_documentation` field in all tool examples
  - Grounding mechanism prevents invented APIs

#### **Paper 7: ToolLLM - Multi-Step Reasoning**
- **Reference:** Qin et al., Tsinghua University, ICLR 2024
- **ArXiv:** https://arxiv.org/abs/2307.16789
- **Key Insight:** DFSDT (Depth-First Search Decision Tree) enables complex multi-tool workflows
- **Application in Generator:**
  - `tool_generator.py` multi-step mode
  - Reasoning chains with intermediate steps
  - Each step: thought → tool selection → execution → next step
  - Pass rate tracking and verification

### 1.2 Design Decision Matrix

| Challenge | Paper Used | Implementation |
|-----------|-----------|----------------|
| Limited training data | LIMA | Strict quality filtering (≥7/10) |
| Need domain knowledge | Instruction Backtranslation | QA from documents in LanceDB |
| Small model reasoning | Distilling Step-by-Step | CoT generation and enhancement |
| Quality assurance | AlpaGasus | LLM-as-Judge with detailed criteria |
| Single-tool calling | Toolformer | Single-step tool examples |
| API hallucination | Gorilla | Documentation grounding |
| Multi-tool workflows | ToolLLM | Multi-step reasoning chains |

---

## 2. Design Philosophy

### 2.1 Quality Over Quantity (LIMA Principle)

**Research Basis:** LIMA demonstrated that 1,000 high-quality examples can match or exceed the performance of models trained on 100,000+ lower-quality examples.

**Implementation:**
```python
# curate.py - Strict quality thresholds
DEFAULT_THRESHOLD = 7.0  # Only keep pairs rated 7/10 or higher

# Detailed criteria scoring (sum = rating)
- Clarity (0-3): Question clear? Answer understandable?
- Accuracy (0-3): Answer correct and supported by evidence?
- Usefulness (0-2): Valuable for training?
- Difficulty (0-2): Appropriate complexity?
```

**Rationale:**
- Small models benefit more from quality than quantity
- High-quality data prevents "model collapse" in fine-tuning
- Efficient use of compute resources
- Better generalization from fewer examples

### 2.2 Ground Truth from Documents (Instruction Backtranslation)

**Research Basis:** Meta's Instruction Backtranslation showed that treating existing high-quality text as "answers" and generating "questions" produces superior training data.

**Why This Approach:**

1. **Factual Accuracy:** Answers are from real documents (research papers, documentation)
2. **Domain Expertise:** Preserves expert knowledge without hallucination
3. **Scalability:** Can generate multiple questions per document chunk
4. **Cost-Effective:** Easier to generate good questions than good answers

**Data Flow:**
```
Documents → Parser → Ingestor → Markdown
    ↓
Processor → Chunks → LanceDB (with embeddings)
    ↓
Generator → Questions ← Chunks (as answers)
    ↓
QA Pairs (grounded in source documents)
```

### 2.3 Progressive Enhancement Pipeline

**Research Basis:** Combining multiple papers' methodologies in sequence yields better results than any single approach.

**Pipeline Design:**
```
1. GENERATE (Instruction Backtranslation)
   Generate QA pairs from document chunks
   ↓
2. ENRICH (Response Rewriting)
   Improve answer formatting while preserving content
   ↓
3. CURATE (LLM-as-Judge)
   Filter by quality using detailed criteria
   ↓
4. ENHANCE (Optional - Distilling Step-by-Step)
   Add chain-of-thought reasoning to best pairs
   ↓
5. EXPORT (Multi-Format)
   Convert to training format (ChatML, Alpaca, ShareGPT, JSONL)
```

**Rationale:**
- Each stage improves specific quality aspects
- Can run full pipeline or individual stages
- Intermediate saves enable debugging and inspection
- Modular design allows experimentation

---

## 3. Architecture Overview

### 3.1 Module Structure

```
Generator/
├── src/generator/
│   ├── clients/           # Multi-provider LLM support
│   │   ├── base.py        # Abstract interface
│   │   ├── ollama.py      # Local Ollama
│   │   ├── claude.py      # Anthropic Claude
│   │   ├── gemini.py      # Google Gemini
│   │   ├── vllm.py        # vLLM server
│   │   └── openai.py      # OpenAI
│   │
│   ├── qa_generator.py    # QA pair generation (Instruction Backtranslation)
│   ├── cot_generator.py   # CoT generation (Distilling Step-by-Step)
│   ├── cot_enhancer.py    # Add CoT to existing QA
│   ├── enrich.py          # Response rewriting for quality
│   ├── curate.py          # LLM-as-Judge filtering
│   ├── compare.py         # Side-by-side comparison UI
│   │
│   ├── tool_parser.py     # Parse API definitions (OpenAPI/JSON)
│   ├── tool_generator.py  # Generate tool-use examples (Toolformer + ToolLLM)
│   ├── tool_executor.py   # Verify tool calls (APIGen methodology)
│   ├── tool_curator.py    # Filter tool examples
│   │
│   ├── formatters.py      # Export to ChatML, Alpaca, ShareGPT, JSONL
│   ├── prompt_loader.py   # Load prompt templates
│   └── cli.py             # Command-line interface
│
├── configs/
│   ├── config.yaml        # LLM provider configuration
│   └── prompts/           # Prompt templates
│       ├── qa_generation.yaml
│       ├── qa_rating.yaml
│       ├── cot_generation.yaml
│       └── tool_prompts.yaml
│
└── tests/                 # Comprehensive test suite
    ├── test_clients.py
    ├── test_qa_generator.py
    ├── test_cot_generator.py
    └── ...
```

### 3.2 Design Patterns

#### Pattern 1: Strategy Pattern for LLM Providers
```python
# base.py - Abstract interface
class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

# Concrete implementations
class OllamaClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        # Ollama-specific implementation
        
class ClaudeClient(BaseLLMClient):
    def generate(self, prompt: str) -> str:
        # Claude-specific implementation
```

**Benefits:**
- Easy to add new providers
- Switch providers via configuration
- Consistent interface across all LLMs

#### Pattern 2: Template Method for Generation
```python
# Common pattern in qa_generator.py, cot_generator.py, tool_generator.py
def generate_from_source(source, output, config):
    # 1. Load source data
    data = load_data(source)
    
    # 2. Process in batches
    for batch in batched(data, batch_size):
        results = generate_batch(batch)
        
        # 3. Save intermediate results
        save_checkpoint(results)
    
    # 4. Aggregate and save final
    save_final(output)
```

**Benefits:**
- Consistent processing across pipelines
- Intermediate saves prevent data loss
- Easy to debug and resume

#### Pattern 3: Decorator Pattern for Retry Logic
```python
# In all clients - exponential backoff
def generate_with_retry(self, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return self.generate(prompt)
        except RateLimitError:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
```

---

## 4. Data Extraction Methodology

### 4.1 From Documents to Training Data

#### **Phase 1: Document Acquisition** (Pre-Generator)

**Source:** Phagocyte's existing pipeline
- Research papers (PDFs via Unpaywall)
- GitHub repositories (code + documentation)
- Web pages (documentation sites)
- Technical documentation

**Tools:**
- `phagocyte research` - AI research assistant
- `phagocyte parse` - Extract references and DOIs
- `phagocyte ingest` - PDF/Web/Code to markdown

#### **Phase 2: Processing to LanceDB** (Pre-Generator)

**Chunking Strategy:**
```python
# processor splits documents into semantic chunks
- Text chunks: 512-1024 tokens
- Code chunks: Function-level or class-level
- Metadata: source file, chunk ID, document type
```

**Storage in LanceDB:**
```python
{
    "id": "chunk_001",
    "text": "HDF5 is a data model...",  # Original chunk text
    "vector": [0.123, 0.456, ...],   # Embedding
    "source": "papers/hdf5_intro.md",
    "doc_type": "research_paper"
}
```

#### **Phase 3: QA Generation** (Generator)

**Extraction Process in `qa_generator.py`:**

```python
def generate_qa_from_lancedb(db_path, ...):
    # 1. Connect to LanceDB
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)  # "text_chunks" or "code_chunks"
    
    # 2. Load chunks (with optional filters)
    chunks = table.to_pandas()
    
    # 3. Calculate per-chunk generation
    if target_pairs:
        n_pairs_per_chunk = target_pairs // len(chunks)
    
    # 4. Process in batches
    for batch in batched(chunks, batch_size):
        for chunk in batch:
            # 5. Generate questions for this chunk
            qa_pairs = generate_pairs_for_chunk(
                chunk_text=chunk["text"],
                n_pairs=n_pairs_per_chunk,
                prompt=prompts["qa_generation"]
            )
            
            # 6. Store with metadata
            for qa in qa_pairs:
                qa["source_chunk_id"] = chunk["id"]
                qa["source_file"] = chunk["source"]
                results.append(qa)
        
        # 7. Save intermediate checkpoint
        save_checkpoint(results)
    
    return results
```

**Prompt Engineering (from `qa_generation.yaml`):**

```yaml
qa_generation: |
  You are an expert at generating high-quality question-answer pairs from documents.
  
  Given this document chunk, generate {n_pairs} question-answer pairs where:
  - Questions are clear, specific, and answerable from the text
  - Answers come directly from the text (quote or paraphrase)
  - Cover different aspects and difficulty levels
  
  Document chunk:
  {chunk_text}
  
  Generate exactly {n_pairs} QA pairs in JSON format:
  [
    {{"question": "...", "answer": "..."}},
    ...
  ]
```

**Key Design Decisions:**

1. **Target-Based Generation** (Added Jan 6, 2026)
   - User specifies total pairs wanted (e.g., 300)
   - System calculates per-chunk: `300 / num_chunks`
   - Ensures balanced distribution across document

2. **Batch Processing**
   - Default batch_size: 50 chunks
   - Balances memory usage and API efficiency
   - Progress tracking with Rich library

3. **Intermediate Saves**
   - Checkpoint every batch
   - Prevents data loss on errors
   - Enables inspection during generation

4. **Topic Filtering** (Added Jan 6, 2026)
   ```python
   # Optional post-generation filtering
   if topic:
       qa_pairs = filter_off_topic(qa_pairs, topic)
   ```
   - Removes pairs not relevant to specified topic
   - Uses LLM to judge relevance
   - Maintains focus on target domain

### 4.2 Chain-of-Thought (CoT) Generation and Enhancement

**Based on:** "Distilling Step-by-Step: Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes" (Google Research, 2023)

#### **What is CoT and Why Use It?**

Chain-of-Thought (CoT) training enables small models to match or exceed the reasoning capabilities of much larger models. Instead of just learning question→answer mappings, models learn the intermediate reasoning steps that lead to correct answers.

**Key Benefits:**
- **Smaller models, better performance:** Small models with CoT can outperform larger models without CoT
- **Data efficiency:** Requires significantly less training data to achieve good performance
- **Interpretability:** Reasoning steps make model decisions transparent and debuggable
- **Transfer learning:** Reasoning patterns generalize better to new domains

**Research Foundation:**
The "Distilling Step-by-Step" paper demonstrated that:
- A 770M parameter T5 model with CoT training outperformed 540B parameter PaLM model
- Required 12.5× less training data than standard fine-tuning
- Reasoning quality improved dramatically on complex tasks

#### **Method 1: Direct CoT Generation** (`cot_generator.py`)

**Purpose:** Generate QA pairs with built-in reasoning from scratch

**When to Use:**
- Starting from raw documents/chunks
- Need reasoning-focused training data from beginning
- Want questions specifically designed to require multi-step thinking

**Process:**
```python
def generate_cot_pairs(lancedb_path, output_path, llm_config, ...):
    """Generate QA pairs with step-by-step reasoning from chunks."""
    
    # Load chunks from LanceDB (same as QA generation)
    chunks = load_chunks(lancedb_path)
    
    for chunk in chunks:
        # Generate with explicit reasoning structure request
        cot_pair = llm.generate(
            prompt=f"""
            Given this content, create a question that requires reasoning.
            Provide:
            1. A question that needs multiple steps to answer
            2. Step-by-step reasoning (2-5 clear steps)
            3. Final answer based on the reasoning
            
            Make reasoning steps:
            - Break down the problem logically
            - Show intermediate conclusions
            - Build progressively toward answer
            
            Content: {chunk.text}
            
            Return JSON with: question, reasoning (array), answer
            """,
            temperature=0.7  # Slightly higher for creative reasoning
        )
        
        # Structure output with reasoning fields
        result = {
            "question": cot_pair["question"],
            "reasoning": cot_pair["reasoning"],  # List of reasoning steps
            "answer": cot_pair["answer"],
            "chunk_id": chunk.id,
            "cot_generated": True
        }
        
    # Quality validation: ensure reasoning is coherent
    validate_reasoning_quality(results)
```

**Example Output:**
```json
{
  "question": "Why would a scientist choose HDF5 over traditional file formats for storing large datasets?",
  "reasoning": [
    "Traditional file formats like CSV or text files load entirely into memory, which fails for TB-scale datasets",
    "HDF5 provides chunked storage, allowing partial reads of specific data regions without loading everything",
    "HDF5 includes built-in compression at the chunk level, reducing storage costs by 5-10×",
    "HDF5 supports hierarchical organization, enabling scientists to structure complex multi-dimensional data logically",
    "These capabilities together make HDF5 essential for scientific computing at scale"
  ],
  "answer": "Scientists choose HDF5 because it enables efficient partial I/O on massive datasets through chunked storage, provides significant compression savings, and supports complex hierarchical data organization—capabilities that traditional formats cannot provide.",
  "cot_generated": true
}
```

**Prompt Configuration** (`configs/prompts/cot_generation.yaml`):
```yaml
cot_generation: |
  Given the following content, generate a question-answer pair with step-by-step reasoning.
  
  Create a question that:
  - Requires understanding and analysis (not just fact recall)
  - Can be answered using the content below
  - Benefits from breaking down into logical steps
  
  Provide:
  1. A clear, specific question
  2. Step-by-step reasoning (2-5 steps showing the thought process)
  3. A complete answer that follows from the reasoning
  
  Content:
  {chunk_text}
  
  Return JSON format:
  {{
    "question": "your question here",
    "reasoning": [
      "First reasoning step explaining...",
      "Second step building on first...",
      "Continue logical progression..."
    ],
    "answer": "complete answer here"
  }}
```

#### **Method 2: Enhance Existing QA with CoT** (`cot_enhancer.py`)

**Purpose:** Add reasoning steps to already-generated high-quality QA pairs

**When to Use:**
- Already have good QA pairs from initial generation
- Want to add reasoning without regenerating answers
- More efficient than full CoT generation
- Preserve specific question-answer pairings

**Process:**
```python
def enhance_with_cot(input_path, output_path, llm_config, ...):
    """Add chain-of-thought reasoning to existing QA pairs."""
    
    qa_pairs = load_json(input_path)
    enhanced = []
    
    for qa in qa_pairs:
        # Skip if already has reasoning
        if "reasoning" in qa and qa["reasoning"]:
            enhanced.append(qa)
            continue
        
        # Generate reasoning steps for existing Q&A
        reasoning_response = llm.generate(
            prompt=f"""
            Given this question-answer pair, add step-by-step reasoning that explains
            how to arrive at the answer.
            
            QUESTION:
            {qa['question']}
            
            ANSWER:
            {qa['answer']}
            
            Provide 2-5 clear reasoning steps that logically lead to this answer.
            The reasoning should:
            - Break down the problem or question
            - Show intermediate conclusions
            - Build progressively toward the final answer
            - Be coherent with the given answer
            
            Return JSON: {{"reasoning": ["step 1", "step 2", ...]}}
            """,
            temperature=0.5  # Lower temp for consistent enhancement
        )
        
        reasoning_data = json5.loads(reasoning_response)
        
        # Add reasoning to original pair
        enhanced_pair = {
            **qa,
            "reasoning": reasoning_data["reasoning"],
            "cot_enhanced": True,
            "enhanced_at": datetime.now().isoformat()
        }
        enhanced.append(enhanced_pair)
    
    save_json(enhanced, output_path)
    return {"total": len(enhanced), "newly_enhanced": len(enhanced) - len([p for p in qa_pairs if "reasoning" in p])}
```

**Example Enhancement:**

*Before (Simple QA):*
```json
{
  "question": "What is the purpose of HDF5 chunking?",
  "answer": "Chunking allows efficient partial I/O by storing data in small blocks that can be read independently."
}
```

*After (With CoT):*
```json
{
  "question": "What is the purpose of HDF5 chunking?",
  "reasoning": [
    "Large datasets often exceed available memory, making full loads impossible",
    "Applications typically need only specific subsets of data at any time",
    "Chunking divides datasets into smaller blocks stored contiguously on disk",
    "Each chunk can be read independently without accessing the entire dataset",
    "This enables efficient random access patterns and partial I/O operations"
  ],
  "answer": "Chunking allows efficient partial I/O by storing data in small blocks that can be read independently.",
  "cot_enhanced": true
}
```

**Prompt Configuration** (`configs/prompts/cot_enhancement.yaml`):
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
  - Show intermediate steps or conclusions
  - Build progressively toward the final answer
  - Be consistent with the given answer
  
  Return JSON:
  {{
    "reasoning": [
      "First step: [explain initial analysis or observation]",
      "Second step: [build on first step with deeper insight]",
      "Third step: [continue logical progression]",
      ...
    ]
  }}
```

#### **CoT Quality Assurance**

**Validation Checks:**
```python
def validate_cot_quality(pair):
    """Ensure CoT reasoning meets quality standards."""
    
    issues = []
    
    # Check reasoning exists and is non-empty
    if "reasoning" not in pair or not pair["reasoning"]:
        issues.append("Missing reasoning")
    
    # Check minimum reasoning steps
    if len(pair["reasoning"]) < 2:
        issues.append("Too few reasoning steps (need 2+)")
    
    # Check reasoning depth (not just repeating question)
    if any(pair["question"].lower() in step.lower() for step in pair["reasoning"]):
        issues.append("Reasoning repeats question verbatim")
    
    # Check reasoning progresses logically
    # (Simple heuristic: later steps should reference earlier concepts)
    
    # Check answer consistency
    if pair["answer"].lower() not in " ".join(pair["reasoning"]).lower():
        # Answer should connect to reasoning
        pass  # Warning, not error
    
    return issues
```

**Curation Criteria for CoT Data:**
- Reasoning has 2-5 clear, distinct steps
- Each step adds new information or insight
- Steps follow logical progression
- Reasoning is coherent with final answer
- No circular reasoning or tautologies
- Steps are detailed enough to be educational

#### **CLI Commands**

**Generate CoT from Scratch:**
```bash
# Generate CoT pairs from LanceDB chunks
uv run generator generate-cot /path/to/lancedb -o cot_pairs.json --model ollama:llama3.2:70b

# With specific number of pairs per chunk
uv run generator generate-cot /path/to/lancedb -o cot_pairs.json -n 3

# With target total pairs (auto-calculates per-chunk)
uv run generator generate-cot /path/to/lancedb -o cot_pairs.json --target-pairs 1000
```

**Enhance Existing QA:**
```bash
# Add CoT reasoning to existing QA pairs
uv run generator enhance-cot curated_qa.json -o qa_with_cot.json --model claude:sonnet

# Process in batches
uv run generator enhance-cot qa_pairs.json -o enhanced.json --batch-size 10
```

#### **When to Use Each Method**

**Use Direct CoT Generation when:**
- Starting fresh with raw documents
- Want questions specifically designed for reasoning
- Need maximum control over reasoning complexity
- Have sufficient LLM API budget (more expensive)

**Use CoT Enhancement when:**
- Already have high-quality QA pairs
- Want to preserve specific question phrasings
- Need to add reasoning to existing datasets
- Want more cost-effective approach
- Can leverage existing curation work

**Hybrid Approach (Recommended):**
1. Generate base QA pairs (cheaper, faster)
2. Curate to filter best pairs (human judgment)
3. Enhance curated pairs with CoT (preserves quality + adds reasoning)
4. Final curation of CoT data

**Benefits:**
- Preserves high-quality QA pairs from generation phase
- Adds reasoning without re-generating answers
- More efficient than full CoT generation
- Better quality control through multi-stage filtering

### 4.3 Tool-Use Data Extraction

#### **Phase 1: API Parsing** (`tool_parser.py`)

**Input Formats:**
- OpenAPI/Swagger specs
- Custom JSON tool definitions
- Python function signatures

**Extraction Process:**
```python
def parse_tools(api_spec_path):
    spec = load_api_spec(api_spec_path)
    
    tools = []
    for endpoint in spec["paths"]:
        tool = {
            "tool_id": generate_id(endpoint),
            "name": endpoint["operationId"],
            "description": endpoint["summary"],
            "parameters": extract_parameters(endpoint),
            "returns": endpoint["responses"]["200"],
            "examples": endpoint.get("examples", []),
            "complexity": classify_complexity(endpoint)
        }
        tools.append(tool)
    
    return tools
```

**Complexity Classification:**
```python
def classify_complexity(tool):
    # Simple: 0-2 required parameters, no nested objects
    if len(required_params) <= 2 and not has_nested:
        return "simple"
    
    # Complex: Many parameters or nested structures
    elif len(required_params) > 5 or has_nested:
        return "complex"
    
    # Medium: Everything else
    return "medium"
```

#### **Phase 2: Example Generation** (`tool_generator.py`)

**Mode 1: Single-Step (Toolformer approach)**
```python
def generate_single_step(tool):
    # Generate instruction requiring this tool
    instruction = llm.generate(
        prompt=f"""
        Create a user request that would require using this tool:
        
        Tool: {tool.name}
        Description: {tool.description}
        Parameters: {tool.parameters}
        
        The request should be natural and realistic.
        """
    )
    
    # Generate tool call
    solution = {
        "instruction": instruction,
        "reasoning": f"To {instruction}, I need to use {tool.name}",
        "tool_calls": [{
            "tool": tool.name,
            "arguments": generate_realistic_args(tool)
        }],
        "api_documentation": format_api_docs(tool)  # Gorilla
    }
    
    return solution
```

**Mode 2: Multi-Step (ToolLLM DFSDT approach)**
```python
def generate_multi_step(tools):
    # Select 2-4 tools that can chain
    tool_subset = select_related_tools(tools, max_tools=4)
    
    # Generate complex instruction
    instruction = llm.generate(
        prompt=f"""
        Create a complex task requiring these tools in sequence:
        {[t.name for t in tool_subset]}
        
        The task should require multiple steps to complete.
        """
    )
    
    # Generate reasoning chain
    reasoning_chain = []
    for step, tool in enumerate(tool_subset, 1):
        step_data = {
            "step": step,
            "thought": f"Step {step}: Use {tool.name} to...",
            "tool": tool.name,
            "arguments": {},
            "expected_result": "..."
        }
        reasoning_chain.append(step_data)
    
    return {
        "instruction": instruction,
        "reasoning_path": reasoning_chain,
        "api_documentation": format_multi_tool_docs(tool_subset)
    }
```

**Key Innovation - Gorilla's Documentation Grounding:**
```python
def format_api_docs(tool):
    """Include API documentation in every example to prevent hallucination."""
    return f"""
API Documentation:

{tool.name}({', '.join(p.name for p in tool.parameters)})

Description: {tool.description}

Parameters:
{format_parameters(tool.parameters)}

Returns: {tool.returns}

Example:
{tool.examples[0] if tool.examples else generate_example(tool)}
"""
```

#### **Phase 3: Execution Verification** (`tool_executor.py`)

**Based on:** APIGen's triple verification (Format → Execution → Semantic)

```python
def verify_example(example, mode="simulated"):
    results = {
        "format_valid": False,
        "execution_valid": False,
        "semantic_valid": False
    }
    
    # 1. Format verification
    try:
        validate_tool_call_format(example["tool_calls"])
        results["format_valid"] = True
    except ValidationError:
        return results
    
    # 2. Execution verification
    if mode == "simulated":
        # Generate plausible output without actual execution
        for call in example["tool_calls"]:
            call["result"] = simulate_execution(call)
        results["execution_valid"] = True
        
    elif mode == "real":
        # Actually execute tool calls
        try:
            for call in example["tool_calls"]:
                call["result"] = execute_tool(call)
            results["execution_valid"] = True
        except Exception as e:
            call["error"] = str(e)
            return results
    
    # 3. Semantic verification
    # Check if results make sense for the instruction
    semantic_check = llm.generate(
        prompt=f"""
        Does this tool calling sequence correctly solve the task?
        
        Task: {example["instruction"]}
        Tools used: {example["tool_calls"]}
        
        Answer yes/no with brief reasoning.
        """
    )
    results["semantic_valid"] = "yes" in semantic_check.lower()
    
    return results
```

---

## 5. Generation Pipelines

### 5.1 QA Pipeline (Knowledge Training)

**Purpose:** Teach domain knowledge from documents

**Full Pipeline:**
```bash
# 1. Generate QA pairs (Instruction Backtranslation)
uv run generator generate /path/to/lancedb -o qa.json --target-pairs 300

# 2. Enrich answers (Response rewriting)
uv run generator enrich qa.json -o enriched.json

# 3. Curate by quality (LLM-as-Judge)
uv run generator curate enriched.json -o curated.json --threshold 7.0

# 4. Optional: Add CoT reasoning
uv run generator enhance-cot curated.json -o cot_enhanced.json

# 5. Export to training format
uv run generator export cot_enhanced.json -o training.jsonl -f chatml
```

**Or Run Full Pipeline:**
```bash
uv run generator pipeline /path/to/lancedb -o training.jsonl
```

**Internal Flow:**
```
LanceDB chunks
    ↓ [qa_generator.py]
QA pairs (raw)
    ↓ [enrich.py]
QA pairs (improved formatting)
    ↓ [curate.py]
QA pairs (filtered, ≥7/10)
    ↓ [cot_enhancer.py] (optional)
QA pairs (with reasoning)
    ↓ [formatters.py]
Training file (ChatML/Alpaca/ShareGPT)
```

### 5.2 CoT Pipeline (Reasoning Training)

**Purpose:** Train models on step-by-step reasoning

**Direct Generation:**
```bash
# Generate CoT pairs directly
uv run generator generate-cot /path/to/lancedb -o cot.json --target-pairs 100

# Curate
uv run generator curate cot.json -o curated_cot.json --threshold 7.0

# Export
uv run generator export curated_cot.json -o cot_training.jsonl -f chatml
```

**Enhancement of Existing QA:**
```bash
# Add CoT to high-quality QA pairs
uv run generator enhance-cot qa_curated.json -o qa_with_cot.json

# Export
uv run generator export qa_with_cot.json -o training.jsonl -f chatml
```

### 5.3 Tool-Use Pipeline (Agentic Training)

**Purpose:** Train models to use APIs and tools

**Full Pipeline:**
```bash
# 1. Parse API definitions
uv run generator tool-parse configs/hdf5_tools.json -o tools.json

# 2. Generate examples (auto-balances single/multi-step)
uv run generator tool-generate tools.json -o examples.json --target-pairs 100

# 3. Verify execution
uv run generator tool-execute examples.json -o verified.json --mode simulated

# 4. Curate by quality
uv run generator tool-curate verified.json -o curated.json --threshold 7.0

# 5. Export
uv run generator export curated.json -o tool_training.json -f tool_use
```

**Or Run Full Pipeline:**
```bash
uv run generator tool-pipeline configs/hdf5_tools.json -o tool_training.json
```

**Mode Selection:**
```bash
# Only single-step (Toolformer)
uv run generator tool-generate tools.json -o single.json --mode single

# Only multi-step (ToolLLM)
uv run generator tool-generate tools.json -o multi.json --mode multi

# Auto (default) - balanced mix
uv run generator tool-generate tools.json -o balanced.json --mode auto
```

---

## 6. Quality Assurance

### 6.1 LLM-as-Judge (LIMA + AlpaGasus)

**Rating System:**
```yaml
# From configs/prompts/qa_rating.yaml
criteria:
  clarity:
    range: 0-3
    description: Question clear? Answer understandable?
    
  accuracy:
    range: 0-3
    description: Answer correct and supported by evidence?
    
  usefulness:
    range: 0-2
    description: Valuable for training?
    
  difficulty:
    range: 0-2
    description: Appropriate complexity?

total_rating: sum(clarity, accuracy, usefulness, difficulty)  # Max 10
```

**Rating Process in `curate.py`:**
```python
def rate_pairs(pairs, llm, threshold=7.0):
    rated_pairs = []
    
    for batch in batched(pairs, batch_size):
        # Send to LLM for rating
        ratings = llm.generate(
            prompt=format_rating_prompt(batch)
        )
        
        # Parse ratings
        for pair, rating in zip(batch, ratings):
            pair["rating"] = rating["total"]
            pair["clarity"] = rating["clarity"]
            pair["accuracy"] = rating["accuracy"]
            pair["usefulness"] = rating["usefulness"]
            pair["difficulty"] = rating["difficulty"]
            pair["reasoning"] = rating["reasoning"]  # Why this rating?
            
            # Filter by threshold
            if pair["rating"] >= threshold:
                rated_pairs.append(pair)
    
    return rated_pairs
```

**Example Output:**
```json
{
  "question": "What is HDF5 used for?",
  "answer": "HDF5 is used for storing and managing large datasets...",
  "rating": 8,
  "clarity": 3,
  "accuracy": 3,
  "usefulness": 2,
  "difficulty": 0,
  "reasoning": "Clear question with accurate, comprehensive answer. Good training value but basic difficulty."
}
```

### 6.2 Format Detection and Preservation

**Challenge:** Support both QA and CoT formats in curation

**Solution in `curate.py`:**
```python
def detect_format(pair):
    """Detect if QA or CoT format."""
    cot_fields = ["reasoning", "thought_process", "step_by_step"]
    
    if any(field in pair for field in cot_fields):
        return "cot"
    return "qa"

def curate_pairs(pairs, ...):
    # 1. Detect format
    format_type = detect_format(pairs[0])
    
    # 2. Convert to conversation format for rating
    conversations = []
    for pair in pairs:
        if format_type == "cot":
            conv = cot_to_conversation(pair)
        else:
            conv = qa_to_conversation(pair)
        conversations.append(conv)
    
    # 3. Rate conversations
    rated = rate_conversations(conversations)
    
    # 4. Restore original format
    results = []
    for original, rated_conv in zip(pairs, rated):
        if format_type == "cot":
            restored = conversation_to_cot(original, rated_conv)
        else:
            restored = conversation_to_qa(original, rated_conv)
        results.append(restored)
    
    return results
```

### 6.3 Verification Layers (Tool-Use)

**Three-Layer Verification (APIGen methodology):**

1. **Format Verification:**
   ```python
   def validate_format(example):
       required_fields = ["instruction", "tool_calls", "api_documentation"]
       assert all(f in example for f in required_fields)
       
       for call in example["tool_calls"]:
           assert "tool" in call
           assert "arguments" in call
           assert isinstance(call["arguments"], dict)
   ```

2. **Execution Verification:**
   ```python
   def verify_execution(example, mode):
       if mode == "simulated":
           # Generate plausible outputs
           for call in example["tool_calls"]:
               call["result"] = simulate(call)
       
       elif mode == "real":
           # Actually execute
           for call in example["tool_calls"]:
               call["result"] = execute_real(call)
       
       # Check for errors
       return not any("error" in call for call in example["tool_calls"])
   ```

3. **Semantic Verification:**
   ```python
   def verify_semantic(example, llm):
       # Does the solution actually solve the task?
       check = llm.generate(
           prompt=f"""
           Task: {example['instruction']}
           Solution: {example['tool_calls']}
           
           Does this solution correctly complete the task? Yes/No
           """
       )
       return "yes" in check.lower()
   ```

---

## 7. Multi-Provider LLM Support

### 7.1 Provider Architecture

**Design Decision:** Abstract provider interface for easy switching

**Base Interface (`clients/base.py`):**
```python
class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text from prompt."""
        pass
```

**Providers Implemented:**

1. **Ollama** (Local, Free)
   ```python
   class OllamaClient(BaseLLMClient):
       def __init__(self, config):
           self.base_url = config.get("base_url", "http://localhost:11434")
           self.model = config["model"]  # e.g., "mistral:latest"
       
       def generate(self, prompt, temperature, max_tokens):
           response = requests.post(
               f"{self.base_url}/api/generate",
               json={"model": self.model, "prompt": prompt, ...}
           )
           return response.json()["response"]
   ```

2. **Claude** (Cloud, Paid)
   ```python
   class ClaudeClient(BaseLLMClient):
       def __init__(self, config):
           self.api_key = config["api_key"]
           self.model = config["model"]  # e.g., "claude-sonnet-4"
       
       def generate(self, prompt, temperature, max_tokens):
           response = anthropic.messages.create(
               model=self.model,
               max_tokens=max_tokens,
               messages=[{"role": "user", "content": prompt}]
           )
           return response.content[0].text
   ```

3. **Gemini** (Cloud, Free Tier)
   ```python
   class GeminiClient(BaseLLMClient):
       def __init__(self, config):
           self.api_key = config["api_key"]
           self.model = config["model"]  # e.g., "gemini-2.0-flash"
           genai.configure(api_key=self.api_key)
       
       def generate(self, prompt, temperature, max_tokens):
           model = genai.GenerativeModel(self.model)
           response = model.generate_content(
               prompt,
               generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
           )
           return response.text
   ```

4. **vLLM** (Local Server, Free)
5. **OpenAI** (Cloud, Paid)
6. **Anthropic** (Direct API, Paid)

### 7.2 Provider Selection

**Configuration (`configs/config.yaml`):**
```yaml
llm:
  provider: gemini  # Change this to switch providers
  
  ollama:
    base_url: http://localhost:11434
    model: mistral:latest
  
  gemini:
    model: gemini-2.0-flash
    api_key: ${GOOGLE_API_KEY}
  
  claude:
    model: claude-sonnet-4
    api_key: ${ANTHROPIC_API_KEY}
```

**Runtime Override:**
```bash
# Use config file
uv run generator generate db/ -o qa.json

# Override provider
uv run generator generate db/ -o qa.json --provider claude

# Override provider and model
uv run generator generate db/ -o qa.json --provider gemini --model gemini-2.0-flash
```

**Client Factory (`clients/__init__.py`):**
```python
def get_client(provider: str, config: Dict) -> BaseLLMClient:
    clients = {
        "ollama": OllamaClient,
        "claude": ClaudeClient,
        "gemini": GeminiClient,
        "vllm": VLLMClient,
        "openai": OpenAIClient,
        "anthropic": AnthropicClient
    }
    
    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}")
    
    return clients[provider](config)
```

### 7.3 Rate Limiting and Retry Logic

**Common Challenge:** Different providers have different rate limits

**Solution - Exponential Backoff:**
```python
class BaseLLMClient(ABC):
    def generate_with_retry(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                return self.generate(prompt)
                
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                
                # Exponential backoff: 2^attempt seconds
                wait_time = 2 ** attempt
                console.print(f"[yellow]Rate limited, waiting {wait_time}s...[/yellow]")
                time.sleep(wait_time)
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                raise
```

**Provider-Specific Handling:**
```python
# Gemini - 10 requests/minute on free tier
class GeminiClient(BaseLLMClient):
    def __init__(self, config):
        super().__init__(config)
        self.requests_per_minute = 10
        self.request_times = []
    
    def generate(self, prompt, ...):
        # Enforce rate limit
        self._wait_for_rate_limit()
        
        # Make request
        response = super().generate(prompt, ...)
        
        # Track timing
        self.request_times.append(time.time())
        
        return response
    
    def _wait_for_rate_limit(self):
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # If at limit, wait
        if len(self.request_times) >= self.requests_per_minute:
            oldest = self.request_times[0]
            wait_time = 60 - (now - oldest) + 1
            time.sleep(wait_time)
```

---

## 8. Implementation Decisions

### 8.1 Why LanceDB as Input?

**Decision:** Use LanceDB (from Phagocyte's processor) instead of raw documents

**Rationale:**

1. **Pre-processed Data:** Documents already cleaned and chunked
2. **Semantic Chunking:** Processor uses intelligent chunking (not naive splits)
3. **Metadata Preservation:** Source files, document types tracked
4. **Vector Search Ready:** Can do similarity-based sampling if needed
5. **Separation of Concerns:** Processor handles document complexity

**Alternative Considered:**
- Direct from markdown → Rejected because chunking strategy matters
- Direct from PDFs → Rejected because parsing is complex

### 8.2 Why JSON5 for Parsing?

**Decision:** Use `json5` library instead of standard `json`

**Rationale:**

1. **LLM Output Forgiveness:** LLMs sometimes return malformed JSON
   - Trailing commas: `{"key": "value",}` ✓
   - Comments: `// This is a comment` ✓
   - Unquoted keys: `{key: "value"}` ✓

2. **Better Error Messages:** json5 provides clearer parsing errors

3. **Backward Compatible:** Valid JSON is valid JSON5

**Implementation:**
```python
import json5

# Try JSON5 first (forgiving)
try:
    data = json5.loads(llm_response)
except json5.JSONDecodeError:
    # Fall back to standard json for better error
    data = json.loads(llm_response)  # Will raise detailed error
```

### 8.3 Why Batch Processing with Checkpoints?

**Decision:** Process in batches with intermediate saves

**Rationale:**

1. **Resume on Failure:** If generation fails at batch 47, don't lose 1-46
2. **Memory Efficiency:** Don't load entire dataset into memory
3. **Progress Visibility:** Can inspect partial results
4. **Cost Management:** Can stop and resume expensive API calls

**Implementation:**
```python
def generate_with_checkpoints(chunks, batch_size):
    results = []
    checkpoint_file = output.with_suffix(".checkpoint.json")
    
    # Resume from checkpoint if exists
    if checkpoint_file.exists():
        results = json.loads(checkpoint_file.read_text())
        processed = len(results)
        console.print(f"[cyan]Resuming from checkpoint: {processed} pairs[/cyan]")
    else:
        processed = 0
    
    # Process remaining batches
    for i, batch in enumerate(batched(chunks[processed:], batch_size)):
        batch_results = process_batch(batch)
        results.extend(batch_results)
        
        # Save checkpoint
        checkpoint_file.write_text(json.dumps(results, indent=2))
        
        console.print(f"[green]Batch {i+1} complete: {len(results)} total pairs[/green]")
    
    # Remove checkpoint on success
    checkpoint_file.unlink()
    
    return results
```

### 8.4 Why Target-Based Generation?

**Decision:** Add `--target-pairs` option (Jan 6, 2026 update)

**Problem:** Fixed `--n-pairs` per chunk leads to unpredictable totals
- 100 chunks × 5 pairs = 500 pairs
- 200 chunks × 5 pairs = 1000 pairs
- User wants specific total (e.g., "give me 300 pairs")

**Solution:**
```python
if target_pairs:
    n_pairs_per_chunk = max(1, target_pairs // total_chunks)
    console.print(f"[cyan]Target: {target_pairs} pairs → {n_pairs_per_chunk} per chunk[/cyan]")
```

**Benefits:**
- Predictable output size
- Better for controlled datasets
- Easier budget management (API costs)

### 8.5 Why Modular Pipeline?

**Decision:** Allow running individual steps OR full pipeline

**Rationale:**

1. **Debugging:** Can inspect each stage's output
2. **Experimentation:** Try different enrichment strategies
3. **Cost Efficiency:** Skip expensive steps if not needed
4. **Flexibility:** Mix and match approaches

**Both Options Supported:**
```bash
# Option 1: Full pipeline (convenience)
uv run generator pipeline /db -o training.jsonl

# Option 2: Individual steps (control)
uv run generator generate /db -o qa.json
uv run generator enrich qa.json -o enriched.json
uv run generator curate enriched.json -o curated.json
uv run generator export curated.json -o training.jsonl
```

### 8.6 Why Rich for Progress Display?

**Decision:** Use Rich library for terminal output

**Rationale:**

1. **User Experience:** Clear, colorful progress bars
2. **Status Visibility:** See what's happening in real-time
3. **Error Highlighting:** Red for errors, yellow for warnings
4. **Professional Look:** Makes tool feel polished

**Example Output:**
```
[cyan]Generating QA pairs from 150 chunks...[/cyan]
Processing ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:02:30
[green]✓ Generated 375 pairs (avg: 2.5 per chunk)[/green]

[cyan]Curating with LLM-as-Judge (threshold: 7.0)...[/cyan]
Rating ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:01:45
[green]✓ Kept 287 pairs (76.5% pass rate)[/green]
```

### 8.7 Why Support Multiple Export Formats?

**Decision:** Implement ChatML, Alpaca, ShareGPT, and JSONL formats

**Rationale:**

1. **Framework Compatibility:**
   - ChatML: Used by Hugging Face, Axolotl
   - Alpaca: Used by Stanford Alpaca, llama.cpp
   - ShareGPT: Used by FastChat, Vicuna
   - JSONL: Universal, used by OpenAI fine-tuning

2. **Future-Proofing:** New tools may prefer different formats

3. **Easy Implementation:** All formats are just JSON restructuring

**Format Examples:**

```python
# ChatML
{
  "messages": [
    {"role": "user", "content": "What is HDF5?"},
    {"role": "assistant", "content": "HDF5 is..."}
  ]
}

# Alpaca
{
  "instruction": "What is HDF5?",
  "input": "",
  "output": "HDF5 is..."
}

# ShareGPT
{
  "conversations": [
    {"from": "human", "value": "What is HDF5?"},
    {"from": "gpt", "value": "HDF5 is..."}
  ]
}

# JSONL (newline-delimited JSON)
{"question": "What is HDF5?", "answer": "HDF5 is..."}
{"question": "How to open HDF5?", "answer": "Use h5py..."}
```

---

## Conclusion

The Generator is designed as a research-driven, production-ready system for generating high-quality synthetic training data. Each design decision is grounded in published research, with implementations that balance scientific rigor with practical usability.

**Key Achievements:**

1. **Research-Based:** Every major component traces to peer-reviewed papers
2. **Quality-Focused:** LIMA principles throughout (quality > quantity)
3. **Flexible:** Modular design allows experimentation
4. **Production-Ready:** Error handling, checkpoints, multi-provider support
5. **Well-Documented:** Comprehensive documentation and examples


---

## References

1. Zhou et al. "LIMA: Less Is More for Alignment" NeurIPS 2023
2. Li et al. "Self-Alignment with Instruction Backtranslation" ICLR 2024
3. Hsieh et al. "Distilling Step-by-Step" Google Research 2023
4. Chen et al. "AlpaGasus" ICLR 2024
5. Schick et al. "Toolformer" NeurIPS 2023
6. Patil et al. "Gorilla" NeurIPS 2024
7. Qin et al. "ToolLLM" ICLR 2024

---

**Document Version:** 1.0  
**Last Updated:** January 8, 2026  
**Maintained By:** Shazzadul
