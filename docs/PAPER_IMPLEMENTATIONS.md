# Research Papers Implementation Reference

**Quick reference for which papers were implemented and where to find them in the codebase.**

---

## Paper → Implementation Mapping

### 1. LIMA: Less Is More for Alignment
**Citation:** Zhou et al., Meta AI, NeurIPS 2023  
**ArXiv:** https://arxiv.org/abs/2305.11206

**Key Contribution:** Quality > Quantity - 1K curated examples rival 100K+ low-quality examples

**Implementation:**
- **File:** `src/generator/qa/curate.py`
- **Feature:** LLM-as-Judge with strict quality thresholds
- **Config:** `configs/prompts/qa_rating.yaml`
- **CLI:** `uv run generator curate <input> -o <output> --threshold 7.0`

**What Was Extracted:**
- Multi-dimensional rating criteria (Clarity, Accuracy, Usefulness, Difficulty)
- Strict threshold-based filtering (default ≥7/10)
- Reasoning field to explain rating decisions
- Quality-focused curation over quantity

**Code Example:**
```python
# curate.py - LIMA principles
DEFAULT_THRESHOLD = 7.0  # Only keep high-quality examples

rating_criteria = {
    "clarity": (0, 3),      # Question clear? Answer understandable?
    "accuracy": (0, 3),     # Answer correct and supported?
    "usefulness": (0, 2),   # Valuable for training?
    "difficulty": (0, 2)    # Appropriate complexity?
}
# Total rating = sum (max 10)
```

---

### 2. Self-Alignment with Instruction Backtranslation
**Citation:** Li et al., Meta AI, ICLR 2024  
**ArXiv:** https://arxiv.org/abs/2308.06259

**Key Contribution:** Treat documents as "answers," generate "questions" for them

**Implementation:**
- **File:** `src/generator/qa/qa_generator.py`
- **Feature:** Primary QA generation methodology
- **Config:** `configs/prompts/qa_generation.yaml`
- **CLI:** `uv run generator generate <lancedb> -o <output> --target-pairs N`

**What Was Extracted:**
- Backward model concept: Generate questions from answer-like text (document chunks)
- Self-augmentation: Multiple questions per document chunk
- Quality over fabrication: Answers grounded in real documents
- Target-based generation: Specify total pairs, auto-calculate per-chunk

**Code Example:**
```python
# qa_generator.py - Instruction Backtranslation
def generate_qa_from_lancedb(db_path, target_pairs, ...):
    chunks = load_chunks(db_path)  # Documents as "answers"
    
    # Calculate how many questions per chunk
    n_pairs_per_chunk = target_pairs // len(chunks)
    
    for chunk in chunks:
        # Treat chunk as answer, generate questions
        questions = llm.generate(
            f"Generate {n_pairs_per_chunk} questions that this text would answer:\n{chunk.text}"
        )
```

**Prompt Template:**
```yaml
qa_generation: |
  Given this document chunk, generate {n_pairs} question-answer pairs.
  
  Requirements:
  - Questions should be answerable from the text
  - Answers come directly from the text
  - Cover different aspects and difficulty levels
  
  Document chunk: {chunk_text}
```

---

### 3. Distilling Step-by-Step (Chain-of-Thought Reasoning)
**Citation:** Hsieh et al., Google Research, 2023  
**ArXiv:** https://arxiv.org/abs/2305.02301

**Key Contribution:** Extract Chain-of-Thought (CoT) rationales to enable small models to match or exceed large model reasoning capabilities

**Core Insight from Paper:**
The paper demonstrated that training small models with intermediate reasoning steps (CoT) rather than just input-output pairs leads to:
- **12.5× less training data** needed for equivalent performance
- **770M parameter model outperforming 540B parameter model** on reasoning tasks
- **Better generalization** to out-of-distribution examples
- **Improved interpretability** through explicit reasoning traces

**Why Chain-of-Thought Training Matters:**

Traditional fine-tuning teaches models to map inputs directly to outputs (Q→A). This works for simple tasks but fails on complex reasoning. CoT training teaches the *process* of reasoning:
- Q → Reasoning Steps → A

This enables smaller models to:
1. Break down complex problems systematically
2. Show their work (interpretable decisions)
3. Generalize reasoning patterns to new domains
4. Match or exceed larger models on reasoning tasks

**Implementation in Generator:**

We implement BOTH methods from the paper:

#### **Method 1: Direct CoT Generation** (`cot_generator.py`)

Generates QA pairs with reasoning from scratch, directly from source documents.

- **File:** `src/generator/cot/cot_generator.py`
- **Config:** `configs/prompts/cot_generation.yaml`
- **CLI:** `uv run generator generate-cot <lancedb> -o <output>`
- **When to Use:** Starting fresh, want reasoning-focused questions

#### **Method 2: CoT Enhancement** (`cot_enhancer.py`)

Adds reasoning steps to existing high-quality QA pairs.

- **File:** `src/generator/cot/cot_enhancer.py`
- **Config:** `configs/prompts/cot_enhancement.yaml`
- **CLI:** `uv run generator enhance-cot <qa_file> -o <output>`
- **When to Use:** Already have good QA, want to add reasoning efficiently

**What Was Extracted from Paper:**
- **CoT reasoning structure:** Step-by-step breakdown showing intermediate thinking
- **Reasoning depth:** 2-5 logical steps that build progressively
- **Rationale fields:** `reasoning` array containing ordered reasoning steps
- **Training format:** Convert reasoning into few-shot examples for fine-tuning
- **Quality criteria:** Reasoning must be logical, non-circular, and lead to answer

**Full Code Example - Direct Generation:**
```python
# cot_generator.py - Implements Distilling Step-by-Step methodology
def generate_cot_pairs(lancedb_path, output_path, llm_config, n_pairs=None, ...):
    """
    Generate CoT pairs from LanceDB chunks.
    
    Based on \"Distilling Step-by-Step\" paper:
    - Extracts reasoning steps from large model
    - Structures for small model training
    - Validates reasoning quality
    """
    # Load LanceDB chunks (same source as QA generation)
    db = lancedb.connect(lancedb_path)
    table = db.open_table(table_name)
    chunks = table.to_pandas()
    
    # Load CoT generation prompt
    prompts = load_prompts(config_dir)
    cot_prompt_template = prompts[\"cot_generation\"]
    
    for chunk in chunks:
        # Generate with explicit reasoning structure request
        prompt = cot_prompt_template.format(
            chunk_text=chunk[\"text\"],
            n_pairs=n_pairs or 3
        )
        
        response = llm.generate(prompt, temperature=0.7)
        cot_pairs = json5.loads(response)
        
        # Structure with reasoning fields
        for pair in cot_pairs:\n            result = {
                \"question\": pair[\"question\"],
                \"reasoning\": pair[\"reasoning\"],  # List of 2-5 reasoning steps
                \"answer\": pair[\"answer\"],
                \"chunk_id\": chunk[\"id\"],
                \"cot_generated\": True
            }
            
            # Validate reasoning quality
            if validate_cot_reasoning(result):
                output.append(result)

**Code Example - Enhancement:**
```python
# cot_enhancer.py - Add CoT to existing QA pairs
def enhance_with_cot(input_path, output_path, llm_config, batch_size=5):
    \"\"\"Add reasoning to existing QA pairs.\"\"\"
    qa_pairs = load_json(input_path)
    
    # Load enhancement prompt
    prompts = load_prompts(config_dir)
    enhance_prompt = prompts[\"cot_enhancement\"]
    
    for qa in qa_pairs:
        # Skip if already has reasoning
        if \"reasoning\" in qa:
            continue
        
        # Generate reasoning for existing Q&A
        prompt = enhance_prompt.format(
            question=qa[\"question\"],
            answer=qa[\"answer\"]
        )
        
        response = llm.generate(prompt, temperature=0.5)
        reasoning_data = json5.loads(response)
        
        # Add reasoning while preserving original Q&A
        qa[\"reasoning\"] = reasoning_data[\"reasoning\"]
        qa[\"cot_enhanced\"] = True
```

**Output Format:**
```json
{
  "question": "Why is chunking important for HDF5 parallel I/O?",
  "reasoning": [
    "HDF5 stores data in fixed-size chunks",
    "Multiple processes need simultaneous access",
    "Proper chunking prevents write conflicts",
    "Independent chunks enable true parallelism"
  ],
  "answer": "Chunking is important because it enables independent parallel access..."
}
```

---

### 4. AlpaGasus
**Citation:** Chen et al., UMD, ICLR 2024  
**ArXiv:** https://arxiv.org/abs/2307.08701

**Key Contribution:** LLM-as-Judge with detailed criteria beats automated filtering

**Implementation:**
- **File:** `src/generator/qa/curate.py` (integrated with LIMA)
- **Feature:** Detailed per-criteria scoring and reasoning
- **Config:** `configs/prompts/qa_rating.yaml`

**What Was Extracted:**
- Multi-dimensional evaluation framework
- Per-criteria breakdown scores
- Reasoning explanations for ratings
- Format detection and preservation (QA vs CoT)

**Code Example:**
```python
# curate.py - AlpaGasus methodology
def rate_pairs(pairs, llm, threshold):
    for batch in batched(pairs):
        ratings = llm.generate(rating_prompt)
        
        for pair, rating in zip(batch, ratings):
            pair["rating"] = rating["total"]
            pair["clarity"] = rating["clarity"]
            pair["accuracy"] = rating["accuracy"]
            pair["usefulness"] = rating["usefulness"]
            pair["difficulty"] = rating["difficulty"]
            pair["reasoning"] = rating["reasoning"]  # Why this rating
            
            if pair["rating"] >= threshold:
                filtered_pairs.append(pair)
```

**Rating Prompt:**
```yaml
qa_rating: |
  Rate on 1-10 scale using these criteria:
  
  - Clarity (0-3): Question clear? Answer understandable?
  - Accuracy (0-3): Answer correct and supported?
  - Usefulness (0-2): Valuable for training?
  - Difficulty (0-2): Appropriate complexity?
  
  Total = sum of criteria (max 10)
  
  For each pair provide:
  - Individual scores
  - Total rating
  - Brief reasoning
```

---

### 5. Toolformer
**Citation:** Schick et al., Meta AI, NeurIPS 2023  
**ArXiv:** https://arxiv.org/abs/2302.04761

**Key Contribution:** Self-supervised tool learning with simple single-step calls

**Implementation:**
- **File:** `src/generator/tool/tool_generator.py`
- **Feature:** Single-step mode for simple tool calls
- **CLI:** `uv run generator tool-generate <tools> -o <output> --mode single`

**What Was Extracted:**
- Single-step tool calling pattern
- Self-supervised filtering concept
- Simple instruction → single tool → result workflow

**Code Example:**
```python
# tool_generator.py - Toolformer approach
def generate_single_step_example(tool, llm):
    # Generate realistic instruction
    instruction = llm.generate(f"Create user request requiring {tool.name}")
    
    # Generate tool call
    example = {
        "instruction": instruction,
        "solution": {
            "reasoning": f"To {instruction}, I need to call {tool.name}",
            "tool_calls": [{
                "tool": tool.name,
                "arguments": generate_realistic_args(tool)
            }]
        }
    }
```

**Example Output:**
```json
{
  "instruction": "Open the file 'data.h5' for reading",
  "solution": {
    "reasoning": "To open the file, I need to call open_file",
    "tool_calls": [
      {
        "tool": "open_file",
        "arguments": {"filename": "data.h5", "mode": "r"}
      }
    ]
  }
}
```

---

### 6. Gorilla
**Citation:** Patil et al., UC Berkeley, NeurIPS 2024  
**ArXiv:** https://arxiv.org/abs/2305.15334

**Key Contribution:** Always include API documentation to reduce hallucination <7%

**Implementation:**
- **File:** `src/generator/tool/tool_generator.py`
- **Feature:** API documentation grounding in every example
- **Field:** `api_documentation` in all tool examples

**What Was Extracted:**
- Documentation grounding prevents API hallucination
- Retrieval-aware training concept
- Every example includes full API signature and description

**Code Example:**
```python
# tool_generator.py - Gorilla's documentation grounding
def format_api_docs(tool):
    """Include API documentation in every example."""
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

# Every tool example includes this
example = {
    "instruction": "...",
    "solution": {...},
    "api_documentation": format_api_docs(tool)  # ← Gorilla
}
```

**Example Output:**
```json
{
  "instruction": "Read dataset /data/temperature from file.h5",
  "solution": {...},
  "api_documentation": "read_dataset(file: File, path: str) -> ndarray\n\nRead a dataset from an HDF5 file.\n\nParameters:\n- file: Open HDF5 file object\n- path: Path to dataset within file\n\nReturns: NumPy array with dataset contents\n\nExample: data = read_dataset(f, '/data/temperature')"
}
```

---

### 7. ToolLLM
**Citation:** Qin et al., Tsinghua University, ICLR 2024  
**ArXiv:** https://arxiv.org/abs/2307.16789

**Key Contribution:** DFSDT (Depth-First Search Decision Tree) for multi-step reasoning

**Implementation:**
- **File:** `src/generator/tool/tool_generator.py`
- **Feature:** Multi-step mode with reasoning chains
- **CLI:** `uv run generator tool-generate <tools> -o <output> --mode multi`

**What Was Extracted:**
- Multi-step reasoning path structure
- Step-by-step thought → action → result pattern
- Tool chaining with intermediate results
- Complex workflow support (2-4 tools in sequence)

**Code Example:**
```python
# tool_generator.py - ToolLLM DFSDT approach
def generate_multi_step_example(tools, llm):
    tool_chain = select_tool_chain(tools, min_tools=2, max_tools=4)
    
    reasoning_steps = []
    for i, tool in enumerate(tool_chain, 1):
        step = {
            "step": i,
            "thought": f"Step {i}: I need to {action}...",
            "tool": tool.name,
            "arguments": generate_realistic_args(tool),
            "expected_result": "..."
        }
        reasoning_steps.append(step)
    
    return {
        "instruction": complex_instruction,
        "solution": {
            "reasoning_path": reasoning_steps,
            "final_answer": "..."
        }
    }
```

**Example Output:**
```json
{
  "instruction": "Calculate mean temperature from climate_sim.h5 for 2023",
  "solution": {
    "reasoning_path": [
      {
        "step": 1,
        "thought": "First, open the HDF5 file",
        "tool": "open_file",
        "arguments": {"filename": "climate_sim.h5", "mode": "r"}
      },
      {
        "step": 2,
        "thought": "Next, read the temperature dataset",
        "tool": "read_dataset",
        "arguments": {"file": "<from step 1>", "path": "/2023/temperature"}
      },
      {
        "step": 3,
        "thought": "Finally, calculate the mean",
        "tool": "calculate_mean",
        "arguments": {"data": "<from step 2>"}
      }
    ],
    "final_answer": "Mean temperature calculated via 3-step process..."
  }
}
```

---

## Additional Papers (Methodology Only)

### APIGen (Salesforce, 2024)
**What Was Used:** Triple verification pipeline (Format → Execution → Semantic)

**Implementation:** `src/generator/tool/tool_executor.py`
```python
def verify_tool_example(example):
    # 1. Format verification
    validate_schema(example)
    
    # 2. Execution verification (simulated or real)
    execute_tools(example)
    
    # 3. Semantic verification (does it solve the task?)
    llm_verify(example)
```

### ToolACE (ICLR 2025)
**What Was Used:** Dual-layer verification concept

**Implementation:** Integrated into `tool_executor.py` verification stages

---

## Summary Table

| Paper | Primary File | Key Extraction | CLI Command |
|-------|--------------|----------------|-------------|
| LIMA | `curate.py` | Quality thresholds | `curate --threshold 7.0` |
| Instruction Backtranslation | `qa_generator.py` | Questions from docs | `generate --target-pairs N` |
| Distilling Step-by-Step | `cot_generator.py`, `cot_enhancer.py` | CoT reasoning | `generate-cot`, `enhance-cot` |
| AlpaGasus | `curate.py` | Detailed criteria | (integrated in curate) |
| Toolformer | `tool_generator.py` | Single-step tools | `tool-generate --mode single` |
| Gorilla | `tool_generator.py` | API documentation | (auto-included) |
| ToolLLM | `tool_generator.py` | Multi-step chains | `tool-generate --mode multi` |
| **DEITA (2024)** | `multi_scorer.py` | 3D quality scoring | `multi-score --top-k N` |
| **TOUCAN (2024)** | `coverage_selector.py` | Semantic deduplication | `select-coverage` |
| **ToolMind (2025)** | `tool_curator.py` | Turn-level filtering | `tool-curate --turn-level-filter` |
| **ToolGrad (2025)** | `tool_generator.py` | Chain-first generation | `tool-generate-chain` |
| **In-N-Out (2025)** | `dependency_graph.py` | Parameter dependency graphs | (library API) |
| **MCP-AgentBench (2025)** | `outcome_evaluator.py` | Outcome-oriented evaluation | (library API) |

---

## How to Read the Implementation

### For Each Paper:

1. **Find the file:** Check "Primary File" in table above
2. **Read the docstring:** Top of file explains methodology
3. **Check the prompt:** In `configs/prompts/` directory
4. **Try the CLI:** Use example commands
5. **Read tests:** In `tests/test_<module>.py`

### Example Workflow:

```bash
# 1. Read the implementation
cat src/generator/qa/qa_generator.py | head -30

# 2. Check the prompt template
cat configs/prompts/qa_generation.yaml

# 3. Try it out
uv run generator generate ./lancedb -o test.json --target-pairs 50

# 4. Inspect output
jq '.[0]' test.json

# 5. Understand rating
uv run generator curate test.json -o rated.json --threshold 7.0
jq '.[0] | {q: .question, rating: .rating, reasoning: .reasoning}' rated.json
```

---

## Design Philosophy Extracted from Papers

### From LIMA:
- ✅ **Quality > Quantity** - Strict filtering over bulk generation
- ✅ **Detailed criteria** - Multi-dimensional evaluation
- ✅ **Human-like judgment** - LLM-as-Judge approach

### From Instruction Backtranslation:
- ✅ **Ground truth preservation** - Documents as answers
- ✅ **Question generation** - Easier than answer generation
- ✅ **Domain knowledge retention** - No hallucination

### From Distilling Step-by-Step:
- ✅ **Reasoning chains** - Break down complex problems
- ✅ **Small model capability** - Enable 7B to match larger models
- ✅ **Progressive learning** - Step-by-step explanations

### From Tool Papers (Toolformer, Gorilla, ToolLLM):
- ✅ **Documentation grounding** - Prevent API hallucination
- ✅ **Progressive complexity** - Single → Multi-step
- ✅ **Verification layers** - Format → Execute → Semantic

---

## New Paper Implementations (Jan 2026)

### 8. DEITA (2024)
**Citation:** Liu et al., 2024  
**ArXiv:** https://arxiv.org/abs/2312.15685

**Key Contribution:** Multi-dimensional scoring achieves 10x data efficiency - 6K examples match 100K randomly selected

**Implementation:**
- **File:** `src/generator/qa/multi_scorer.py`
- **Feature:** 3D scoring (complexity, quality, diversity)
- **CLI:** `uv run generator multi-score <input> -o <output> --top-k N`

**What Was Extracted:**
- Complexity scoring: Reasoning depth, multi-step thinking required
- Quality scoring: Clarity, accuracy, formatting, usefulness
- Diversity scoring: Semantic uniqueness via embeddings
- Weighted combination for optimal selection

**Code Example:**
```python
from generator import MultiDimensionalScorer

scorer = MultiDimensionalScorer()
scored = scorer.score_pair(qa_pair, existing_pairs)
# Returns: MultiScore(complexity=7.5, quality=8.2, diversity=6.0, combined=7.3)

# Select top 500 most valuable examples
selected = scorer.select_top_k(all_pairs, k=500)
```

---

### 9. TOUCAN (Oct 2024)
**Citation:** arXiv:2510.01179

**Key Contribution:** Coverage-based selection reduces dataset by 40-60% with minimal information loss

**Implementation:**
- **File:** `src/generator/tool/coverage_selector.py`
- **Feature:** Semantic clustering + representative selection
- **CLI:** `uv run generator select-coverage <input> -o <output>`

**What Was Extracted:**
- Sentence transformer embeddings for semantic similarity
- K-means clustering to group similar examples
- Two strategies: centroid (closest to center) or diverse (maximize spread)
- Coverage metrics to measure information retention

**Code Example:**
```python
from generator import CoverageSelector

selector = CoverageSelector(embedding_model="all-MiniLM-L6-v2")
selected = selector.select(pairs, target_count=500, strategy="diverse")
# Returns 500 diverse examples covering all semantic clusters
```

---

### 10. ToolMind (Nov 2025)
**Citation:** arXiv:2511.15718

**Key Contribution:** Turn-level filtering removes poor-quality steps even in otherwise good examples

**Implementation:**
- **File:** `src/generator/tool/tool_curator.py`
- **Feature:** Per-step quality rating and filtering
- **Method:** `filter_by_turn_quality()`

**What Was Extracted:**
- Per-turn quality assessment (not just overall example quality)
- Minimum step quality threshold (default 0.7)
- Remove entire example if any step falls below threshold
- Detailed step-level feedback for debugging

**Code Example:**
```python
curator = ToolCurator(client)
filtered = curator.curate(examples, turn_level_filter=True, min_step_quality=0.7)
```

---

### 11. ToolGrad (Aug 2025)
**Citation:** arXiv:2508.04086

**Key Contribution:** Chain-first generation creates more coherent multi-tool examples

**Implementation:**
- **File:** `src/generator/tool/tool_generator.py`
- **Feature:** Generate tool chain first, then synthesize query
- **CLI:** `uv run generator tool-generate-chain <tools> -o <output>`

**What Was Extracted:**
- Reverse generation order: Tools → Query (not Query → Tools)
- More logical tool chains with proper data flow
- Query synthesis that naturally requires the generated chain
- Hybrid mode combining both approaches

**Code Example:**
```python
generator = ToolGenerator(client, tools)
examples = generator.generate_chain_first(n_examples=100, min_steps=2, max_steps=5)
# Or hybrid mode (recommended)
examples = generator.generate_examples_hybrid(n_examples=100)
```

---

### 12. In-N-Out (2025)
**Citation:** Parameter dependency graphs for tool orchestration

**Key Contribution:** Model parameter dependencies as graphs for better tool chaining

**Implementation:**
- **File:** `src/generator/tool/dependency_graph.py`
- **Feature:** Build and query parameter dependency graphs
- **Class:** `DependencyGraph`

**What Was Extracted:**
- Parameter-level dependency tracking between tools
- Automatic dependency inference from tool definitions
- Topological ordering for execution planning
- Visualization support (DOT format)

**Code Example:**
```python
from generator import DependencyGraph

graph = DependencyGraph()
graph.add_tools(tools)
order = graph.get_execution_order(["read_dataset", "calculate_stats"])
# Returns: ["open_file", "read_dataset", "calculate_stats"]
```

---

### 13. MCP-AgentBench (2025)
**Citation:** Outcome-oriented evaluation for agentic tasks

**Key Contribution:** Evaluate by task completion, not just individual tool calls

**Implementation:**
- **File:** `src/generator/tool/outcome_evaluator.py`
- **Feature:** Outcome-based success metrics
- **Class:** `OutcomeEvaluator`

**What Was Extracted:**
- Task-completion evaluation (did the example achieve its goal?)
- Constraint checking (were all requirements met?)
- Partial credit scoring for multi-step tasks
- Detailed evaluation breakdown

**Code Example:**
```python
from generator import OutcomeEvaluator

evaluator = OutcomeEvaluator(client)
result = evaluator.evaluate(example, expected_outcomes=["file opened", "data loaded"])
# Returns: OutcomeResult(success=True, score=0.95, constraint_violations=[])
```

---

## Future Paper Implementations (Planned)

### Self-Instruct (Stanford)
- Iterative refinement of training data
- Bootstrap from small seed set

### FLAN (Google)
- Multi-task instruction tuning
- Format mixing

### Constitutional AI (Anthropic)
- Self-critique and improvement
- Harmlessness and helpfulness

---

**Document Version:** 2.0  
**Last Updated:** January 9, 2026  
**Maintained By:** Shazzadul
