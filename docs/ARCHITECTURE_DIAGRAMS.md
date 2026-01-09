# Generator Architecture Diagrams

**Author:** Shazzadul  
Visual representations of the Generator system design.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GENERATOR SYSTEM                            │
│                                                                     │
│  ┌──────────────────────┐          ┌──────────────────────┐       │
│  │   QA Pipeline        │          │  Tool-Use Pipeline   │       │
│  │   (Knowledge)        │          │  (Agentic Skills)    │       │
│  └──────────────────────┘          └──────────────────────┘       │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐         │
│  │          Multi-Provider LLM Support                   │         │
│  │  Ollama │ Claude │ Gemini │ vLLM │ OpenAI │ Anthropic │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐         │
│  │              Quality Assurance Layer                  │         │
│  │  LLM-as-Judge │ Triple Verification │ Format Detection │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐         │
│  │          Advanced Selection Layer (Jan 2026) ⭐ NEW   │         │
│  │  Multi-Score (DEITA) │ Coverage Select (TOUCAN) │     │         │
│  │  Turn-Level Filter (ToolMind) │ Chain-First (ToolGrad)│         │
│  │  Dependency Graphs (In-N-Out) │ Outcome Eval (MCP)    │         │
│  └──────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## QA Pipeline (Instruction Backtranslation)

```
┌────────────────────────────────────────────────────────────────────┐
│                     Input: LanceDB with Document Chunks             │
└────────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  GENERATE     │  Instruction Backtranslation (Meta AI, 2024)
         │  QA Pairs     │  • Treat chunks as "answers"
         │               │  • Generate questions
         └───────┬───────┘  • Target-based or per-chunk
                 │
                 │  Output: qa_raw.json
                 │  [{"question": "...", "answer": "...", "source_chunk_id": "..."}]
                 │
                 ▼
         ┌───────────────┐
         │  ENRICH       │  Response Rewriting
         │  Answers      │  • Improve formatting
         │               │  • Maintain all information
         └───────┬───────┘  • Better structure
                 │
                 │  Output: qa_enriched.json
                 │  [{"question": "...", "answer": "...", "original_answer": "..."}]
                 │
                 ▼
         ┌───────────────┐
         │  CURATE       │  LIMA (Meta, 2023) + AlpaGasus (UMD, 2024)
         │  by Quality   │  • LLM-as-Judge
         │               │  • Detailed criteria (Clarity, Accuracy, etc.)
         └───────┬───────┘  • Threshold filtering (≥7/10)
                 │
                 │  Output: qa_curated.json
                 │  [{"question": "...", "answer": "...", "rating": 8, "clarity": 3, ...}]
                 │
                 ▼
         ┌───────────────┐
         │  ENHANCE COT  │  Distilling Step-by-Step (Google, 2023)
         │  (Optional)   │  • Add reasoning steps
         │               │  • Enable small model reasoning
         └───────┬───────┘  • 2-5 steps typically
                 │
                 │  Output: qa_cot.json
                 │  [{"question": "...", "reasoning": ["Step 1...", "Step 2..."], "answer": "..."}]
                 │
                 ▼
         ┌───────────────┐
         │  EXPORT       │  Format Conversion
         │  to Format    │  • ChatML, Alpaca, ShareGPT, JSONL
         │               │  • Framework-specific
         └───────┬───────┘  • Ready for training
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                Output: training.jsonl (ChatML/Alpaca/etc.)          │
│                Ready for fine-tuning                                │
└────────────────────────────────────────────────────────────────────┘
```

### Quality Gate (Curation)

```
┌─────────────────┐
│   300 QA Pairs  │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────┐
│      LLM-as-Judge Rating         │
│                                  │
│  For each pair:                  │
│  ├─ Clarity (0-3)                │
│  ├─ Accuracy (0-3)               │
│  ├─ Usefulness (0-2)             │
│  ├─ Difficulty (0-2)             │
│  └─ Total = Sum (max 10)         │
│                                  │
│  Plus: Reasoning explanation     │
└────────┬─────────────────────────┘
         │
         ▼
    ┌────────────┐
    │ Threshold  │
    │  ≥ 7.0     │
    └────┬───────┘
         │
         ├─────────────┐
         │             │
    Rating ≥ 7    Rating < 7
         │             │
         ▼             ▼
    ┌────────┐   ┌──────────┐
    │  KEEP  │   │  FILTER  │
    │ (226)  │   │   (74)   │
    └────────┘   └──────────┘
         │
         ▼
   75% Pass Rate
```

---

## Advanced Selection Pipeline (Jan 2026) ⭐ NEW

### Multi-Dimensional Scoring (DEITA)

```
┌────────────────────────────────────────────────────────────────────┐
│                 Input: Curated QA Pairs                             │
└────────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────────────┐
    │          MULTI-DIMENSIONAL SCORING          │
    │               (DEITA, 2024)                 │
    │                                            │
    │  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐
    │  │ COMPLEXITY  │ │  QUALITY    │ │  DIVERSITY   │
    │  │   (0-10)    │ │   (0-10)    │ │    (0-10)    │
    │  │             │ │             │ │              │
    │  │ • Reasoning │ │ • Clarity   │ │ • Semantic   │
    │  │   depth     │ │ • Accuracy  │ │   uniqueness │
    │  │ • Multi-step│ │ • Format    │ │ • Embedding  │
    │  │ • Domain    │ │ • Useful    │ │   distance   │
    │  │   knowledge │ │             │ │              │
    │  └──────┬──────┘ └──────┬──────┘ └──────┬───────┘
    │         │               │               │
    │         └───────────────┼───────────────┘
    │                         │
    │                         ▼
    │               ┌──────────────────┐
    │               │  COMBINED SCORE   │
    │               │ = 0.4×C + 0.4×Q   │
    │               │   + 0.2×D         │
    │               └────────┬─────────┘
    │                        │
    └────────────────────────┼───────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │    SELECTION STRATEGIES      │
              │                              │
              │  threshold: score ≥ min      │
              │  top-k: select best K        │
              │  combined: both              │
              └──────────────┬───────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────────┐
│              Output: Optimally Selected Examples                    │
│              10x data efficiency (6K = 100K random)                 │
└────────────────────────────────────────────────────────────────────┘
```

### Coverage-Based Selection (TOUCAN)

```
┌────────────────────────────────────────────────────────────────────┐
│                 Input: 2000 Curated QA Pairs                        │
└────────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  EMBED        │  Sentence Transformers
         │  Questions    │  • all-MiniLM-L6-v2 (default)
         │               │  • 384-dim vectors
         └───────┬───────┘  
                 │
                 ▼
         ┌───────────────┐
         │  CLUSTER      │  K-Means Clustering
         │  Semantically │  • Group similar questions
         │               │  • auto k = target_count
         └───────┬───────┘  • Each cluster = topic
                 │
                 │
        ┌────────┴────────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐   ┌──────────────────┐
│  CENTROID        │   │  DIVERSE         │
│  Strategy        │   │  Strategy        │
│                  │   │                  │
│  Select closest  │   │  Maximize spread │
│  to cluster      │   │  within cluster  │
│  center          │   │                  │
└────────┬─────────┘   └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌───────────────────┐
         │  SELECT           │  Per-cluster selection
         │  Representatives  │  • 1+ examples per cluster
         │                   │  • Maximize coverage
         └───────────┬───────┘  
                     │
                     ▼
┌────────────────────────────────────────────────────────────────────┐
│              Output: 500 Diverse Examples                           │
│              40-60% reduction, minimal information loss             │
│              Coverage score: 0.55-0.70                              │
└────────────────────────────────────────────────────────────────────┘
```

### Combined Selection Workflow (Recommended)

```
┌──────────────────────────────────────────────────────────────────────┐
│              OPTIMAL SELECTION PIPELINE                              │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              2000 Generated QA Pairs                         │  │
│   └───────────────────────────┬─────────────────────────────────┘  │
│                               │                                     │
│   Step 1: Quality Curation    ▼                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  curate.py --threshold 7.0                                   │  │
│   │  → 1500 pairs (75% pass rate)                                │  │
│   └───────────────────────────┬─────────────────────────────────┘  │
│                               │                                     │
│   Step 2: Multi-Score         ▼                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  multi-score --min-score 6.0                                 │  │
│   │  Score by complexity + quality + diversity                   │  │
│   │  → 1200 pairs (filter low-value)                            │  │
│   └───────────────────────────┬─────────────────────────────────┘  │
│                               │                                     │
│   Step 3: Coverage Select     ▼                                    │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │  select-coverage --target-count 500 --strategy diverse       │  │
│   │  Semantic deduplication                                      │  │
│   │  → 500 diverse, high-quality pairs                          │  │
│   └───────────────────────────┬─────────────────────────────────┘  │
│                               │                                     │
│                               ▼                                     │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              500 OPTIMAL TRAINING EXAMPLES                   │  │
│   │              • High quality (LLM-as-Judge ≥7)               │  │
│   │              • High value (multi-score ≥6)                  │  │
│   │              • High diversity (coverage selection)           │  │
│   │              • 10x efficiency vs random selection            │  │
│   └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Chain-of-Thought (CoT) Pipeline

### Method 1: Direct CoT Generation

```
┌────────────────────────────────────────────────────────────────────┐
│                     Input: LanceDB with Document Chunks             │
└────────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  GENERATE     │  Distilling Step-by-Step (Google, 2023)
         │  CoT Pairs    │  
         │               │  Prompt: "Generate Q&A with reasoning"
         └───────┬───────┘  • Question requiring multi-step thinking
                 │           • 2-5 reasoning steps
                 │           • Final answer
                 │
                 │  Output: cot_raw.json
                 │  [{"question": "...", "reasoning": ["Step 1...", "Step 2...", ...], "answer": "..."}]
                 │
                 ▼
         ┌───────────────┐
         │  VALIDATE     │  CoT Quality Checks
         │  Reasoning    │  • Minimum 2 steps
         │               │  • Non-circular logic
         └───────┬───────┘  • Reasoning → answer coherence
                 │
                 │  Output: cot_validated.json
                 │  [valid pairs only]
                 │
                 ▼
         ┌───────────────┐
         │  CURATE       │  LLM-as-Judge for CoT
         │  by Quality   │  • Rate reasoning quality
         │               │  • Check logical progression
         └───────┬───────┘  • Filter by threshold
                 │
                 │  Output: cot_curated.json
                 │  [{"question": "...", "reasoning": [...], "answer": "...", "rating": 8}]
                 │
                 ▼
         ┌───────────────┐
         │  EXPORT       │  Format Conversion
         │  to Format    │  • ChatML with reasoning
         │               │  • Include reasoning in assistant response
         └───────┬───────┘  • Framework-specific
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                Output: cot_training.jsonl                           │
│                Small models learn to reason like large models       │
└────────────────────────────────────────────────────────────────────┘
```

### Method 2: CoT Enhancement (Adding Reasoning to Existing QA)

```
┌────────────────────────────────────────────────────────────────────┐
│                Input: Curated QA Pairs (qa_curated.json)            │
│                [{"question": "...", "answer": "..."}]               │
└────────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  DETECT       │  Format Check
         │  Existing     │  • Skip if already has "reasoning" field
         │  Reasoning    │  • Process only simple QA
         └───────┬───────┘  • Preserve structure
                 │
                 │  Pairs without reasoning
                 │
                 ▼
         ┌───────────────┐
         │  ENHANCE      │  Add Reasoning Steps
         │  with CoT     │  
         │               │  For each QA:
         └───────┬───────┘  "Given Q and A, add 2-5 reasoning steps
                 │            that logically connect them"
                 │
                 │  LLM generates reasoning
                 │
                 ▼
         ┌───────────────┐
         │  VALIDATE     │  Reasoning Quality Check
         │  Enhancement  │  • Has 2+ steps?
         │               │  • Steps are distinct?
         └───────┬───────┘  • Coherent with answer?
                 │
                 │  Valid   │  Invalid
                 ├──────────┼──────────┐
                 │          │          │
                 ▼          ▼          ▼
         ┌───────────┐ ┌────────┐ ┌──────────┐
         │ Enhanced  │ │ Retry  │ │  Skip    │
         │   QA      │ │ Once   │ │  Keep    │
         │           │ │        │ │ Original │
         └─────┬─────┘ └────────┘ └────┬─────┘
               │                       │
               └───────────┬───────────┘
                           │
                           ▼
                  ┌────────────────┐
                  │  MERGE         │  Combine Results
                  │  Results       │  • Enhanced pairs (with reasoning)
                  │                │  • Original pairs (unchanged)
                  └────────┬───────┘  • Preserved pairs (already had reasoning)
                           │
                           │  Output: qa_with_cot.json
                           │  [{"question": "...", "reasoning": [...], "answer": "...", "cot_enhanced": true}]
                           │
                           ▼
                  ┌────────────────┐
                  │  EXPORT        │  Format Conversion
                  │  to Format     │  • ChatML with reasoning in messages
                  │                │  • Reasoning integrated into response
                  └────────┬───────┘  • Training-ready
                           │
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│             Output: enhanced_training.jsonl                         │
│             High-quality QA + reasoning capabilities                │
└────────────────────────────────────────────────────────────────────┘
```

### CoT Data Structure Comparison

```
┌──────────────────────────────────────────────────────────────────────┐
│                         SIMPLE QA FORMAT                             │
├──────────────────────────────────────────────────────────────────────┤
│ {                                                                    │
│   "question": "What is HDF5 chunking?",                             │
│   "answer": "Chunking divides datasets into blocks for efficient    │
│              partial I/O operations."                                │
│ }                                                                    │
│                                                                      │
│ Training Result: Model memorizes Q→A pattern                        │
│ Generalization: Poor on variations                                  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                       CoT FORMAT (WITH REASONING)                    │
├──────────────────────────────────────────────────────────────────────┤
│ {                                                                    │
│   "question": "What is HDF5 chunking?",                             │
│   "reasoning": [                                                     │
│     "Large datasets often exceed available memory capacity",         │
│     "Applications typically need only specific data subsets",        │
│     "Chunking divides data into smaller, independently accessible    │
│      blocks stored contiguously",                                    │
│     "Each chunk can be read without loading the entire dataset",     │
│     "This enables efficient random access and partial I/O"           │
│   ],                                                                 │
│   "answer": "Chunking divides datasets into blocks for efficient    │
│              partial I/O operations."                                │
│ }                                                                    │
│                                                                      │
│ Training Result: Model learns reasoning process (Q→Steps→A)          │
│ Generalization: Excellent - applies reasoning to new scenarios      │
└──────────────────────────────────────────────────────────────────────┘
```


### Hybrid Workflow (Recommended)

```
┌────────────────────────────────────────────────────────────────────┐
│                 COST-EFFECTIVE CoT STRATEGY                         │
│                                                                     │
│   Step 1: Generate Base QA                                         │
│   ┌─────────────────────────────┐                                 │
│   │  qa_generator.py            │  Fast, cheap                    │
│   │  Generate 500-1000 pairs    │  $5-10 cost                     │
│   └─────────────┬───────────────┘                                 │
│                 │                                                   │
│   Step 2: Curate for Quality    ▼                                 │
│   ┌─────────────────────────────┐                                 │
│   │  curate.py                  │  Human + LLM-as-Judge           │
│   │  Filter to best 200-300     │  Keep only high quality         │
│   └─────────────┬───────────────┘                                 │
│                 │                                                   │
│   Step 3: Add CoT to Best Pairs ▼                                 │
│   ┌─────────────────────────────┐                                 │
│   │  cot_enhancer.py            │  Add reasoning to curated pairs │
│   │  Enhance 200-300 pairs      │  More efficient than full CoT   │
│   └─────────────┬───────────────┘  $10-15 cost                    │
│                 │                                                   │
│   Step 4: Final Curation        ▼                                 │
│   ┌─────────────────────────────┐                                 │
│   │  curate.py                  │  Validate reasoning quality     │
│   │  Final 150-200 pairs        │  Training-ready                 │
│   └─────────────┬───────────────┘                                 │
│                 │                                                   │
│                 ▼                                                   │
│   ┌─────────────────────────────┐                                 │
│   │  Training Dataset            │                                 │
│   │  150-200 high-quality        │                                 │
│   │  CoT examples                │                                 │
│   └──────────────────────────────┘                                 │
│                                                                     │
│   Total Cost: ~$25-30 (vs $100+ for direct CoT generation)         │
│   Quality: HIGH (double curation + reasoning)                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Tool-Use Pipeline (Agentic Training)

```
┌────────────────────────────────────────────────────────────────────┐
│             Input: API Specifications (OpenAPI/JSON)                │
└────────────────┬───────────────────────────────────────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │  PARSE        │  API Definition Parser
         │  Tools        │  • Extract parameters
         │               │  • Classify complexity (simple/medium/complex)
         └───────┬───────┘  • Build tool schema
                 │
                 │  Output: tools.json
                 │  [{tool_id, name, description, parameters, returns, complexity, ...}]
                 │
                 ▼
         ┌───────────────┐
         │  GENERATE     │  Toolformer + ToolLLM + Gorilla
         │  Examples     │  
         │               │  Mode: Auto (balanced mix)
         └───────┬───────┘  ├─ Single-step (Toolformer)
                 │           ├─ Multi-step (ToolLLM)
                 │           └─ Documentation (Gorilla)
                 │
                 │  Output: examples.json
                 │  [{instruction, solution: {reasoning_path, tool_calls}, api_documentation}]
                 │
                 ▼
         ┌───────────────┐
         │  EXECUTE      │  APIGen Triple Verification
         │  & Verify     │  Layer 1: Format validation
         │               │  Layer 2: Execution (simulated/real)
         └───────┬───────┘  Layer 3: Semantic check
                 │
                 │  Output: verified.json
                 │  [... + {format_valid, execution_valid, semantic_valid, actual_results}]
                 │
                 ▼
         ┌───────────────┐
         │  CURATE       │  LLM-as-Judge for Tool Examples
         │  by Quality   │  • Valid tool calls?
         │               │  • Realistic scenarios?
         └───────┬───────┘  • Threshold filtering
                 │
                 │  Output: curated.json
                 │  [filtered verified examples]
                 │
                 ▼
         ┌───────────────┐
         │  EXPORT       │  Tool-Use Format
         │  to Format    │  • Function calling format
         │               │  • Messages with tool_calls
         └───────┬───────┘  • API documentation included
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│            Output: tool_training.json (Tool-Use Format)             │
│                   Ready for fine-tuning                             │
└────────────────────────────────────────────────────────────────────┘
```

### Generation Modes

```
                    ┌──────────────────┐
                    │   Tool Parser    │
                    │  (Classification) │
                    └────────┬──────────┘
                             │
                   ┌─────────┼─────────┐
                   │         │         │
                   ▼         ▼         ▼
              ┌────────┐ ┌──────┐ ┌─────────┐
              │ Simple │ │Medium│ │ Complex │
              │ Tools  │ │Tools │ │  Tools  │
              └────┬───┘ └───┬──┘ └────┬────┘
                   │         │         │
                   │         │         │
        ┌──────────┴─────────┴─────────┴──────────┐
        │                                          │
        ▼                                          ▼
┌──────────────────┐                    ┌──────────────────┐
│  Single-Step     │                    │   Multi-Step     │
│  (Toolformer)    │                    │   (ToolLLM)      │
│                  │                    │                  │
│  User: "Open X"  │                    │  User: "Get Y    │
│  ↓               │                    │         from X"  │
│  Thought: Need   │                    │  ↓               │
│    open_file()   │                    │  Step 1: Open    │
│  ↓               │                    │  Step 2: Read    │
│  Tool Call       │                    │  Step 3: Extract │
│  ↓               │                    │  ↓               │
│  Result          │                    │  Final Result    │
└──────────────────┘                    └──────────────────┘
        │                                          │
        └──────────────┬───────────────────────────┘
                       │
                       │  Both include:
                       │  • API Documentation (Gorilla)
                       │  • Reasoning explanation
                       │  • Expected results
                       │
                       ▼
              ┌─────────────────┐
              │ Complete Example │
              └─────────────────┘
```
### Advanced Tool-Use Features (Jan 2026) ⭐ NEW

#### Chain-First Generation (ToolGrad)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL vs CHAIN-FIRST                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  TRADITIONAL (Query → Tools):                                        │
│  ┌────────────────┐                                                  │
│  │  User Query    │ "Calculate mean temperature from HDF5 file"      │
│  └───────┬────────┘                                                  │
│          │                                                           │
│          ▼  LLM figures out tools                                   │
│  ┌────────────────┐                                                  │
│  │  Tool Calls    │  May miss dependencies, incomplete chains        │
│  └────────────────┘                                                  │
│                                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  CHAIN-FIRST (Tools → Query) - ToolGrad:                            │
│  ┌────────────────┐                                                  │
│  │  Tool Chain    │  open_file → read_dataset → calculate_mean       │
│  │  (Generated    │  Logical, dependency-aware                       │
│  │   First)       │                                                  │
│  └───────┬────────┘                                                  │
│          │                                                           │
│          ▼  Synthesize matching query                               │
│  ┌────────────────┐                                                  │
│  │  User Query    │  "Get average temperature from climate.h5"       │
│  │  (Synthesized) │  Naturally requires the full chain               │
│  └────────────────┘                                                  │
│                                                                      │
│  Result: More coherent multi-tool examples                           │
└──────────────────────────────────────────────────────────────────────┘
```

#### Turn-Level Filtering (ToolMind)

```
┌────────────────────────────────────────────────────────────────────┐
│              TURN-LEVEL QUALITY ASSESSMENT                          │
└────────────────────────────────────────────────────────────────────┘

Traditional (Overall Rating):
┌─────────────────────────────────────────────────────────────────────┐
│  Multi-step example with 5 tool calls                               │
│                                                                     │
│  Step 1: ✅ Good (8/10)                                             │
│  Step 2: ✅ Good (7/10)                                             │
│  Step 3: ❌ Poor (3/10)  ← Hidden weakness                          │
│  Step 4: ✅ Good (8/10)                                             │
│  Step 5: ✅ Good (9/10)                                             │
│                                                                     │
│  Overall Rating: 7.0/10 → PASSES threshold                          │
│  Problem: Poor step 3 teaches bad patterns!                         │
└─────────────────────────────────────────────────────────────────────┘

Turn-Level (ToolMind):
┌─────────────────────────────────────────────────────────────────────┐
│  Multi-step example with 5 tool calls                               │
│                                                                     │
│  Step 1: ✅ Pass (0.8) ≥ 0.7                                        │
│  Step 2: ✅ Pass (0.7) ≥ 0.7                                        │
│  Step 3: ❌ FAIL (0.3) < 0.7  ← Caught!                             │
│  Step 4: ─ (not evaluated)                                          │
│  Step 5: ─ (not evaluated)                                          │
│                                                                     │
│  Result: REJECT - Any step below threshold fails example            │
│  Benefit: No weak steps in training data                            │
└─────────────────────────────────────────────────────────────────────┘
```

#### Parameter Dependency Graphs (In-N-Out)

```
┌────────────────────────────────────────────────────────────────────┐
│              DEPENDENCY GRAPH VISUALIZATION                         │
└────────────────────────────────────────────────────────────────────┘

         ┌──────────────┐
         │  open_file   │
         │              │
         │ in: path     │
         │ out: File    │
         └──────┬───────┘
                │ File (output)
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│ read_dataset │  │ list_groups  │
│              │  │              │
│ in: File,    │  │ in: File     │
│     path     │  │ out: [str]   │
│ out: ndarray │  └──────────────┘
└──────┬───────┘
       │ ndarray (output)
       │
       ▼
┌──────────────┐
│ calc_stats   │
│              │
│ in: ndarray  │
│ out: Stats   │
└──────────────┘

Automatic Inference:
• open_file.out:File → read_dataset.in:File ✅
• read_dataset.out:ndarray → calc_stats.in:ndarray ✅
• Topological Order: [open_file, read_dataset, calc_stats]
```

#### Outcome-Oriented Evaluation (MCP-AgentBench)

```
┌────────────────────────────────────────────────────────────────────┐
│              OUTCOME vs STEP EVALUATION                             │
└────────────────────────────────────────────────────────────────────┘

Step Evaluation (Traditional):
┌─────────────────────────────────────────────────────────────────────┐
│  Task: "Get temperature data for 2023 from climate.h5"              │
│                                                                     │
│  ✅ Step 1: open_file("climate.h5") - correct                       │
│  ✅ Step 2: read_dataset("/2023/temp") - correct                    │
│  ❓ Evaluation: 2/2 steps correct = 100%                           │
│                                                                     │
│  But did the user get what they wanted?                             │
└─────────────────────────────────────────────────────────────────────┘

Outcome Evaluation (MCP-AgentBench):
┌─────────────────────────────────────────────────────────────────────┐
│  Task: "Get temperature data for 2023 from climate.h5"              │
│                                                                     │
│  Expected Outcomes:                                                  │
│  ├─ 📋 File opened successfully                                     │
│  ├─ 📋 Temperature data retrieved                                   │
│  ├─ 📋 Data is from year 2023                                       │
│  └─ 📋 Data is complete (no missing values)                         │
│                                                                     │
│  Actual Results:                                                     │
│  ├─ ✅ File opened: climate.h5                                      │
│  ├─ ✅ Data retrieved: ndarray(365, 24)                             │
│  ├─ ✅ Year verified: 2023                                          │
│  └─ ⚠️ Partial: 3 days have NaN values                             │
│                                                                     │
│  Outcome Score: 0.85 (3.5/4 requirements)                           │
│  Constraint Violations: ["missing_data"]                             │
│  Status: PARTIAL_SUCCESS                                            │
└─────────────────────────────────────────────────────────────────────┘
```
---

## Multi-Provider Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Generator Core Logic                             │
│                  (QA, CoT, Tool Generation)                         │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     │  Calls: llm.generate(prompt)
                     │
                     ▼
         ┌───────────────────────┐
         │   BaseLLMClient       │
         │   (Abstract Interface)│
         └───────────┬───────────┘
                     │
         ┌───────────┼─────────────────┬──────────────────┐
         │           │                 │                  │
         ▼           ▼                 ▼                  ▼
    ┌────────┐ ┌──────────┐     ┌──────────┐      ┌──────────┐
    │ Ollama │ │  Claude  │ ... │  Gemini  │  ... │  OpenAI  │
    │ Client │ │  Client  │     │  Client  │      │  Client  │
    └────────┘ └──────────┘     └──────────┘      └──────────┘
         │           │                 │                  │
         │           │                 │                  │
         ▼           ▼                 ▼                  ▼
    ┌────────┐ ┌──────────┐     ┌──────────┐      ┌──────────┐
    │Local   │ │Anthropic │     │ Google   │      │ OpenAI   │
    │Server  │ │   API    │     │   API    │      │   API    │
    └────────┘ └──────────┘     └──────────┘      └──────────┘
      (Free)      (Paid)           (Free Tier)       (Paid)
```

### Provider Selection Flow

```
User Configuration
  configs/config.yaml
        │
        │  llm.provider: "gemini"
        │
        ▼
  ┌──────────────┐
  │ get_client() │  Factory function
  └──────┬───────┘
         │
         │  Returns: GeminiClient(config)
         │
         ▼
  ┌──────────────┐
  │ Client Ready │
  └──────┬───────┘
         │
         │  With:
         │  • Rate limiting
         │  • Retry logic
         │  • Error handling
         │
         ▼
  ┌──────────────┐
  │ Generation   │
  └──────────────┘
```

---

## Data Flow: Document → Training Data

```
                              COMPLETE PIPELINE
                              
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Research   │ ──> │   Download   │ ──> │   Convert    │
│   Papers     │     │     PDFs     │     │  to Markdown │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                   │
                                                   ▼
                                          ┌──────────────┐
                                          │   Semantic   │
                                          │   Chunking   │
                                          └──────┬───────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LanceDB                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │ chunk_id │ text         │ vector      │ metadata          │     │
│  ├──────────┼──────────────┼─────────────┼──────────────────┤     │
│  │ 001      │ "HDF5 is..." │ [0.1, ...]  │ {source, type}   │     │
│  │ 002      │ "Parallel..."│ [0.2, ...]  │ {source, type}   │     │
│  │ ...      │ ...          │ ...         │ ...              │     │
│  └───────────────────────────────────────────────────────────┘     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            │  GENERATOR INPUT
                            │
                            ▼
             ┌──────────────────────────────┐
             │                              │
        QA Pipeline                  Tool-Use Pipeline
             │                              │
             ▼                              ▼
    ┌────────────────┐            ┌────────────────┐
    │  QA Pairs      │            │ Tool Examples  │
    │  (Knowledge)   │            │ (Agentic)      │
    └────────┬───────┘            └────────┬───────┘
             │                              │
             └──────────┬───────────────────┘
                        │
                        ▼
               ┌────────────────┐
               │  Curation &    │
               │  Quality Check │
               └────────┬───────┘
                        │
                        ▼
               ┌────────────────┐
               │  Export to     │
               │  Training      │
               │  Format        │
               └────────┬───────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   training.jsonl              │
        │   tool_training.json          │
        │                               │
        │   READY FOR FINE-TUNING       │
        └───────────────────────────────┘
```

---

## Quality Assurance System

```
                      ┌─────────────────────┐
                      │   Generated Data    │
                      │  (QA or Tool-Use)   │
                      └──────────┬──────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Pre-Filtering        │
                    │  • Empty fields?       │
                    │  • Minimum length?     │
                    │  • Format valid?       │
                    └──────────┬─────────────┘
                               │
                               ▼
                    ┌────────────────────────┐
                    │   LLM-as-Judge         │
                    │  • Clarity (0-3)       │
                    │  • Accuracy (0-3)      │
                    │  • Usefulness (0-2)    │
                    │  • Difficulty (0-2)    │
                    │  • Reasoning           │
                    └──────────┬─────────────┘
                               │
                               │  Rating ≥ Threshold?
                               │
                    ┌──────────┴──────────┐
                    │                     │
                  Yes                    No
                    │                     │
                    ▼                     ▼
            ┌───────────────┐    ┌──────────────┐
            │  Format Check │    │   FILTERED   │
            │  (QA vs CoT)  │    └──────────────┘
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  Preserve     │
            │  Original     │
            │  Format       │
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  HIGH QUALITY │
            │    OUTPUT     │
            └───────────────┘
```

### Tool-Use Triple Verification

```
              ┌─────────────────────┐
              │   Tool Example      │
              └──────────┬──────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
   ┌────────────────┐      ┌─────────────────┐
   │  Layer 1:      │      │  If fails any   │
   │  Format Valid? │─NO──>│  layer:         │
   └────────┬───────┘      │  REJECT         │
           YES             └─────────────────┘
            │
            ▼
   ┌────────────────┐
   │  Layer 2:      │
   │  Execution OK? │─NO──>─────┐
   └────────┬───────┘            │
           YES                   │
            │                    │
            ▼                    │
   ┌────────────────┐            │
   │  Layer 3:      │            │
   │  Semantic OK?  │─NO──>──────┤
   └────────┬───────┘            │
           YES                   │
            │                    │
            ▼                    ▼
   ┌────────────────┐   ┌───────────────┐
   │   VERIFIED     │   │   REJECTED    │
   │   EXAMPLE      │   └───────────────┘
   └────────────────┘
```

---

## Research Papers → Implementation

```
┌─────────────────────────────────────────────────────────────────────┐
│                      RESEARCH FOUNDATION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LIMA (Meta, 2023)                                                  │
│  └─> Quality > Quantity ──> curate.py (threshold ≥7/10)            │
│                                                                     │
│  Instruction Backtranslation (Meta, 2024)                           │
│  └─> Docs as answers ──> qa_generator.py (questions from chunks)   │
│                                                                     │
│  Distilling Step-by-Step (Google, 2023)                             │
│  └─> CoT reasoning ──> cot/cot_generator.py + cot/cot_enhancer.py  │
│                                                                     │
│  AlpaGasus (UMD, 2024)                                              │
│  └─> Multi-criteria ──> qa/curate.py (detailed scoring)            │
│                                                                     │
│  Toolformer (Meta, 2023)                                            │
│  └─> Single-step tools ──> tool/tool_generator.py (single mode)    │
│                                                                     │
│  Gorilla (Berkeley, 2024)                                           │
│  └─> API docs ──> tool/tool_generator.py (api_documentation field) │
│                                                                     │
│  ToolLLM (Tsinghua, 2024)                                           │
│  └─> Multi-step ──> tool/tool_generator.py (multi mode, DFSDT)     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                  NEW IMPLEMENTATIONS (Jan 2026) ⭐                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DEITA (2024)                                                       │
│  └─> 3D scoring (complexity+quality+diversity)                     │
│      ──> qa/multi_scorer.py (10x data efficiency)                  │
│                                                                     │
│  TOUCAN (Oct 2024)                                                  │
│  └─> Semantic clustering + coverage selection                      │
│      ──> tool/coverage_selector.py (40-60% reduction)              │
│                                                                     │
│  ToolMind (Nov 2025)                                                │
│  └─> Turn-level filtering (per-step quality)                       │
│      ──> tool/tool_curator.py filter_by_turn_quality()             │
│                                                                     │
│  ToolGrad (Aug 2025)                                                │
│  └─> Chain-first generation (tools → query)                        │
│      ──> tool/tool_generator.py generate_chain_first()             │
│                                                                     │
│  In-N-Out (2025)                                                    │
│  └─> Parameter dependency graphs                                    │
│      ──> tool/dependency_graph.py DependencyGraph class            │
│                                                                     │
│  MCP-AgentBench (2025)                                              │
│  └─> Outcome-oriented evaluation (task completion)                  │
│      ──> tool/outcome_evaluator.py OutcomeEvaluator class          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Organization

```
Generator/
│
├── src/generator/               ← Core Implementation
│   ├── __init__.py             ← Package exports
│   ├── cli.py                  ← Command-line interface
│   ├── formatters.py           ← Export (ChatML, Alpaca, ShareGPT, JSONL)
│   ├── prompt_loader.py        ← Load prompt templates
│   │
│   ├── clients/                 ← Multi-provider LLM support
│   │   ├── __init__.py
│   │   ├── base.py             ← Abstract interface
│   │   ├── ollama.py           ← Local Ollama
│   │   ├── claude.py           ← Anthropic Claude
│   │   ├── google_adk.py       ← Google Gemini (ADK)
│   │   ├── vllm.py             ← vLLM server
│   │   ├── openai.py           ← OpenAI
│   │   └── anthropic.py        ← Anthropic direct API
│   │
│   ├── qa/                      ← QA Pipeline (Knowledge Training) ⭐
│   │   ├── __init__.py
│   │   ├── qa_generator.py     ← QA generation (Instruction Backtranslation)
│   │   ├── curate.py           ← LLM-as-Judge (LIMA + AlpaGasus)
│   │   ├── enrich.py           ← Response rewriting
│   │   ├── compare.py          ← Dataset comparison
│   │   └── multi_scorer.py     ← DEITA multi-dimensional scoring ⭐ NEW
│   │
│   ├── cot/                     ← CoT Pipeline (Reasoning Enhancement) ⭐
│   │   ├── __init__.py
│   │   ├── cot_generator.py    ← CoT generation (Distilling Step-by-Step)
│   │   └── cot_enhancer.py     ← Add CoT to existing QA
│   │
│   └── tool/                    ← Tool-Use Pipeline (Agentic Training) ⭐
│       ├── __init__.py
│       ├── tool_schemas.py     ← Tool/Parameter dataclasses
│       ├── tool_parser.py      ← Parse API definitions
│       ├── tool_generator.py   ← Generate tool examples (Toolformer + ToolLLM + Gorilla + ToolGrad)
│       ├── tool_executor.py    ← Verify tool calls (APIGen)
│       ├── tool_curator.py     ← Filter tool examples + turn-level filtering (ToolMind) ⭐ NEW
│       ├── coverage_selector.py ← Semantic deduplication (TOUCAN) ⭐ NEW
│       ├── dependency_graph.py  ← Parameter dependency graphs (In-N-Out) ⭐ NEW
│       └── outcome_evaluator.py ← Outcome-oriented evaluation (MCP-AgentBench) ⭐ NEW
│
├── configs/                     ← Configuration
│   ├── config.yaml             ← LLM providers
│   ├── hdf5_tools.json         ← Tool definitions
│   └── prompts/                ← Prompt templates
│       ├── qa_generation.yaml
│       ├── qa_rating.yaml
│       ├── cot_generation.yaml
│       ├── cot_enhancement.yaml
│       └── tool_prompts.yaml
│
├── tests/                       ← Test suite (188 tests total)
│   ├── conftest.py
│   ├── test_qa_generator.py
│   ├── test_curate.py
│   ├── test_enrich.py
│   ├── test_cot_generator.py
│   ├── test_cot_enhancer.py
│   ├── test_tool_use.py
│   ├── test_multi_scorer.py     ← 22 tests ⭐ NEW
│   ├── test_coverage_selector.py ← 16 tests ⭐ NEW
│   ├── test_turn_level_filter.py ← 7 tests ⭐ NEW
│   ├── test_chain_first.py      ← 7 tests ⭐ NEW
│   ├── test_dependency_graph.py ← 19 tests ⭐ NEW
│   ├── test_outcome_evaluator.py ← 20 tests ⭐ NEW
│   └── ...
│
├── output/                      ← Generated datasets
│   ├── hdf5_qa/
│   └── jarvis_qa/
│
└── docs/                        ← Documentation
    ├── OVERVIEW.md                        ← Quick overview
    ├── DESIGN_DOCUMENTATION.md            ← Design rationale
    ├── EXTRACTION_METHODOLOGY.md          ← How extraction works
    ├── PAPER_IMPLEMENTATIONS.md           ← Paper → code mapping
    ├── ARCHITECTURE_DIAGRAMS.md           ← Visual diagrams (this file)
    └── Papers_Analysis_and_Relationships.md ← 33 papers analysis
```

---

**Visual representations of the Generator architecture, data flows, and quality assurance systems.**

**Document Version:** 2.0  
**Last Updated:** January 9, 2026
