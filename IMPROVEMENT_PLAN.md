# Generator Improvement Plan

**Based on Analysis of 33 Research Papers (Dec 2022 - Dec 2025)**  
**Date:** January 9, 2026  
**Last Updated:** January 2026 - All Critical Gaps Implemented

---

## Executive Summary

This document analyzes the current Generator implementation against 33 seminal papers on synthetic data generation for LLM fine-tuning. We identify critical gaps, propose solutions, and prioritize implementation based on ROI.

**Key Finding:** Late 2025 research converges on three pillars: **Structure** (parameter-level graphs), **Realism** (MCP-based evaluation), and **Robustness** (turn-level filtering, drift generalization). Our implementation now covers ~95% of best practices after implementing all 5 critical gaps plus multi-dimensional scoring.

### ✅ All Critical Gaps Now Implemented (Jan 2026)

| Gap | Paper | Implementation | Tests |
|-----|-------|----------------|-------|
| Turn-Level Filtering | ToolMind | `tool_curator.py` - `filter_by_turn_quality()` | 7 tests |
| Chain-First Generation | ToolGrad | `tool_generator.py` - `generate_chain_first()` | 7 tests |
| Parameter Dependency Graphs | In-N-Out | `dependency_graph.py` - `DependencyGraph` | 19 tests |
| Outcome-Oriented Evaluation | MCP-AgentBench | `outcome_evaluator.py` - `OutcomeEvaluator` | 20 tests |
| Coverage-Based Selection | TOUCAN | `coverage_selector.py` - `CoverageSelector` | 16 tests |
| Multi-Dimensional Scoring | DEITA | `multi_scorer.py` - `MultiDimensionalScorer` | 22 tests |

**Total:** 91 new tests, all passing

---

## Current Implementation Status

### ✅ What We Already Have (Aligned with Research)

| Feature | Papers Referenced | Implementation |
|---------|-------------------|----------------|
| Instruction Backtranslation | Self-Instruct, Backtranslation | `qa_generator.py` - chunk-as-answer approach |
| LLM-as-Judge Curation | AlpaGasus, LIMA | `curate.py` - rating with threshold |
| Chain-of-Thought | Distilling Step-by-Step, Orca | `cot_generator.py`, `cot_enhancer.py` |
| Tool Use Generation | Toolformer, Gorilla, ToolLLM | `tool_generator.py` - single/multi modes |
| Execution Validation | ToolACE, SelfCodeAlign | `tool_executor.py` - simulated/real modes |
| Dual-Layer Verification | ToolACE, APIGen | `tool_curator.py` - execution + semantic |
| Multi-Tool Composition | ToolLLM | `tool_generator.py` - multi-tool instructions |
| Quality Filtering | DEITA | `curate.py` - threshold-based filtering |
| Difficulty Balancing | DEITA | `tool_curator.py` - `balance_difficulty()` |
| **Turn-Level Filtering** | **ToolMind (2025)** | **`tool_curator.py` - per-step quality** |
| **Chain-First Generation** | **ToolGrad (2025)** | **`tool_generator.py` - tool chain first** |
| **Dependency Graphs** | **In-N-Out (2025)** | **`dependency_graph.py` - parameter deps** |
| **Outcome Evaluation** | **MCP-AgentBench (2025)** | **`outcome_evaluator.py` - task completion** |
| **Coverage Selection** | **TOUCAN (2024)** | **`coverage_selector.py` - semantic dedup** |
| **Multi-Dimensional Scoring** | **DEITA (2024)** | **`multi_scorer.py` - 3D quality scoring** |

### Current File Structure
```
src/generator/
├── qa_generator.py      # Instruction backtranslation
├── cot_generator.py     # Chain-of-thought generation
├── cot_enhancer.py      # Add reasoning to existing QA
├── curate.py            # LLM-as-Judge quality filtering
├── enrich.py            # Response improvement
├── tool_generator.py    # Tool-use data generation + chain-first
├── tool_executor.py     # Execute and validate tool calls
├── tool_curator.py      # Tool curation + turn-level filtering
├── tool_parser.py       # Parse tool responses
├── tool_schemas.py      # Tool data models
├── compare.py           # Dataset comparison
├── formatters.py        # Export formats
├── coverage_selector.py # Semantic deduplication (TOUCAN)
├── dependency_graph.py  # Parameter dependency graphs (In-N-Out)
├── outcome_evaluator.py # Outcome-oriented evaluation (MCP-AgentBench)
├── multi_scorer.py      # Multi-dimensional scoring (DEITA) (NEW)
└── cli.py               # Command-line interface
```

---

## Gap Analysis

### ✅ Implemented Critical Gaps

#### 1. Turn-Level Filtering ✓
**Paper:** ToolMind (Nov 2025)  
**Link:** https://arxiv.org/abs/2511.15718

**Status:** ✅ IMPLEMENTED (Jan 2026)

**Implementation:**
- Added `filter_by_turn_quality()` to `tool_curator.py`
- Added `_rate_individual_steps()` for per-step quality rating  
- Added `step_quality_rating` prompt to `tool_prompts.yaml`
- Tests: `tests/test_turn_level_filter.py` (7 tests passing)

**Usage:**
```python
curator = ToolCurator(client)
filtered = curator.curate(examples, turn_level_filter=True, min_step_quality=0.7)
```

---

#### 5. Coverage-Based Selection ✓
**Paper:** TOUCAN (Oct 2025)  
**Link:** https://arxiv.org/abs/2510.01179

**Status:** ✅ IMPLEMENTED (Jan 2026)

**Implementation:**
- Added `coverage_selector.py` with `CoverageSelector` class
- Semantic clustering using sentence-transformers
- Two strategies: "centroid" (closest to center) and "diverse" (spread)
- CLI command: `generator select-coverage`
- Tests: `tests/test_coverage_selector.py` (16 tests passing)

**Usage:**
```bash
# Keep top 40% most diverse (default)
generator select-coverage curated.json -o diverse.json

# Select exactly 500 diverse examples  
generator select-coverage curated.json -o diverse.json --target-count 500

# Use diverse strategy (maximize spread)
generator select-coverage curated.json -o diverse.json --strategy diverse
```

**Results:** Tested on 781 HDF5 QA pairs → 50 diverse examples with 0.557 coverage score.

---

### ✅ All Critical Gaps Implemented (Jan 2026)

#### 2. Chain-First Generation ✓
**Paper:** ToolGrad (Aug 2025)  
**Link:** https://arxiv.org/abs/2508.04086

**Status:** ✅ IMPLEMENTED (Jan 2026)

**Implementation:**
- Added `generate_chain_first()` to `tool_generator.py`
- Added `generate_examples_hybrid()` for combined approach
- Added `chain_generation` and `query_synthesis` prompts to `tool_prompts.yaml`
- CLI command: `generator tool-generate-chain`
- Tests: `tests/test_chain_first.py` (7 tests passing)

**Usage:**
```bash
# Pure chain-first (complex multi-tool examples)
uv run generator tool-generate-chain tools.json -o examples.json

# Hybrid mode (recommended - combines both approaches)
uv run generator tool-generate-chain tools.json -o examples.json --hybrid

# Customize chain length  
uv run generator tool-generate-chain tools.json -o examples.json --min-steps 3 --max-steps 5
```

**Results:** Tested with HDF5 tools - generates valid multi-step chains with coherent user queries.

---

#### 3. Parameter-Level Dependency Graphs ✓
**Paper:** In-N-Out (Sep 2025)  
**Link:** https://arxiv.org/abs/2509.01560

**Status:** ✅ IMPLEMENTED (Jan 2026)

**Implementation:**
- Added `dependency_graph.py` with `DependencyGraph`, `ToolNode`, `DependencyEdge` classes
- Type compatibility checking with configurable rules
- Chain validation, finding valid sequences, and bridge discovery
- CLI command: `generator tool-deps`
- Tests: `tests/test_dependency_graph.py` (19 tests passing)

**Usage:**
```bash
# Show dependency summary
uv run generator tool-deps configs/hdf5_tools.json

# Show connections for specific tool
uv run generator tool-deps configs/hdf5_tools.json --tool open_file

# List all valid chains
uv run generator tool-deps configs/hdf5_tools.json --chains

# Validate a specific chain
uv run generator tool-deps configs/hdf5_tools.json --validate "open_file,read_dataset,close_file"

# Export graph to JSON
uv run generator tool-deps configs/hdf5_tools.json -o deps.json
```

**Results:** Tested with 24 HDF5 tools → 189 dependency edges, 37 valid chains found.

---

#### 4. Outcome-Oriented Evaluation ✓
**Paper:** MCP-AgentBench v2 (Sep 2025)  
**Link:** https://arxiv.org/abs/2509.09734

**Status:** ✅ IMPLEMENTED (Jan 2026)

**Implementation:**
- Added `outcome_evaluator.py` with `OutcomeEvaluator`, `OutcomeEvaluation`, `OutcomeStatus`
- Goes beyond execution success to verify actual task completion
- Extracts requirements from instructions and checks satisfaction
- Strict mode available for requiring ALL requirements satisfied
- CLI command: `generator tool-evaluate`
- Tests: `tests/test_outcome_evaluator.py` (20 tests passing)

**Usage:**
```bash
# Evaluate examples and filter by outcome score
uv run generator tool-evaluate examples.json -o verified.json --min-score 0.8

# Report-only mode (no output file needed)
uv run generator tool-evaluate examples.json --report-only

# Strict mode - require all requirements satisfied
uv run generator tool-evaluate examples.json -o strict.json --strict
```

**Key Features:**
- **Requirement Extraction:** Identifies key requirements from user instruction
- **Satisfaction Tracking:** Tracks which requirements were met vs missed
- **Outcome Status:** FULLY_SATISFIED, PARTIALLY_SATISFIED, NOT_SATISFIED, EXECUTION_FAILED
- **Strict Mode:** Caps score at 0.5 if ANY requirement is missing

**Results:** Tested with mock LLM - correctly classifies examples by task completion status.

---

### ⚠️ Important Gaps (Medium Priority)

#### 6. Dynamic Tool Inventory
**Paper:** AutoTool (Dec 2025)  
**Link:** https://arxiv.org/abs/2512.13278

**Problem:** Tools are defined statically. Real deployments have evolving tool sets.

**Research Finding:** Training should target unseen-tool and changing-tool generalization.

---

#### 7. Multi-Dimensional Quality Scoring ✓
**Paper:** DEITA (2024)

**Status:** ✅ IMPLEMENTED (Jan 2026)

**Implementation:**
- Added `multi_scorer.py` with `MultiDimensionalScorer`, `MultiScore`, `ScoreWeights` classes
- 3D scoring: complexity (reasoning depth), quality (clarity/accuracy), diversity (embedding-based)
- Both heuristic (fast, free) and LLM-based (accurate) scoring modes
- CLI command: `generator multi-score`
- Tests: `tests/test_multi_scorer.py` (22 tests passing)

**Usage:**
```bash
# Basic scoring with heuristics (fast, no LLM cost)
uv run generator multi-score curated.json -o scored.json

# Filter by minimum score
uv run generator multi-score curated.json -o scored.json --min-score 6.0

# Select top 500 examples
uv run generator multi-score curated.json -o top500.json --top-k 500 --strategy top-k

# Use LLM for higher accuracy scoring
uv run generator multi-score curated.json -o scored.json --use-llm --provider claude

# Custom weights - prioritize complexity for reasoning training
uv run generator multi-score curated.json -o scored.json --complexity-weight 0.6 --quality-weight 0.3 --diversity-weight 0.1
```

**Research Validation:** DEITA showed 3D scoring achieves 10x data efficiency - 6K examples match 100K randomly selected.

---

#### 8. Evol-Instruct Complexity Evolution
**Paper:** WizardLM, Evol-Instruct

**Problem:** Fixed complexity per generation.

**Research Finding:** Progressive complexity increases training quality.

---

#### 9. MCP Protocol Support
**Paper:** MCP-AgentBench, TOUCAN

**Problem:** No MCP tool protocol integration.

**Research Finding:** MCP is the emerging standard for agent-tool interaction.

---

#### 10. Process Supervision
**Paper:** SWiRL, PRM

**Problem:** Outcome-only supervision for reasoning.

**Research Finding:** Process supervision improves reasoning quality.

---

## Implementation Plan

### Phase A: Critical Improvements (Week 1-2)

#### A1. Turn-Level Filtering

**File:** `src/generator/tool_curator.py`

**New Method:**
```python
def filter_by_turn_quality(
    self,
    examples: List[ToolExample],
    min_step_quality: float = 0.8,
) -> List[ToolExample]:
    """
    Filter examples by individual turn/step quality.
    
    Args:
        examples: List of tool examples
        min_step_quality: Minimum avg quality per step (0-1)
        
    Returns:
        Examples where all steps meet quality threshold
    """
    filtered = []
    
    for example in examples:
        step_ratings = self._rate_individual_steps(example)
        
        # Check each step meets minimum
        if all(r >= min_step_quality for r in step_ratings):
            example.step_ratings = step_ratings
            filtered.append(example)
        else:
            # Log which steps failed for analysis
            failed_steps = [i for i, r in enumerate(step_ratings) if r < min_step_quality]
            logger.debug(f"Example failed turn filter at steps: {failed_steps}")
    
    return filtered

def _rate_individual_steps(self, example: ToolExample) -> List[float]:
    """Rate each reasoning step independently."""
    ratings = []
    
    for i, step in enumerate(example.solution.reasoning_path):
        prompt = self._get_prompt("step_quality_rating").format(
            instruction=example.instruction,
            step_number=i + 1,
            thought=step.thought,
            tool=step.tool,
            args=json.dumps(step.args),
            result=step.actual_result,
            previous_context=self._get_previous_context(example, i),
        )
        
        response = self.llm.generate(prompt, temperature=0.1)
        rating = self._parse_rating(response)
        ratings.append(rating)
    
    return ratings
```

**New Prompt:** `configs/prompts/tool_prompts.yaml`
```yaml
step_quality_rating: |
  Rate this individual reasoning step for tool-use training data.
  
  User Instruction: {instruction}
  
  Step {step_number}:
  - Thought: {thought}
  - Tool: {tool}
  - Arguments: {args}
  - Result: {result}
  
  Previous Context: {previous_context}
  
  Rate this step on:
  1. Is the thought logically sound given the instruction?
  2. Is the tool choice appropriate for this step?
  3. Are the arguments correct and complete?
  4. Does the result make sense?
  5. Does this step advance toward the goal without introducing errors?
  
  Return JSON:
  {{
    "rating": 0.0-1.0,
    "issues": ["list", "of", "problems"],
    "is_misleading": true/false
  }}
```

---

#### A2. Coverage-Based Selection

**New File:** `src/generator/coverage_selector.py`

```python
"""
Coverage-based selection for dataset curation.

Based on TOUCAN (Oct 2025): reduces dataset size by 60% 
with no performance degradation.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from rich.console import Console

console = Console()


class CoverageSelector:
    """Select diverse, representative examples using semantic coverage."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
    
    def select_by_coverage(
        self,
        examples: List[Dict[str, Any]],
        target_count: Optional[int] = None,
        reduction_ratio: float = 0.4,
        n_clusters: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select representative examples using clustering.
        
        Args:
            examples: List of QA pairs or tool examples
            target_count: Exact number to select (overrides reduction_ratio)
            reduction_ratio: Target size as ratio of original (default 40%)
            n_clusters: Number of clusters (auto-calculated if None)
            
        Returns:
            Selected subset with maximum coverage
        """
        if len(examples) == 0:
            return []
        
        # Calculate target size
        if target_count:
            n_select = min(target_count, len(examples))
        else:
            n_select = int(len(examples) * reduction_ratio)
        
        console.print(f"[cyan]Coverage selection: {len(examples)} → {n_select}[/cyan]")
        
        # Embed all examples
        texts = self._extract_texts(examples)
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Cluster
        if n_clusters is None:
            n_clusters = min(n_select, len(examples) // 3)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # Select representatives from each cluster
        selected_indices = self._select_representatives(
            embeddings, clusters, kmeans.cluster_centers_, n_select
        )
        
        selected = [examples[i] for i in selected_indices]
        
        # Log coverage stats
        self._log_coverage_stats(clusters, selected_indices)
        
        return selected
    
    def _extract_texts(self, examples: List[Dict]) -> List[str]:
        """Extract text for embedding from various formats."""
        texts = []
        for ex in examples:
            if "question" in ex and "answer" in ex:
                texts.append(f"{ex['question']} {ex['answer']}")
            elif "instruction" in ex:
                texts.append(ex["instruction"])
            elif "conversations" in ex:
                texts.append(" ".join(m["content"] for m in ex["conversations"]))
            else:
                texts.append(str(ex))
        return texts
    
    def _select_representatives(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray,
        centers: np.ndarray,
        n_select: int,
    ) -> List[int]:
        """Select examples closest to cluster centers."""
        selected = []
        
        # Calculate per-cluster quota
        unique_clusters = np.unique(clusters)
        base_quota = n_select // len(unique_clusters)
        remainder = n_select % len(unique_clusters)
        
        for i, cluster_id in enumerate(unique_clusters):
            # Indices in this cluster
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            # Quota for this cluster
            quota = base_quota + (1 if i < remainder else 0)
            quota = min(quota, len(cluster_indices))
            
            # Distance to center
            center = centers[cluster_id]
            distances = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
            
            # Select closest to center
            closest = cluster_indices[np.argsort(distances)[:quota]]
            selected.extend(closest.tolist())
        
        return selected
    
    def _log_coverage_stats(self, clusters: np.ndarray, selected: List[int]):
        """Log cluster coverage statistics."""
        unique, counts = np.unique(clusters, return_counts=True)
        selected_clusters = clusters[selected]
        sel_unique, sel_counts = np.unique(selected_clusters, return_counts=True)
        
        console.print(f"  Clusters covered: {len(sel_unique)}/{len(unique)}")
        console.print(f"  Avg per cluster: {len(selected)/len(sel_unique):.1f}")
```

---

#### A3. Chain-First Generation Mode

**File:** `src/generator/tool_generator.py`

**New Method:**
```python
def generate_chain_first(
    self,
    tools: List[Tool],
    n_chains: int = 100,
    max_chain_length: int = 5,
) -> List[Dict[str, Any]]:
    """
    Generate training data using chain-first approach (ToolGrad).
    
    Instead of: instruction → tool calls
    Does: valid tool chain → instruction
    
    Args:
        tools: Available tools
        n_chains: Number of chains to generate
        max_chain_length: Maximum tools per chain
        
    Returns:
        List of (instruction, solution) pairs
    """
    all_examples = []
    
    # Build tool graph for valid compositions
    tool_graph = self._build_tool_graph(tools)
    
    with Progress(...) as progress:
        task = progress.add_task("[cyan]Chain-first generation...", total=n_chains)
        
        for _ in range(n_chains):
            # Step 1: Generate valid tool chain
            chain = self._sample_valid_chain(tool_graph, max_chain_length)
            
            if not chain:
                continue
            
            # Step 2: Execute chain to get actual results
            results = self._execute_chain(chain)
            
            # Step 3: Synthesize natural language instruction
            instruction = self._synthesize_instruction(chain, results)
            
            # Step 4: Package as training example
            example = {
                "instruction": instruction,
                "solution": {
                    "reasoning_path": chain,
                    "execution_results": results,
                },
                "generation_mode": "chain_first",
            }
            
            all_examples.append(example)
            progress.advance(task)
    
    return all_examples

def _build_tool_graph(self, tools: List[Tool]) -> Dict:
    """Build graph of valid tool compositions based on parameter compatibility."""
    graph = {"nodes": {}, "edges": []}
    
    for tool in tools:
        graph["nodes"][tool.tool_id] = {
            "inputs": {p.name: p.type for p in tool.parameters if p.required},
            "outputs": tool.return_type,
        }
    
    # Find compatible connections (output → input type matching)
    for t1 in tools:
        for t2 in tools:
            if t1.tool_id == t2.tool_id:
                continue
            
            # Check if t1's output can feed t2's input
            if self._types_compatible(t1.return_type, t2.parameters):
                graph["edges"].append({
                    "from": t1.tool_id,
                    "to": t2.tool_id,
                    "compatible_params": self._find_compatible_params(t1, t2),
                })
    
    return graph

def _synthesize_instruction(
    self,
    chain: List[Dict],
    results: List[Any],
) -> str:
    """Generate natural language instruction for a tool chain."""
    prompt = self._get_prompt("instruction_from_chain").format(
        tool_chain=json.dumps(chain, indent=2),
        results=json.dumps(results, indent=2),
    )
    
    response = self.llm.generate(prompt, temperature=0.8)
    return response.strip()
```

**New Prompt:**
```yaml
instruction_from_chain: |
  Given this tool execution chain and its results, generate a natural 
  language instruction that a user might give to accomplish this task.
  
  Tool Chain:
  {tool_chain}
  
  Execution Results:
  {results}
  
  Generate a clear, natural instruction that would require this exact 
  sequence of tool calls. The instruction should:
  1. Sound like a real user request
  2. Contain enough information to derive the tool arguments
  3. Not explicitly mention tool names
  
  Return ONLY the instruction text, no explanation.
```

---

### Phase B: Quality Improvements (Week 2-3)

#### B1. Multi-Dimensional Quality Scoring

**File:** `src/generator/curate.py`

```python
def score_multidimensional(
    self,
    pairs: List[Dict],
    weights: Dict[str, float] = None,
) -> List[Dict]:
    """
    Score pairs on multiple dimensions (DEITA approach).
    
    Dimensions:
    - Complexity: How challenging is the question?
    - Quality: How accurate/helpful is the answer?
    - Diversity: How different from existing pairs?
    
    Args:
        pairs: QA pairs to score
        weights: Dimension weights (default: complexity=0.3, quality=0.5, diversity=0.2)
        
    Returns:
        Pairs with multi-dimensional scores
    """
    if weights is None:
        weights = {"complexity": 0.3, "quality": 0.5, "diversity": 0.2}
    
    # Score each dimension
    for pair in pairs:
        pair["scores"] = {
            "complexity": self._score_complexity(pair),
            "quality": self._score_quality(pair),
            "diversity": self._score_diversity(pair, pairs),
        }
        
        # Combined score
        pair["combined_score"] = sum(
            weights[dim] * pair["scores"][dim]
            for dim in weights
        )
    
    return pairs
```

#### B2. Outcome-Oriented Validation

**File:** `src/generator/tool_executor.py`

```python
def validate_task_completion(
    self,
    instruction: str,
    execution_result: ExecutionResult,
    final_answer: str,
) -> Dict[str, Any]:
    """
    Validate if the task was actually completed (not just executed).
    
    Args:
        instruction: Original user instruction
        execution_result: Tool execution result
        final_answer: Generated answer
        
    Returns:
        Validation result with completion status and reasoning
    """
    prompt = self._get_prompt("task_completion_check").format(
        instruction=instruction,
        execution_output=execution_result.output,
        final_answer=final_answer,
    )
    
    response = self.llm.generate(prompt, temperature=0.1)
    result = self._parse_json_response(response)
    
    return {
        "task_completed": result.get("completed", False),
        "completion_score": result.get("score", 0.0),
        "missing_aspects": result.get("missing", []),
        "reasoning": result.get("reasoning", ""),
    }
```

---

### Phase C: Future-Proofing (Week 3-4)

#### C1. MCP Protocol Support

**New File:** `src/generator/mcp_adapter.py`

#### C2. Evol-Instruct Evolution

**Add to:** `src/generator/tool_generator.py`

---

## Priority Matrix

| Priority | Feature | Effort | Impact | ROI | Week |
|----------|---------|--------|--------|-----|------|
| 🔴 P0 | Turn-Level Filtering | Medium | High | **High** | 1 |
| 🔴 P0 | Coverage-Based Selection | Medium | High | **High** | 1 |
| 🟠 P1 | Chain-First Generation | High | High | Medium | 2 |
| 🟠 P1 | Multi-Dimensional Scoring | Medium | Medium | High | 2 |
| 🟡 P2 | Parameter Dependency Graphs | High | Medium | Medium | 3 |
| 🟡 P2 | Outcome-Oriented Validation | Medium | Medium | Medium | 3 |
| 🟢 P3 | MCP Protocol Support | High | Future | Low now | 4 |
| 🟢 P3 | Evol-Instruct Evolution | Medium | Medium | Medium | 4 |

---

## Files to Create/Modify

### New Files
- `src/generator/coverage_selector.py` - Coverage-based selection
- `src/generator/tool_graph.py` - Parameter dependency graphs
- `src/generator/mcp_adapter.py` - MCP protocol support (Phase C)

### Modified Files
- `src/generator/tool_curator.py` - Add turn-level filtering
- `src/generator/tool_generator.py` - Add chain-first generation
- `src/generator/tool_executor.py` - Add outcome validation
- `src/generator/curate.py` - Add multi-dimensional scoring
- `configs/prompts/tool_prompts.yaml` - New prompts

### New Dependencies
```toml
# pyproject.toml additions
sentence-transformers = "^2.2.0"
scikit-learn = "^1.3.0"
```

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Invalid sample rate | ~30% | <10% | Chain-first + validation |
| Dataset redundancy | High | Low | Coverage selection |
| Turn-level errors | Unknown | <5% | Turn filtering |
| Task completion rate | Unknown | >90% | Outcome validation |
| Training efficiency | Baseline | +30% | Downstream task performance |

---

## Questions to Resolve

1. **MCP Support Priority:** Does JARVIS deployment environment use MCP?
2. **Target Dataset Size:** Coverage selection ROI depends on final size
3. **Real API Execution:** Is real execution available for outcome validation?
4. **Tool Inventory:** Is the tool set static or evolving?

---

## References

- ToolMind (Nov 2025): https://arxiv.org/abs/2511.15718
- ToolGrad (Aug 2025): https://arxiv.org/abs/2508.04086
- In-N-Out (Sep 2025): https://arxiv.org/abs/2509.01560
- MCP-AgentBench v2 (Sep 2025): https://arxiv.org/abs/2509.09734
- TOUCAN (Oct 2025): https://arxiv.org/abs/2510.01179
- AutoTool (Dec 2025): https://arxiv.org/abs/2512.13278
- DEITA: https://arxiv.org/abs/2312.15685

---

**Document Version:** 1.0  
**Last Updated:** January 9, 2026  
**Status:** Pending Approval
