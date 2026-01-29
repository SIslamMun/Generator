"""
Multi-dimensional quality scoring for QA pairs.

Based on DEITA (2024): 3D scoring achieves 10x data efficiency.
Paper: https://arxiv.org/abs/2312.15685

Dimensions:
- Complexity: How challenging is the question? (reasoning depth)
- Quality: How accurate/helpful is the answer?
- Diversity: How different from other pairs in the dataset?
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from ..clients import get_client, BaseLLMClient

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class ScoreWeights:
    """Weights for multi-dimensional scoring."""
    complexity: float = 0.3
    quality: float = 0.5
    diversity: float = 0.2
    
    def normalize(self) -> "ScoreWeights":
        """Normalize weights to sum to 1.0."""
        total = self.complexity + self.quality + self.diversity
        if total > 0:
            return ScoreWeights(
                complexity=self.complexity / total,
                quality=self.quality / total,
                diversity=self.diversity / total,
            )
        return self


@dataclass
class MultiScore:
    """Multi-dimensional score for a QA pair."""
    complexity: float = 0.0  # 0-10
    quality: float = 0.0     # 0-10
    diversity: float = 0.0   # 0-10 (computed from embedding distance)
    combined: float = 0.0    # Weighted combination
    
    # Sub-scores for quality
    clarity: float = 0.0
    accuracy: float = 0.0
    usefulness: float = 0.0
    
    # Complexity breakdown
    reasoning_depth: int = 0      # Number of reasoning steps needed
    knowledge_breadth: int = 0    # Number of concepts involved
    
    reasoning: str = ""  # Explanation from LLM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "complexity": float(self.complexity),
            "quality": float(self.quality),
            "diversity": float(self.diversity),
            "combined": float(self.combined),
            "clarity": float(self.clarity),
            "accuracy": float(self.accuracy),
            "usefulness": float(self.usefulness),
            "reasoning_depth": int(self.reasoning_depth),
            "knowledge_breadth": int(self.knowledge_breadth),
            "reasoning": self.reasoning,
        }


class MultiDimensionalScorer:
    """
    Score QA pairs on multiple dimensions for efficient data selection.
    
    Based on DEITA: Instead of a single quality threshold, use 3D scoring:
    1. Complexity - Prefer challenging questions that require reasoning
    2. Quality - Prefer accurate, clear, helpful answers  
    3. Diversity - Prefer pairs that cover different topics/patterns
    """
    
    DEFAULT_PROMPTS = {
        "complexity_rating": """Rate the complexity of this question for LLM training.

**Question:** {question}

Consider:
1. **Reasoning Depth:** How many logical steps are needed to answer?
   - 1 = Direct fact lookup
   - 2-3 = Simple inference
   - 4-5 = Multi-step reasoning
   - 6+ = Complex analysis

2. **Knowledge Breadth:** How many concepts/domains does it span?
   - 1 = Single concept
   - 2-3 = Multiple related concepts
   - 4+ = Cross-domain knowledge

3. **Cognitive Load:** What type of thinking is required?
   - Low: Recognition, recall
   - Medium: Application, analysis
   - High: Synthesis, evaluation, creation

Rate complexity from 1-10:
- 1-3: Simple (single fact, direct answer)
- 4-6: Medium (requires inference or explanation)
- 7-10: Complex (multi-step reasoning, synthesis)

Return JSON:
{{
    "complexity": 1-10,
    "reasoning_depth": 1-6,
    "knowledge_breadth": 1-5,
    "cognitive_type": "recognition|application|analysis|synthesis|evaluation",
    "reasoning": "Brief explanation"
}}

Return ONLY valid JSON.""",

        "quality_rating": """Rate the quality of this QA pair for LLM training.

**Question:** {question}
**Answer:** {answer}

Rate each dimension 1-10:

1. **Clarity** (Is it well-written and easy to understand?)
   - Grammar, structure, unambiguous language

2. **Accuracy** (Is the answer factually correct?)
   - No hallucinations, supported by question context

3. **Usefulness** (Would this help train an LLM?)
   - Educational value, practical applicability

Return JSON:
{{
    "clarity": 1-10,
    "accuracy": 1-10,
    "usefulness": 1-10,
    "quality": (average of above),
    "issues": ["list of problems if any"],
    "reasoning": "Brief explanation"
}}

Return ONLY valid JSON."""
    }
    
    def __init__(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        prompts: Optional[Dict[str, str]] = None,
        weights: Optional[ScoreWeights] = None,
        use_embeddings: bool = True,
    ):
        """
        Initialize multi-dimensional scorer.
        
        Args:
            llm_config: LLM configuration for complexity/quality rating
            prompts: Custom prompt templates
            weights: Dimension weights (default: complexity=0.3, quality=0.5, diversity=0.2)
            use_embeddings: Whether to compute diversity via embeddings
        """
        self.prompts = {**self.DEFAULT_PROMPTS, **(prompts or {})}
        self.weights = (weights or ScoreWeights()).normalize()
        self.use_embeddings = use_embeddings
        self.llm: Optional[BaseLLMClient] = None
        self._embeddings_cache: Dict[str, np.ndarray] = {}
        self._encoder = None
        
        if llm_config:
            config = llm_config.copy()
            provider = config.pop("provider", "ollama")
            self.llm = get_client(provider, config)
    
    def _get_encoder(self):
        """Lazy load sentence transformer for diversity scoring."""
        if self._encoder is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed, diversity scoring disabled")
                self.use_embeddings = False
        return self._encoder
    
    def score_complexity(self, question: str) -> Dict[str, Any]:
        """
        Score question complexity.
        
        Args:
            question: The question text
            
        Returns:
            Dict with complexity score and breakdown
        """
        if not self.llm:
            # Heuristic fallback
            return self._heuristic_complexity(question)
        
        prompt = self.prompts["complexity_rating"].format(question=question)
        
        # Retry with exponential backoff for 503 errors
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(prompt, temperature=0.1)
                result = self._parse_json(response)
                
                if result:
                    return {
                        "complexity": float(result.get("complexity", 5)),
                        "reasoning_depth": int(result.get("reasoning_depth", 2)),
                        "knowledge_breadth": int(result.get("knowledge_breadth", 2)),
                        "cognitive_type": result.get("cognitive_type", "application"),
                        "reasoning": result.get("reasoning", ""),
                    }
            except Exception as e:
                error_str = str(e)
                if ("503" in error_str or "UNAVAILABLE" in error_str or 
                    "429" in error_str or "RESOURCE_EXHAUSTED" in error_str):
                    if attempt < max_retries - 1:
                        # Longer wait for 429 quota errors
                        wait_time = (2 ** attempt) * 10 if "429" in error_str else (2 ** attempt) * 5
                        logger.warning(f"Rate limit/overload error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                logger.warning(f"Complexity scoring failed: {e}")
        
        return self._heuristic_complexity(question)
    
    def _heuristic_complexity(self, question: str) -> Dict[str, Any]:
        """Fallback heuristic for complexity scoring."""
        # Simple heuristics based on question structure
        words = question.lower().split()
        
        # Complexity indicators
        reasoning_words = {"how", "why", "explain", "analyze", "compare", "evaluate", "design"}
        complex_words = {"trade-off", "optimize", "architecture", "mechanism", "relationship"}
        
        has_reasoning = any(w in words for w in reasoning_words)
        has_complex = any(w in question.lower() for w in complex_words)
        word_count = len(words)
        
        # Base score
        score = 4  # Medium baseline
        
        if has_reasoning:
            score += 2
        if has_complex:
            score += 1
        if word_count > 20:
            score += 1
        if "?" in question and question.count("?") == 1:
            score += 0  # Simple single question
        
        score = min(10, max(1, score))
        
        return {
            "complexity": score,
            "reasoning_depth": 2 if has_reasoning else 1,
            "knowledge_breadth": 2 if has_complex else 1,
            "cognitive_type": "analysis" if has_reasoning else "recall",
            "reasoning": "Heuristic scoring (no LLM)",
        }
    
    def score_quality(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Score answer quality.
        
        Args:
            question: The question text
            answer: The answer text
            
        Returns:
            Dict with quality score and breakdown
        """
        if not self.llm:
            return self._heuristic_quality(question, answer)
        
        prompt = self.prompts["quality_rating"].format(
            question=question,
            answer=answer,
        )
        
        # Retry with exponential backoff for 503 errors
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.generate(prompt, temperature=0.1)
                result = self._parse_json(response)
                
                if result:
                    clarity = float(result.get("clarity", 5))
                    accuracy = float(result.get("accuracy", 5))
                    usefulness = float(result.get("usefulness", 5))
                    
                    return {
                        "clarity": clarity,
                        "accuracy": accuracy,
                        "usefulness": usefulness,
                        "quality": (clarity + accuracy + usefulness) / 3,
                        "issues": result.get("issues", []),
                        "reasoning": result.get("reasoning", ""),
                    }
            except Exception as e:
                error_str = str(e)
                if ("503" in error_str or "UNAVAILABLE" in error_str or 
                    "429" in error_str or "RESOURCE_EXHAUSTED" in error_str):
                    if attempt < max_retries - 1:
                        # Longer wait for 429 quota errors
                        wait_time = (2 ** attempt) * 10 if "429" in error_str else (2 ** attempt) * 5
                        logger.warning(f"Rate limit/overload error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                logger.warning(f"Quality scoring failed: {e}")
        
        return self._heuristic_quality(question, answer)
    
    def _heuristic_quality(self, question: str, answer: str) -> Dict[str, Any]:
        """Fallback heuristic for quality scoring."""
        # Length-based heuristics
        q_len = len(question.split())
        a_len = len(answer.split())
        
        # Very short answers are often low quality
        if a_len < 10:
            quality = 4
        elif a_len < 30:
            quality = 6
        else:
            quality = 7
        
        # Answers much shorter than questions might be incomplete
        if q_len > 0 and a_len / q_len < 0.5:
            quality -= 1
        
        # Answers with structure (lists, steps) are often better
        if any(marker in answer for marker in ["1.", "2.", "- ", "• ", "First", "Second"]):
            quality += 1
        
        quality = min(10, max(1, quality))
        
        return {
            "clarity": quality,
            "accuracy": quality,  # Can't verify without LLM
            "usefulness": quality,
            "quality": quality,
            "issues": [],
            "reasoning": "Heuristic scoring (no LLM)",
        }
    
    def score_diversity(
        self,
        question: str,
        answer: str,
        existing_pairs: List[Dict[str, Any]],
    ) -> float:
        """
        Score diversity relative to existing pairs.
        
        Uses cosine distance from nearest neighbor in embedding space.
        
        Args:
            question: The question text
            answer: The answer text
            existing_pairs: List of existing QA pairs to compare against
            
        Returns:
            Diversity score 0-10 (10 = most diverse/unique)
        """
        if not self.use_embeddings or not existing_pairs:
            return 5.0  # Neutral if can't compute
        
        encoder = self._get_encoder()
        if encoder is None:
            return 5.0
        
        # Embed current pair
        text = f"{question} {answer}"
        current_emb = encoder.encode([text])[0]
        
        # Get embeddings for existing pairs
        existing_texts = [
            f"{p.get('question', '')} {p.get('answer', '')}"
            for p in existing_pairs
        ]
        
        if not existing_texts:
            return 10.0  # First pair is maximally diverse
        
        existing_embs = encoder.encode(existing_texts)
        
        # Compute cosine distances
        from numpy.linalg import norm
        
        distances = []
        for emb in existing_embs:
            cos_sim = np.dot(current_emb, emb) / (norm(current_emb) * norm(emb) + 1e-8)
            distances.append(1 - cos_sim)  # Convert to distance
        
        # Use minimum distance (nearest neighbor)
        min_distance = min(distances) if distances else 1.0
        
        # Scale to 0-10 (distance 0 = score 0, distance 1+ = score 10)
        diversity_score = min(10.0, min_distance * 10)
        
        return diversity_score
    
    def score_pair(
        self,
        pair: Dict[str, Any],
        existing_pairs: Optional[List[Dict[str, Any]]] = None,
    ) -> MultiScore:
        """
        Score a single QA pair on all dimensions.
        
        Args:
            pair: QA pair with 'question' and 'answer' keys
            existing_pairs: Existing pairs for diversity scoring
            
        Returns:
            MultiScore with all dimensions
        """
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        
        # Score each dimension
        complexity_result = self.score_complexity(question)
        quality_result = self.score_quality(question, answer)
        diversity = self.score_diversity(question, answer, existing_pairs or [])
        
        # Calculate combined score
        combined = (
            self.weights.complexity * complexity_result["complexity"] +
            self.weights.quality * quality_result["quality"] +
            self.weights.diversity * diversity
        )
        
        return MultiScore(
            complexity=complexity_result["complexity"],
            quality=quality_result["quality"],
            diversity=diversity,
            combined=combined,
            clarity=quality_result.get("clarity", 0),
            accuracy=quality_result.get("accuracy", 0),
            usefulness=quality_result.get("usefulness", 0),
            reasoning_depth=complexity_result.get("reasoning_depth", 0),
            knowledge_breadth=complexity_result.get("knowledge_breadth", 0),
            reasoning=f"Complexity: {complexity_result.get('reasoning', '')}; Quality: {quality_result.get('reasoning', '')}",
        )
    
    def score_pairs(
        self,
        pairs: List[Dict[str, Any]],
        incremental_diversity: bool = True,
        workers: int = 1,
    ) -> List[tuple[Dict[str, Any], MultiScore]]:
        """
        Score multiple QA pairs.
        
        Args:
            pairs: List of QA pairs
            incremental_diversity: If True, compute diversity incrementally
                                   (each pair compared to previous)
            workers: Number of parallel workers (1=sequential)
            
        Returns:
            List of (pair, score) tuples
        """
        results = []
        scored_pairs = []  # For incremental diversity
        
        from rich.progress import TaskProgressColumn, TimeRemainingColumn
        
        if incremental_diversity or workers == 1:
            # Sequential scoring (required for incremental diversity)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
                TextColumn("[yellow]({task.percentage:>3.0f}%)[/yellow]"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Scoring pairs...",
                    total=len(pairs),
                )
                
                for pair in pairs:
                    existing = scored_pairs if incremental_diversity else []
                    score = self.score_pair(pair, existing)
                    
                    results.append((pair, score))
                    scored_pairs.append(pair)
                    
                    progress.advance(task)
        else:
            # Parallel scoring (no incremental diversity)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
                TextColumn("[yellow]({task.percentage:>3.0f}%)[/yellow]"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Scoring pairs...",
                    total=len(pairs),
                )
                
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {}
                    for i, pair in enumerate(pairs):
                        future = executor.submit(self.score_pair, pair, [])
                        futures[future] = (i, pair)
                    
                    # Collect results in order
                    temp_results = [None] * len(pairs)
                    for future in as_completed(futures):
                        idx, pair = futures[future]
                        try:
                            score = future.result()
                            temp_results[idx] = (pair, score)
                            progress.advance(task)
                        except Exception as e:
                            console.print(f"[red]Error scoring pair {idx}: {e}[/red]")
                            progress.advance(task)
                    
                    results = [r for r in temp_results if r is not None]
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def filter_by_combined_score(
        self,
        pairs: List[Dict[str, Any]],
        min_score: float = 5.0,
        incremental_diversity: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Filter pairs by combined multi-dimensional score.
        
        Args:
            pairs: List of QA pairs
            min_score: Minimum combined score to keep (0-10)
            incremental_diversity: Compute diversity incrementally
            
        Returns:
            Filtered pairs with scores attached
        """
        scored = self.score_pairs(pairs, incremental_diversity)
        
        filtered = []
        for pair, score in scored:
            if score.combined >= min_score:
                # Attach scores to pair
                pair["multi_score"] = score.to_dict()
                filtered.append(pair)
        
        console.print(f"[green]✓ Filtered {len(filtered)}/{len(pairs)} pairs (min score: {min_score})[/green]")
        
        return filtered
    
    def select_top_k(
        self,
        pairs: List[Dict[str, Any]],
        k: int,
        strategy: str = "combined",
    ) -> List[Dict[str, Any]]:
        """
        Select top-k pairs by score.
        
        Args:
            pairs: List of QA pairs
            k: Number to select
            strategy: "combined", "complexity", "quality", or "diversity"
            
        Returns:
            Top-k pairs with scores attached
        """
        scored = self.score_pairs(pairs, incremental_diversity=(strategy == "diversity"))
        
        # Sort by selected dimension
        if strategy == "combined":
            scored.sort(key=lambda x: x[1].combined, reverse=True)
        elif strategy == "complexity":
            scored.sort(key=lambda x: x[1].complexity, reverse=True)
        elif strategy == "quality":
            scored.sort(key=lambda x: x[1].quality, reverse=True)
        elif strategy == "diversity":
            scored.sort(key=lambda x: x[1].diversity, reverse=True)
        
        # Select top-k
        selected = []
        for pair, score in scored[:k]:
            pair["multi_score"] = score.to_dict()
            selected.append(pair)
        
        console.print(f"[green]✓ Selected top {len(selected)} pairs by {strategy}[/green]")
        
        return selected
    
    def _parse_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response."""
        text = response.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json"):
                lines = lines[1:]
            elif lines[0] == "```":
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"JSON parse error: {text[:100]}")
            return None
    
    def _print_summary(self, results: List[tuple[Dict[str, Any], MultiScore]]) -> None:
        """Print scoring summary table."""
        if not results:
            return
        
        scores = [s for _, s in results]
        
        table = Table(title="Multi-Dimensional Scoring Summary")
        table.add_column("Dimension", style="cyan")
        table.add_column("Min", justify="right")
        table.add_column("Avg", justify="right")
        table.add_column("Max", justify="right")
        
        for dim in ["complexity", "quality", "diversity", "combined"]:
            values = [getattr(s, dim) for s in scores]
            table.add_row(
                dim.capitalize(),
                f"{min(values):.1f}",
                f"{sum(values)/len(values):.1f}",
                f"{max(values):.1f}",
            )
        
        console.print(table)


def score_qa_pairs(
    pairs: List[Dict[str, Any]],
    llm_config: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
    min_score: float = 5.0,
) -> List[Dict[str, Any]]:
    """
    Convenience function to score and filter QA pairs.
    
    Args:
        pairs: List of QA pairs
        llm_config: LLM configuration
        weights: Dimension weights dict (complexity, quality, diversity)
        min_score: Minimum combined score to keep
        
    Returns:
        Filtered pairs with multi-dimensional scores
    """
    score_weights = None
    if weights:
        score_weights = ScoreWeights(
            complexity=weights.get("complexity", 0.3),
            quality=weights.get("quality", 0.5),
            diversity=weights.get("diversity", 0.2),
        )
    
    scorer = MultiDimensionalScorer(
        llm_config=llm_config,
        weights=score_weights,
    )
    
    return scorer.filter_by_combined_score(pairs, min_score=min_score)
