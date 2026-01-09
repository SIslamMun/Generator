"""
Coverage-based selection for dataset curation.

Based on TOUCAN (Oct 2025): Reduces dataset size by 60% with no performance degradation.
Paper: https://arxiv.org/abs/2510.01179

Uses semantic clustering to select diverse, representative examples.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()
logger = logging.getLogger(__name__)


class CoverageSelector:
    """Select diverse, representative examples using semantic coverage."""
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True,
    ):
        """
        Initialize coverage selector.
        
        Args:
            model_name: Sentence transformer model for embeddings
            cache_embeddings: Whether to cache computed embeddings
        """
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self._encoder = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    @property
    def encoder(self):
        """Lazy load sentence transformer."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: uv pip install sentence-transformers"
                )
        return self._encoder
    
    def select_by_coverage(
        self,
        examples: List[Dict[str, Any]],
        target_count: Optional[int] = None,
        reduction_ratio: float = 0.4,
        n_clusters: Optional[int] = None,
        strategy: str = "centroid",
    ) -> List[Dict[str, Any]]:
        """
        Select representative examples using semantic clustering.
        
        Args:
            examples: List of QA pairs or tool examples
            target_count: Exact number to select (overrides reduction_ratio)
            reduction_ratio: Target size as ratio of original (default 40%)
            n_clusters: Number of clusters (auto-calculated if None)
            strategy: Selection strategy - "centroid" (closest to center) 
                      or "diverse" (spread across cluster)
            
        Returns:
            Selected subset with maximum semantic coverage
        """
        if len(examples) == 0:
            return []
        
        if len(examples) == 1:
            return examples
        
        # Calculate target size
        if target_count is not None:
            n_select = min(target_count, len(examples))
        else:
            n_select = max(1, int(len(examples) * reduction_ratio))
        
        if n_select >= len(examples):
            console.print(
                f"[yellow]Target ({n_select}) >= total ({len(examples)}), "
                f"returning all examples[/yellow]"
            )
            return examples
        
        console.print(
            f"[cyan]Coverage selection: {len(examples)} → {n_select} "
            f"({100 * n_select / len(examples):.1f}%)[/cyan]"
        )
        
        # Extract text and compute embeddings
        texts = self._extract_texts(examples)
        embeddings = self._compute_embeddings(texts)
        
        # Auto-calculate clusters if not specified
        if n_clusters is None:
            # Heuristic: sqrt(n_select) clusters, minimum 2
            n_clusters = max(2, min(n_select, int(np.sqrt(n_select) * 2)))
        
        n_clusters = min(n_clusters, len(examples))
        
        # Cluster embeddings
        from sklearn.cluster import KMeans
        
        console.print(f"[dim]Clustering into {n_clusters} groups...[/dim]")
        
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10,
            max_iter=300
        )
        clusters = kmeans.fit_predict(embeddings)
        
        # Select representatives
        if strategy == "centroid":
            selected_indices = self._select_by_centroid(
                embeddings, clusters, kmeans.cluster_centers_, n_select
            )
        elif strategy == "diverse":
            selected_indices = self._select_diverse(
                embeddings, clusters, n_select
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        selected = [examples[i] for i in selected_indices]
        
        # Add coverage metadata
        for i, idx in enumerate(selected_indices):
            selected[i]["_coverage_metadata"] = {
                "cluster_id": int(clusters[idx]),
                "original_index": idx,
            }
        
        # Log coverage statistics
        self._log_coverage_stats(clusters, selected_indices, n_clusters)
        
        return selected
    
    def _extract_texts(self, examples: List[Dict[str, Any]]) -> List[str]:
        """
        Extract text for embedding from various formats.
        
        Handles:
        - QA pairs: question + answer
        - Tool examples: instruction
        - Conversations: all messages concatenated
        - CoT: question + reasoning + answer
        """
        texts = []
        
        for ex in examples:
            text_parts = []
            
            # QA format
            if "question" in ex:
                text_parts.append(ex["question"])
            if "answer" in ex:
                text_parts.append(ex["answer"])
            
            # CoT format
            if "reasoning" in ex:
                reasoning = ex["reasoning"]
                if isinstance(reasoning, list):
                    text_parts.append(" ".join(reasoning))
                else:
                    text_parts.append(str(reasoning))
            
            # Tool instruction format
            if "instruction" in ex and "question" not in ex:
                text_parts.append(ex["instruction"])
            
            # Conversation format
            if "conversations" in ex:
                for msg in ex["conversations"]:
                    if isinstance(msg, dict):
                        text_parts.append(msg.get("content", msg.get("value", "")))
            
            # Messages format
            if "messages" in ex:
                for msg in ex["messages"]:
                    if isinstance(msg, dict):
                        text_parts.append(msg.get("content", ""))
            
            # Fallback: convert entire example to string
            if not text_parts:
                text_parts.append(str(ex))
            
            texts.append(" ".join(text_parts))
        
        return texts
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings with optional caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = hash(text)
            if self.cache_embeddings and text_hash in self._embedding_cache:
                embeddings.append(self._embedding_cache[text_hash])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Compute uncached embeddings
        if uncached_texts:
            console.print(f"[dim]Computing embeddings for {len(uncached_texts)} texts...[/dim]")
            new_embeddings = self.encoder.encode(
                uncached_texts, 
                show_progress_bar=len(uncached_texts) > 100
            )
            
            for j, idx in enumerate(uncached_indices):
                embeddings[idx] = new_embeddings[j]
                if self.cache_embeddings:
                    text_hash = hash(texts[idx])
                    self._embedding_cache[text_hash] = new_embeddings[j]
        
        return np.array(embeddings)
    
    def _select_by_centroid(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray,
        centers: np.ndarray,
        n_select: int,
    ) -> List[int]:
        """Select examples closest to cluster centers."""
        selected = []
        
        # Calculate per-cluster quota (proportional to cluster size)
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        total_count = sum(counts)
        
        quotas = {}
        remaining = n_select
        
        for cluster_id, count in zip(unique_clusters, counts):
            # Proportional allocation
            quota = max(1, int(n_select * count / total_count))
            quotas[cluster_id] = min(quota, count)
            remaining -= quotas[cluster_id]
        
        # Distribute remaining quota to largest clusters
        if remaining > 0:
            sorted_clusters = sorted(
                unique_clusters, 
                key=lambda c: counts[list(unique_clusters).index(c)], 
                reverse=True
            )
            for cluster_id in sorted_clusters:
                if remaining <= 0:
                    break
                cluster_size = counts[list(unique_clusters).index(cluster_id)]
                can_add = cluster_size - quotas[cluster_id]
                add = min(remaining, can_add)
                quotas[cluster_id] += add
                remaining -= add
        
        # Select from each cluster
        for cluster_id in unique_clusters:
            cluster_indices = np.where(clusters == cluster_id)[0]
            quota = quotas[cluster_id]
            
            # Distance to center
            center = centers[cluster_id]
            distances = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
            
            # Select closest to center
            closest = cluster_indices[np.argsort(distances)[:quota]]
            selected.extend(closest.tolist())
        
        return selected[:n_select]
    
    def _select_diverse(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray,
        n_select: int,
    ) -> List[int]:
        """Select diverse examples spread across clusters."""
        selected = []
        unique_clusters = np.unique(clusters)
        
        # Round-robin selection from clusters
        cluster_indices = {
            c: list(np.where(clusters == c)[0]) 
            for c in unique_clusters
        }
        
        while len(selected) < n_select:
            for cluster_id in unique_clusters:
                if len(selected) >= n_select:
                    break
                
                available = cluster_indices[cluster_id]
                if not available:
                    continue
                
                # Select the one most different from already selected
                if not selected:
                    idx = available[0]
                else:
                    selected_embeddings = embeddings[selected]
                    best_idx = None
                    best_min_dist = -1
                    
                    for idx in available:
                        min_dist = np.min(
                            np.linalg.norm(embeddings[idx] - selected_embeddings, axis=1)
                        )
                        if min_dist > best_min_dist:
                            best_min_dist = min_dist
                            best_idx = idx
                    
                    idx = best_idx
                
                selected.append(idx)
                cluster_indices[cluster_id].remove(idx)
        
        return selected
    
    def _log_coverage_stats(
        self, 
        clusters: np.ndarray, 
        selected_indices: List[int],
        n_clusters: int,
    ):
        """Log cluster coverage statistics."""
        unique, counts = np.unique(clusters, return_counts=True)
        selected_clusters = clusters[selected_indices]
        sel_unique, sel_counts = np.unique(selected_clusters, return_counts=True)
        
        console.print(f"  [green]✓ Clusters covered: {len(sel_unique)}/{len(unique)}[/green]")
        console.print(f"  [dim]Avg examples per cluster: {len(selected_indices)/len(sel_unique):.1f}[/dim]")
        
        # Show distribution
        distribution = {int(c): int(cnt) for c, cnt in zip(sel_unique, sel_counts)}
        console.print(f"  [dim]Distribution: {distribution}[/dim]")
    
    def compute_coverage_score(
        self,
        selected: List[Dict[str, Any]],
        all_examples: List[Dict[str, Any]],
    ) -> float:
        """
        Compute how well the selection covers the full dataset.
        
        Returns a score from 0 to 1, where 1 means perfect coverage.
        """
        if not selected or not all_examples:
            return 0.0
        
        selected_texts = self._extract_texts(selected)
        all_texts = self._extract_texts(all_examples)
        
        selected_embeddings = self._compute_embeddings(selected_texts)
        all_embeddings = self._compute_embeddings(all_texts)
        
        # For each example in all_examples, find distance to nearest selected
        # Lower total distance = better coverage
        total_distance = 0.0
        
        for emb in all_embeddings:
            distances = np.linalg.norm(selected_embeddings - emb, axis=1)
            total_distance += np.min(distances)
        
        # Normalize: perfect coverage would have 0 distance
        # Convert to score: 1 / (1 + avg_distance)
        avg_distance = total_distance / len(all_examples)
        coverage_score = 1.0 / (1.0 + avg_distance)
        
        return coverage_score


def select_by_coverage(
    examples: List[Dict[str, Any]],
    target_count: Optional[int] = None,
    reduction_ratio: float = 0.4,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    """
    Convenience function for coverage-based selection.
    
    Args:
        examples: List of QA pairs or tool examples
        target_count: Exact number to select
        reduction_ratio: Target size as ratio of original
        model_name: Sentence transformer model
        
    Returns:
        Selected diverse subset
    """
    selector = CoverageSelector(model_name=model_name)
    return selector.select_by_coverage(
        examples, 
        target_count=target_count,
        reduction_ratio=reduction_ratio
    )
