"""Tests for multi-dimensional scoring (DEITA-style)."""

import pytest
import json
from unittest.mock import Mock, patch

from generator.qa.multi_scorer import (
    MultiDimensionalScorer,
    MultiScore,
    ScoreWeights,
    score_qa_pairs,
)


# --- Fixtures ---

@pytest.fixture
def sample_pairs():
    """Create sample QA pairs for testing."""
    return [
        {
            "question": "What is HDF5?",
            "answer": "HDF5 is a file format for storing large datasets.",
        },
        {
            "question": "How does HDF5 chunking improve performance for large datasets with random access patterns?",
            "answer": "HDF5 chunking improves performance by dividing datasets into fixed-size blocks. This allows: 1) Efficient partial I/O - only needed chunks are read. 2) Better compression - each chunk compressed independently. 3) Improved caching - chunks fit in memory. For random access, chunking reduces disk seeks by localizing related data.",
        },
        {
            "question": "Compare B-tree and linear indexing in HDF5.",
            "answer": "B-tree indexing offers O(log n) lookup but higher overhead. Linear indexing is O(n) but simpler for sequential access.",
        },
    ]


@pytest.fixture
def mock_llm():
    """Create a mock LLM client."""
    mock = Mock()
    return mock


# --- ScoreWeights Tests ---

class TestScoreWeights:
    """Tests for ScoreWeights dataclass."""
    
    def test_default_weights(self):
        """Test default weight values."""
        weights = ScoreWeights()
        assert weights.complexity == 0.3
        assert weights.quality == 0.5
        assert weights.diversity == 0.2
    
    def test_normalize_weights(self):
        """Test weight normalization."""
        weights = ScoreWeights(complexity=1.0, quality=1.0, diversity=1.0)
        normalized = weights.normalize()
        
        assert abs(normalized.complexity - 0.333) < 0.01
        assert abs(normalized.quality - 0.333) < 0.01
        assert abs(normalized.diversity - 0.333) < 0.01
    
    def test_normalize_zero_weights(self):
        """Test normalization with all zeros."""
        weights = ScoreWeights(complexity=0, quality=0, diversity=0)
        normalized = weights.normalize()
        
        # Should return same zeros
        assert normalized.complexity == 0
        assert normalized.quality == 0
        assert normalized.diversity == 0


# --- MultiScore Tests ---

class TestMultiScore:
    """Tests for MultiScore dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        score = MultiScore(
            complexity=7.0,
            quality=8.0,
            diversity=6.0,
            combined=7.3,
            clarity=8.0,
            accuracy=8.0,
            usefulness=8.0,
            reasoning_depth=3,
            knowledge_breadth=2,
            reasoning="Test reasoning",
        )
        
        d = score.to_dict()
        
        assert d["complexity"] == 7.0
        assert d["quality"] == 8.0
        assert d["diversity"] == 6.0
        assert d["combined"] == 7.3
        assert d["reasoning"] == "Test reasoning"


# --- MultiDimensionalScorer Tests ---

class TestMultiDimensionalScorer:
    """Tests for MultiDimensionalScorer class."""
    
    def test_init_no_llm(self):
        """Test initialization without LLM."""
        scorer = MultiDimensionalScorer()
        assert scorer.llm is None
        assert scorer.weights.complexity == 0.3
    
    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        weights = ScoreWeights(complexity=0.5, quality=0.3, diversity=0.2)
        scorer = MultiDimensionalScorer(weights=weights)
        
        assert scorer.weights.complexity == 0.5
        assert scorer.weights.quality == 0.3
    
    def test_heuristic_complexity_simple(self):
        """Test heuristic complexity for simple question."""
        scorer = MultiDimensionalScorer()
        result = scorer._heuristic_complexity("What is HDF5?")
        
        assert result["complexity"] >= 1
        assert result["complexity"] <= 10
        assert "reasoning" in result
    
    def test_heuristic_complexity_complex(self):
        """Test heuristic complexity for complex question."""
        scorer = MultiDimensionalScorer()
        result = scorer._heuristic_complexity(
            "How does HDF5 chunking improve performance and what are the trade-offs with different chunk sizes?"
        )
        
        # Should score higher due to "how" and "trade-off"
        simple_result = scorer._heuristic_complexity("What is HDF5?")
        assert result["complexity"] >= simple_result["complexity"]
    
    def test_heuristic_quality_short_answer(self):
        """Test heuristic quality for short answer."""
        scorer = MultiDimensionalScorer()
        result = scorer._heuristic_quality(
            "What is HDF5?",
            "A file format."  # Very short
        )
        
        assert result["quality"] < 6  # Should be lower quality
    
    def test_heuristic_quality_structured_answer(self):
        """Test heuristic quality for structured answer."""
        scorer = MultiDimensionalScorer()
        result = scorer._heuristic_quality(
            "What is HDF5?",
            "HDF5 is a file format with these features: 1. Large dataset support. 2. Compression. 3. Hierarchical structure."
        )
        
        # Should score higher due to structure
        assert result["quality"] >= 5
    
    def test_score_diversity_first_pair(self):
        """Test diversity score for first pair (no existing)."""
        scorer = MultiDimensionalScorer(use_embeddings=False)
        score = scorer.score_diversity("Question", "Answer", [])
        
        # No embeddings = neutral score
        assert score == 5.0
    
    def test_score_pair_no_llm(self, sample_pairs):
        """Test scoring a pair without LLM (heuristic mode)."""
        scorer = MultiDimensionalScorer(use_embeddings=False)
        pair = sample_pairs[1]  # Complex question
        
        score = scorer.score_pair(pair)
        
        assert isinstance(score, MultiScore)
        assert 0 <= score.complexity <= 10
        assert 0 <= score.quality <= 10
        assert 0 <= score.combined <= 10
    
    @patch("generator.qa.multi_scorer.get_client")
    def test_score_complexity_with_llm(self, mock_get_client):
        """Test complexity scoring with LLM."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "complexity": 8,
            "reasoning_depth": 4,
            "knowledge_breadth": 3,
            "cognitive_type": "analysis",
            "reasoning": "Multi-step reasoning required",
        })
        mock_get_client.return_value = mock_llm
        
        scorer = MultiDimensionalScorer(llm_config={"provider": "ollama", "model": "test"})
        scorer.llm = mock_llm
        
        result = scorer.score_complexity("How does X improve Y?")
        
        assert result["complexity"] == 8
        assert result["reasoning_depth"] == 4
    
    @patch("generator.qa.multi_scorer.get_client")
    def test_score_quality_with_llm(self, mock_get_client):
        """Test quality scoring with LLM."""
        mock_llm = Mock()
        mock_llm.generate.return_value = json.dumps({
            "clarity": 9,
            "accuracy": 8,
            "usefulness": 7,
            "quality": 8,
            "issues": [],
            "reasoning": "Well-structured answer",
        })
        mock_get_client.return_value = mock_llm
        
        scorer = MultiDimensionalScorer(llm_config={"provider": "ollama", "model": "test"})
        scorer.llm = mock_llm
        
        result = scorer.score_quality("Question?", "Detailed answer.")
        
        assert result["quality"] == 8
        assert result["clarity"] == 9
    
    def test_filter_by_combined_score(self, sample_pairs):
        """Test filtering by combined score."""
        scorer = MultiDimensionalScorer(use_embeddings=False)
        
        # Use low threshold to ensure some pass
        filtered = scorer.filter_by_combined_score(sample_pairs, min_score=1.0)
        
        assert len(filtered) <= len(sample_pairs)
        # Check scores are attached
        for pair in filtered:
            assert "multi_score" in pair
    
    def test_select_top_k(self, sample_pairs):
        """Test selecting top-k pairs."""
        scorer = MultiDimensionalScorer(use_embeddings=False)
        
        selected = scorer.select_top_k(sample_pairs, k=2, strategy="combined")
        
        assert len(selected) == 2
        for pair in selected:
            assert "multi_score" in pair


class TestScoreQaPairs:
    """Tests for convenience function."""
    
    def test_convenience_function(self, sample_pairs):
        """Test score_qa_pairs convenience function."""
        # Use low threshold and no LLM
        result = score_qa_pairs(sample_pairs, min_score=1.0)
        
        assert len(result) <= len(sample_pairs)


class TestJsonParsing:
    """Tests for JSON response parsing."""
    
    def test_parse_plain_json(self):
        """Test parsing plain JSON."""
        scorer = MultiDimensionalScorer()
        result = scorer._parse_json('{"complexity": 7}')
        assert result["complexity"] == 7
    
    def test_parse_json_with_markdown(self):
        """Test parsing JSON in markdown code block."""
        scorer = MultiDimensionalScorer()
        response = """```json
{"complexity": 8, "reasoning": "test"}
```"""
        result = scorer._parse_json(response)
        assert result["complexity"] == 8
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        scorer = MultiDimensionalScorer()
        result = scorer._parse_json("not json")
        assert result is None


class TestDiversityWithEmbeddings:
    """Tests for diversity scoring with embeddings."""
    
    def test_diversity_requires_embeddings(self, sample_pairs):
        """Test that diversity uses embeddings when available."""
        try:
            from sentence_transformers import SentenceTransformer
            has_embeddings = True
        except ImportError:
            has_embeddings = False
        
        scorer = MultiDimensionalScorer(use_embeddings=True)
        
        if has_embeddings:
            # With no existing pairs, returns neutral 5.0 (can't compute diversity)
            score = scorer.score_diversity(
                sample_pairs[0]["question"],
                sample_pairs[0]["answer"],
                [],
            )
            assert score == 5.0  # Neutral when no comparison available
            
            # Second pair compared to first should have some diversity computed
            score2 = scorer.score_diversity(
                sample_pairs[1]["question"],
                sample_pairs[1]["answer"],
                [sample_pairs[0]],
            )
            assert 0 <= score2 <= 10
            
            # Different pairs should have positive diversity
            score3 = scorer.score_diversity(
                sample_pairs[2]["question"],
                sample_pairs[2]["answer"],
                [sample_pairs[0], sample_pairs[1]],
            )
            assert 0 <= score3 <= 10


class TestWeightedScoring:
    """Tests for weighted combined scores."""
    
    def test_weights_affect_combined(self, sample_pairs):
        """Test that weights affect combined score."""
        # High complexity weight
        weights_complexity = ScoreWeights(complexity=0.8, quality=0.1, diversity=0.1)
        scorer_c = MultiDimensionalScorer(weights=weights_complexity, use_embeddings=False)
        
        # High quality weight
        weights_quality = ScoreWeights(complexity=0.1, quality=0.8, diversity=0.1)
        scorer_q = MultiDimensionalScorer(weights=weights_quality, use_embeddings=False)
        
        pair = sample_pairs[1]  # Complex question
        
        score_c = scorer_c.score_pair(pair)
        score_q = scorer_q.score_pair(pair)
        
        # Scores should differ based on weights
        # (exact comparison depends on heuristics, just verify they're different)
        assert score_c.combined != score_q.combined or \
               (score_c.complexity == score_c.quality)  # Equal only if dimensions equal
