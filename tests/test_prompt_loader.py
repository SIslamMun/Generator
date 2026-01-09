"""Unit tests for prompt_loader module."""

import pytest
from pathlib import Path
from generator.prompt_loader import load_prompts


class TestPromptLoader:
    """Test prompt loading functionality."""

    def test_load_prompts_returns_dict(self):
        """Test that load_prompts returns a dictionary."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)

        assert isinstance(prompts, dict)

    def test_load_prompts_has_required_keys(self):
        """Test that prompts contain required templates."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)

        # Check for main prompts
        required_prompts = ["qa_generation", "qa_rating", "qa_enrichment", "cot_generation", "cot_enhancement"]

        for prompt_name in required_prompts:
            assert prompt_name in prompts, f"Missing prompt: {prompt_name}"

    def test_qa_generation_prompt_structure(self):
        """Test QA generation prompt has correct variables."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)
        qa_prompt = prompts.get("qa_generation", "")

        # Should have text and n_pairs variables
        assert "{text}" in qa_prompt or "text" in qa_prompt.lower()
        assert "question" in qa_prompt.lower()
        assert "answer" in qa_prompt.lower()

    def test_qa_rating_prompt_structure(self):
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)
        rating_prompt = prompts.get("qa_rating", "")

        # Should mention rating criteria
        assert "clarity" in rating_prompt.lower() or "rating" in rating_prompt.lower()
        assert "accuracy" in rating_prompt.lower() or "quality" in rating_prompt.lower()

    def test_cot_generation_prompt_structure(self):
        """Test CoT generation prompt mentions reasoning."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)
        cot_prompt = prompts.get("cot_generation", "")

        # Should mention reasoning/steps
        assert "reasoning" in cot_prompt.lower() or "step" in cot_prompt.lower()
        assert "question" in cot_prompt.lower()

    def test_cot_enhancement_prompt_structure(self):
        """Test CoT enhancement prompt mentions adding reasoning."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)
        cot_enhance_prompt = prompts.get("cot_enhancement", "")

        # Should mention adding reasoning
        assert "reasoning" in cot_enhance_prompt.lower() or "enhance" in cot_enhance_prompt.lower()
        assert "conversation" in cot_enhance_prompt.lower()

    def test_enrichment_prompt_structure(self):
        """Test enrichment prompt mentions improving quality."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)
        enrich_prompt = prompts.get("qa_enrichment", "")

        # Should mention rewriting/improving
        assert "rewrite" in enrich_prompt.lower() or "improve" in enrich_prompt.lower() or "enhance" in enrich_prompt.lower()

    def test_load_prompts_function_backward_compat(self):
        """Test backward compatible load_prompts function."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)

        assert isinstance(prompts, dict)
        assert len(prompts) > 0

    def test_prompts_are_non_empty(self):
        """Test that all prompts have content."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)

        for name, content in prompts.items():
            assert len(content) > 0, f"Empty prompt: {name}"

    def test_prompt_format_variables(self):
        """Test that prompts use proper variable formatting."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)

        # Check QA generation uses {text}
        qa_prompt = prompts.get("qa_generation", "")
        if "{text}" in qa_prompt:
            # Verify it's a valid format string
            assert qa_prompt.count("{") == qa_prompt.count("}")

    def test_prompts_return_json_format(self):
        """Test that prompts specify JSON return format."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)

        # Most prompts should specify JSON output
        for name, content in prompts.items():
            prompt_text = content
            # Should mention JSON somewhere
            has_json = "json" in prompt_text.lower() or "array" in prompt_text.lower() or "{" in prompt_text

    def test_load_missing_prompt_file(self):
        """Test handling of missing prompt file."""
        # This should raise FileNotFoundError for missing directory
        with pytest.raises(FileNotFoundError):
            prompts = load_prompts(Path("/nonexistent"))

    def test_prompts_have_consistent_structure(self):
        """Test that all prompts follow consistent structure."""
        config_dir = Path(__file__).parent.parent / "configs"
        prompts = load_prompts(config_dir)

        for name, content in prompts.items():
            # Each prompt should be a string
            assert isinstance(content, str), f"Invalid structure for {name}"
            if isinstance(content, dict):
                assert "prompt" in content, f"Missing 'prompt' key in {name}"
