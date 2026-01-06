"""
Compare multiple QA datasets using LLM judge to determine the best one.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

from .clients import get_client
from .prompt_loader import load_prompts

logger = logging.getLogger(__name__)


class DatasetComparator:
    """Compare multiple QA datasets using metrics and LLM judgment."""
    
    def __init__(self, llm_config: Dict[str, Any]):
        self.llm_config = llm_config
        self.client = get_client(
            llm_config["provider"],
            llm_config.get("api_key"),
            llm_config.get("model")
        )
        # Load prompts
        prompts_dir = Path(__file__).parent.parent.parent / "configs" / "prompts"
        self.prompts = load_prompts(prompts_dir)
    
    def load_datasets(self, paths: List[Path]) -> Dict[str, List[Dict]]:
        """Load multiple datasets."""
        datasets = {}
        for path in paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                datasets[path.stem] = data
                logger.info(f"Loaded {len(data)} pairs from {path.name}")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
        return datasets
    
    def compute_metrics(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Compute comprehensive metrics for a dataset."""
        if not dataset:
            return {"count": 0, "error": "Empty dataset"}
        
        metrics = {
            "count": len(dataset),
            "avg_question_length": sum(len(p.get("question", "")) for p in dataset) / len(dataset),
            "avg_answer_length": sum(len(p.get("answer", "")) for p in dataset) / len(dataset),
        }
        
        # Rating statistics if available
        ratings = [p.get("rating") for p in dataset if p.get("rating")]
        if ratings:
            metrics["avg_rating"] = sum(ratings) / len(ratings)
            metrics["min_rating"] = min(ratings)
            metrics["max_rating"] = max(ratings)
            metrics["rating_distribution"] = dict(Counter(ratings))
        
        # Source diversity
        sources = [p.get("source", "unknown") for p in dataset]
        metrics["unique_sources"] = len(set(sources))
        metrics["source_distribution"] = dict(Counter(sources).most_common(10))
        
        # Question type diversity (simple heuristic)
        question_starts = [p.get("question", "").split()[0].lower() for p in dataset if p.get("question")]
        metrics["question_type_diversity"] = len(set(question_starts))
        metrics["top_question_types"] = dict(Counter(question_starts).most_common(5))
        
        return metrics
    
    def sample_pairs(self, dataset: List[Dict], n: int = 10) -> List[Dict]:
        """Sample n random pairs from dataset."""
        import random
        if len(dataset) <= n:
            return dataset
        return random.sample(dataset, n)
    
    def llm_judge_samples(self, samples: Dict[str, List[Dict]]) -> Dict[str, str]:
        """Use LLM to judge quality of samples from each dataset."""
        results = {}
        
        for dataset_name, pairs in samples.items():
            # Format samples for prompt
            samples_text = ""
            for i, pair in enumerate(pairs, 1):
                samples_text += f"""
Sample {i}:
Q: {pair.get('question', 'N/A')}
A: {pair.get('answer', 'N/A')[:500]}...

"""
            
            # Use prompt template
            prompt = self.prompts["qa_comparison_evaluation"]["prompt"].format(
                sample_count=len(pairs),
                dataset_name=dataset_name,
                samples=samples_text.strip()
            )
            
            try:
                logger.info(f"LLM judging samples from {dataset_name}...")
                response = self.client.generate(prompt)
                results[dataset_name] = response
            except Exception as e:
                logger.error(f"LLM judgment failed for {dataset_name}: {e}")
                results[dataset_name] = f"ERROR: {str(e)}"
        
        return results
    
    def make_final_decision(self, 
                           metrics: Dict[str, Dict],
                           llm_judgments: Dict[str, str]) -> Dict[str, Any]:
        """Make final decision based on all data."""
        
        # Extract scores from LLM judgments
        scores = {}
        for name, judgment in llm_judgments.items():
            try:
                if "SCORE:" in judgment:
                    score_line = [l for l in judgment.split('\n') if 'SCORE:' in l][0]
                    score = float(score_line.split(':')[1].split('/')[0].strip())
                    scores[name] = score
            except:
                scores[name] = 0.0
        
        # Decision prompt
        metrics_text = ""
        for name, m in metrics.items():
            metrics_text += f"\n{name}:\n"
            metrics_text += f"  - Count: {m.get('count', 0)} pairs\n"
            metrics_text += f"  - Avg Rating: {m.get('avg_rating', 'N/A')}\n"
            metrics_text += f"  - Avg Answer Length: {int(m.get('avg_answer_length', 0))} chars\n"
            metrics_text += f"  - Source Diversity: {m.get('unique_sources', 0)} sources\n"
            metrics_text += f"  - Question Type Diversity: {m.get('question_type_diversity', 0)} types\n"
        
        scores_text = ""
        for name, score in scores.items():
            scores_text += f"  - {name}: {score}/10\n"
        
        # Use prompt template
        decision_prompt = self.prompts["qa_comparison_selection"]["prompt"].format(
            metrics_comparison=metrics_text.strip(),
            quality_scores=scores_text.strip()
        )
        
        try:
            decision = self.client.generate(decision_prompt)
            winner = None
            if "WINNER:" in decision:
                winner_line = [l for l in decision.split('\n') if 'WINNER:' in l][0]
                winner = winner_line.split('WINNER:')[1].strip()
            
            return {
                "winner": winner,
                "llm_decision": decision,
                "scores": scores,
                "metrics_summary": metrics
            }
        except Exception as e:
            logger.error(f"Final decision failed: {e}")
            # Fallback: pick highest LLM score
            if scores:
                winner = max(scores.items(), key=lambda x: x[1])[0]
                return {
                    "winner": winner,
                    "llm_decision": f"Fallback decision based on LLM score: {winner} ({scores[winner]}/10)",
                    "scores": scores,
                    "metrics_summary": metrics
                }
            return {"error": "Could not make decision"}
    
    def compare(self, 
                dataset_paths: List[Path],
                output_path: Path,
                sample_size: int = 10) -> Dict[str, Any]:
        """
        Compare datasets and generate comprehensive report.
        
        Args:
            dataset_paths: List of paths to QA JSON files
            output_path: Where to save comparison report
            sample_size: Number of samples to judge per dataset
        
        Returns:
            Comparison results dictionary
        """
        logger.info(f"Comparing {len(dataset_paths)} datasets...")
        
        # Load all datasets
        datasets = self.load_datasets(dataset_paths)
        if not datasets:
            raise ValueError("No datasets loaded successfully")
        
        # Compute metrics for each
        logger.info("Computing metrics for each dataset...")
        all_metrics = {}
        for name, data in datasets.items():
            all_metrics[name] = self.compute_metrics(data)
        
        # Sample and judge
        logger.info(f"Sampling {sample_size} pairs from each dataset for LLM judgment...")
        samples = {name: self.sample_pairs(data, sample_size) 
                  for name, data in datasets.items()}
        
        llm_judgments = self.llm_judge_samples(samples)
        
        # Make final decision
        logger.info("Making final recommendation...")
        final_decision = self.make_final_decision(all_metrics, llm_judgments)
        
        # Compile full report
        report = {
            "comparison_date": str(Path().absolute()),
            "datasets_compared": list(datasets.keys()),
            "metrics": all_metrics,
            "llm_judgments": llm_judgments,
            "final_decision": final_decision,
            "sample_size": sample_size
        }
        
        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Comparison report saved to {output_path}")
        logger.info(f"🏆 Recommended winner: {final_decision.get('winner', 'N/A')}")
        
        return report


def compare_datasets(
    dataset_paths: List[Path],
    output_path: Path,
    llm_config: Dict[str, Any],
    sample_size: int = 10
) -> Dict[str, Any]:
    """
    Main entry point for dataset comparison.
    
    Args:
        dataset_paths: Paths to QA datasets to compare
        output_path: Where to save comparison report
        llm_config: LLM configuration
        sample_size: How many samples to judge per dataset
    
    Returns:
        Comparison report
    """
    comparator = DatasetComparator(llm_config)
    return comparator.compare(dataset_paths, output_path, sample_size)
