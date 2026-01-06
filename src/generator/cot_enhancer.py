"""
CoT (Chain-of-Thought) Enhancer

Adds step-by-step reasoning to existing QA pairs.
Based on "Distilling Step-by-Step" methodology (Google, 2023).

Paper: https://arxiv.org/abs/2305.02301
"""

import json
import json5
import logging
from pathlib import Path
from typing import Dict, List
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .prompt_loader import load_prompts

logger = logging.getLogger(__name__)


def enhance_with_cot(
    input_path: str,
    output_path: str,
    llm_config: Dict,
    batch_size: int = 5,
) -> Dict:
    """
    Add chain-of-thought reasoning to existing QA pairs.
    
    Converts QA pairs to conversation format, adds reasoning steps,
    and outputs enhanced conversations.
    
    Args:
        input_path: Path to input JSON file (QA pairs)
        output_path: Path to save enhanced pairs
        llm_config: LLM configuration dict
        batch_size: Number of pairs to enhance per batch
    
    Returns:
        Dict with summary statistics
    """
    from .clients import get_client

    # Load prompt template
    config_dir = Path(__file__).parent.parent.parent / "configs"
    prompts = load_prompts(config_dir)
    cot_enhance_prompt = prompts.get("cot_enhancement")
    if not cot_enhance_prompt:
        raise ValueError("CoT enhancement prompt not found in prompts")

    # Load input QA pairs
    logger.info(f"Loading QA pairs from {input_path}")
    with open(input_path, "r") as f:
        qa_pairs = json.load(f)

    if not isinstance(qa_pairs, list):
        raise ValueError("Input must be a list of QA pairs")

    total_pairs = len(qa_pairs)
    logger.info(f"Loaded {total_pairs} QA pairs")

    # Convert QA pairs to conversation format
    conversations = _qa_to_conversations(qa_pairs)

    # Initialize LLM client
    provider = llm_config.pop("provider")
    client = get_client(provider, llm_config)

    # Enhance in batches
    enhanced_pairs = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task(
            "[cyan]Enhancing with CoT reasoning...",
            total=total_pairs,
        )

        for i in range(0, len(conversations), batch_size):
            batch = conversations[i : i + batch_size]

            try:
                # Enhance batch with config from llm_config (includes cot settings)
                config = llm_config.copy() if isinstance(llm_config, dict) else {}
                enhanced_batch = _enhance_batch(
                    batch, client, cot_enhance_prompt, config
                )
                enhanced_pairs.extend(enhanced_batch)

                progress.update(
                    task,
                    advance=len(batch),
                    description=f"[cyan]Enhanced {len(enhanced_pairs)}/{total_pairs} pairs...",
                )

            except Exception as e:
                logger.warning(f"Failed to enhance batch {i}: {e}")
                # On failure, keep original pairs
                enhanced_pairs.extend(_conversations_to_cot(batch))
                progress.update(task, advance=len(batch))
                continue

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(enhanced_pairs, f, indent=2)

    logger.info(f"Saved {len(enhanced_pairs)} enhanced CoT pairs to {output_path}")

    return {
        "total_pairs": total_pairs,
        "enhanced_pairs": len(enhanced_pairs),
        "output_file": str(output_path),
    }


def _qa_to_conversations(qa_pairs: List[Dict]) -> List[List[Dict]]:
    """
    Convert QA pairs to conversation format.
    
    Input: [{"question": "...", "answer": "..."}]
    Output: [[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]]
    """
    conversations = []

    for pair in qa_pairs:
        if not isinstance(pair, dict):
            continue

        question = pair.get("question", "")
        answer = pair.get("answer", "")

        if not question or not answer:
            continue

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        conversations.append(conversation)

    return conversations


def _conversations_to_cot(conversations: List[List[Dict]]) -> List[Dict]:
    """
    Convert conversations back to CoT format (fallback, no reasoning added).
    
    Output: [{"question": "...", "reasoning": "", "answer": "..."}]
    """
    cot_pairs = []

    for conv in conversations:
        question = ""
        answer = ""

        for msg in conv:
            if msg["role"] == "user":
                question = msg["content"]
            elif msg["role"] == "assistant":
                answer = msg["content"]

        if question and answer:
            cot_pairs.append({
                "question": question,
                "reasoning": "",  # No reasoning added (fallback)
                "answer": answer,
            })

    return cot_pairs


def _enhance_batch(
    conversations: List[List[Dict]],
    client,
    prompt_template: str,
    config: Dict = None,
) -> List[Dict]:
    """
    Enhance a batch of conversations with CoT reasoning.
    
    Returns list of CoT pairs: [{"question": "...", "reasoning": "...", "answer": "..."}]
    """
    if config is None:
        config = {}
        
    # Format conversations as JSON string for prompt
    conversations_json = json.dumps(conversations, indent=2)

    # Generate prompt
    prompt = prompt_template.format(conversations=conversations_json)

    # Call LLM with lower temperature for consistency (default: 0.2)
    temperature = config.get('temperature', 0.2)
    response = client.generate(prompt, temperature=temperature)

    # Parse response (expect array of conversations with reasoning)
    enhanced_convs = _parse_enhanced_response(response)

    # Convert enhanced conversations to CoT format
    cot_pairs = []
    for conv in enhanced_convs:
        question = ""
        answer = ""
        reasoning = ""

        for msg in conv:
            if msg["role"] == "user":
                question = msg["content"]
            elif msg["role"] == "assistant":
                content = msg["content"]
                # Try to extract reasoning from enhanced answer
                reasoning, answer = _extract_reasoning(content)

        if question and answer:
            cot_pairs.append({
                "question": question,
                "reasoning": reasoning,
                "answer": answer,
            })

    return cot_pairs


def _parse_enhanced_response(response: str) -> List[List[Dict]]:
    """
    Parse enhanced conversations from LLM response.
    
    Expected format:
    [
      [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "Step 1: ...\nStep 2: ...\nTherefore, ..."}
      ]
    ]
    """
    try:
        # Extract JSON array
        start_idx = response.find("[")
        end_idx = response.rfind("]")

        if start_idx == -1 or end_idx == -1:
            logger.warning("No JSON array found in enhanced response")
            return []

        json_str = response[start_idx : end_idx + 1]
        conversations = json5.loads(json_str)

        if not isinstance(conversations, list):
            return []

        return conversations

    except Exception as e:
        logger.error(f"Failed to parse enhanced response: {e}")
        return []


def _extract_reasoning(content: str) -> tuple[str, str]:
    """
    Extract reasoning and answer from enhanced assistant response.
    
    Looks for patterns like:
    - "Step 1: ...\nStep 2: ...\nTherefore, answer"
    - "Let me think step by step:\n1. ...\n2. ...\nAnswer: ..."
    
    Returns: (reasoning, answer)
    """
    # Common reasoning indicators
    indicators = [
        "step 1:",
        "let me think",
        "let's break this down",
        "first,",
        "to answer this",
    ]

    content_lower = content.lower()

    # Check if content has reasoning indicators
    has_reasoning = any(ind in content_lower for ind in indicators)

    if not has_reasoning:
        return "", content

    # Try to split reasoning from final answer
    # Look for final answer indicators
    final_indicators = [
        "\ntherefore",
        "\nso,",
        "\nin conclusion",
        "\nfinal answer:",
        "\nanswer:",
    ]

    best_split = -1
    for ind in final_indicators:
        idx = content_lower.rfind(ind)
        if idx > best_split:
            best_split = idx

    if best_split > 0:
        reasoning = content[:best_split].strip()
        answer = content[best_split:].strip()
        # Clean up answer (remove "Therefore," etc.)
        for ind in final_indicators:
            if answer.lower().startswith(ind.strip()):
                answer = answer[len(ind):].strip()
                break
        return reasoning, answer

    # Fallback: treat everything as reasoning + answer
    return content, content
