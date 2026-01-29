"""
CoT (Chain-of-Thought) Enhancer

Adds step-by-step reasoning to existing QA pairs.
Based on "Distilling Step-by-Step" methodology (Google, 2023).

Paper: https://arxiv.org/abs/2305.02301
"""

import json
import json5
import re
import logging
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

from ..prompt_loader import load_prompts

logger = logging.getLogger(__name__)


def enhance_with_cot(
    input_path: str,
    output_path: str,
    llm_config: Dict,
    batch_size: int = 5,
    workers: int = 1,
    prepend_pairs: List = None,
    intermediate_path: str = None,
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
        workers: Number of parallel workers (1=sequential, 4+ recommended for Ollama)
        prepend_pairs: Optional list of already-processed pairs to include in intermediate saves
        intermediate_path: Optional path for intermediate saves (defaults to output_path_intermediate.json)
    
    Returns:
        Dict with summary statistics
    """
    from ..clients import get_client

    # Load prompt template - use correct path relative to project root
    config_dir = Path(__file__).parent.parent.parent.parent / "configs"
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

    # Prepare batches
    batches = []
    for i in range(0, len(conversations), batch_size):
        batches.append(conversations[i : i + batch_size])

    # Setup intermediate file saving
    if intermediate_path is None:
        output_path_obj = Path(output_path)
        intermediate_path = output_path_obj.parent / f"{output_path_obj.stem}_intermediate.json"
    else:
        intermediate_path = Path(intermediate_path)
    
    save_freq = llm_config.get('rate_limit', {}).get('save_every_n_pairs', 50)
    
    # Initialize prepend_pairs if None
    if prepend_pairs is None:
        prepend_pairs = []

    if workers == 1:
        # Sequential processing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
            TextColumn("[yellow]({task.percentage:>3.0f}%)[/yellow]"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Enhancing with CoT...",
                total=total_pairs,
            )

            for batch in batches:
                try:
                    config = llm_config.copy() if isinstance(llm_config, dict) else {}
                    enhanced_batch = _enhance_batch(
                        batch, client, cot_enhance_prompt, config
                    )
                    enhanced_pairs.extend(enhanced_batch)
                    progress.update(task, advance=len(batch))
                    
                    # Save intermediate results periodically (include prepend_pairs)
                    if len(enhanced_pairs) % save_freq == 0:
                        combined = prepend_pairs + enhanced_pairs
                        with open(intermediate_path, "w") as f:
                            json.dump(combined, f, indent=2, ensure_ascii=False)
                        logger.info(f"💾 Saved {len(combined)} pairs to intermediate file ({len(prepend_pairs)} prepended + {len(enhanced_pairs)} new)")
                except Exception as e:
                    logger.warning(f"Failed to enhance batch: {e}")
                    enhanced_pairs.extend(_conversations_to_cot(batch))
                    progress.update(task, advance=len(batch))
    else:
        # Parallel processing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}[/cyan]"),
            TextColumn("[yellow]({task.percentage:>3.0f}%)[/yellow]"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Enhancing with CoT...",
                total=total_pairs,
            )

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                config = llm_config.copy() if isinstance(llm_config, dict) else {}
                
                for i, batch in enumerate(batches):
                    future = executor.submit(
                        _enhance_batch,
                        batch, client, cot_enhance_prompt, config
                    )
                    futures[future] = (i, batch)

                for future in as_completed(futures):
                    batch_idx, batch = futures[future]
                    try:
                        enhanced_batch = future.result()
                        enhanced_pairs.extend(enhanced_batch)
                        progress.update(task, advance=len(batch))
                        
                        # Save intermediate results periodically (include prepend_pairs)
                        if len(enhanced_pairs) % save_freq == 0:
                            combined = prepend_pairs + enhanced_pairs
                            with open(intermediate_path, "w") as f:
                                json.dump(combined, f, indent=2, ensure_ascii=False)
                            logger.info(f"💾 Saved {len(combined)} pairs to intermediate file ({len(prepend_pairs)} prepended + {len(enhanced_pairs)} new)")
                    except Exception as e:
                        logger.warning(f"Failed to enhance batch {batch_idx}: {e}")
                        enhanced_pairs.extend(_conversations_to_cot(batch))
                        progress.update(task, advance=len(batch))

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
    # Increase max_tokens significantly for batch processing to prevent truncation
    # With batch_size=5 and detailed reasoning, responses can be 10K+ tokens
    temperature = config.get('temperature', 0.2)
    max_tokens = config.get('max_tokens', 24576)  # Default to 24K for CoT batches
    response = client.generate(prompt, temperature=temperature, max_tokens=max_tokens)

    # Log response for debugging
    logger.debug(f"LLM Response preview (first 500 chars): {response[:500]}")

    # Parse response (expect array of conversations with reasoning)
    enhanced_convs = _parse_enhanced_response(response)
    
    # If parsing failed, return original conversations with fallback
    if not enhanced_convs:
        logger.warning(f"Parsing failed for batch of {len(conversations)} conversations")
        logger.debug(f"Response length: {len(response)} chars, preview: {response[:500]}")
        # Save debug info for investigation
        try:
            from pathlib import Path
            import time
            debug_dir = Path("debug_responses")
            debug_dir.mkdir(exist_ok=True)
            debug_file = debug_dir / f"cot_parse_fail_{int(time.time())}.txt"
            with open(debug_file, 'w') as f:
                f.write(f"Failed to parse batch of {len(conversations)} conversations\n")
                f.write(f"Response length: {len(response)}\n\n")
                f.write("="*80 + "\n")
                f.write(f"Response:\n{response}\n")
            logger.debug(f"Debug info saved to {debug_file}")
        except:
            pass
        return _conversations_to_cot(conversations)

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
        # Clean response - check for markdown wrapping FIRST
        cleaned_response = response.strip()
        
        # Remove markdown code blocks ONLY if they wrap the entire response
        # Check if response STARTS with markdown, not if it contains ``` inside the JSON
        if cleaned_response.startswith("```json"):
            # Remove opening ```json and closing ```
            parts = cleaned_response[7:].split("```", 1)  # Skip "```json" prefix
            if len(parts) > 0:
                cleaned_response = parts[0].strip()
        elif cleaned_response.startswith("```"):
            # Remove opening ``` and closing ```
            parts = cleaned_response[3:].split("```", 1)  # Skip "```" prefix
            if len(parts) > 0:
                cleaned_response = parts[0].strip()
        
        # Try direct JSON parse first (fastest path)
        try:
            result = json.loads(cleaned_response)
            if isinstance(result, list) and len(result) > 0:
                # Validate structure
                if isinstance(result[0], list):
                    logger.debug(f"Direct JSON parse successful, got {len(result)} conversations")
                    return result
                # Maybe it's a flat list of dicts, wrap them
                logger.debug("Got flat list, attempting to restructure")
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parse failed: {e}")
            pass
        
        # Extract JSON array - find the outermost brackets
        start_idx = cleaned_response.find("[")
        if start_idx == -1:
            logger.warning("No JSON array found in enhanced response")
            logger.debug(f"Response preview: {cleaned_response[:300]}")
            return []
        
        # Count brackets to find the matching closing bracket
        bracket_count = 0
        end_idx = -1
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(cleaned_response)):
            char = cleaned_response[i]
            
            # Handle string escapes
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
                
            # Track if we're inside a string
            if char == '"':
                in_string = not in_string
                continue
                
            # Only count brackets outside strings
            if not in_string:
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_idx = i
                        break
        
        if end_idx == -1:
            logger.warning("No matching closing bracket found")
            logger.debug(f"Bracket count at end: {bracket_count}, searched {len(cleaned_response) - start_idx} chars")
            return []

        json_str = cleaned_response[start_idx : end_idx + 1]
        
        # Try multiple parsing strategies
        # Strategy 1: Standard JSON
        try:
            conversations = json.loads(json_str)
            if isinstance(conversations, list) and len(conversations) > 0:
                logger.debug("Standard JSON parse successful")
                return conversations
        except json.JSONDecodeError as json_err:
            logger.debug(f"Standard JSON parse failed: {json_err}")
        
        # Strategy 2: json5 for relaxed JSON (trailing commas, single quotes, etc.)
        try:
            conversations = json5.loads(json_str)
            if isinstance(conversations, list) and len(conversations) > 0:
                logger.debug("json5 parse successful")
                return conversations
        except Exception as parse_err:
            logger.debug(f"json5 parse failed: {parse_err}")
        
        # Strategy 3: Fix trailing commas and retry
        try:
            fixed_json = re.sub(r',\s*([}\]])', r'\1', json_str)
            conversations = json.loads(fixed_json)
            if isinstance(conversations, list) and len(conversations) > 0:
                logger.debug("Fixed trailing commas successfully")
                return conversations
        except Exception as fix_err:
            logger.debug(f"Fix trailing commas failed: {fix_err}")
        
        # Strategy 4: Try to fix common quote escaping issues
        try:
            # Replace problematic escaped quotes in content
            fixed_json = json_str.replace('\\"', "'")
            conversations = json5.loads(fixed_json)
            if isinstance(conversations, list) and len(conversations) > 0:
                logger.debug("Fixed quote escaping successfully")
                return conversations
        except Exception as quote_err:
            logger.debug(f"Fix quote escaping failed: {quote_err}")
        
        # Strategy 5: Try with json5 on the original cleaned_response (skip bracket extraction)
        try:
            conversations = json5.loads(cleaned_response)
            if isinstance(conversations, list) and len(conversations) > 0:
                logger.debug("json5 on original cleaned response successful")
                return conversations
        except Exception as orig_err:
            logger.debug(f"json5 on original failed: {orig_err}")
        
        # Strategy 6: Last resort - try standard json on cleaned_response
        try:
            conversations = json.loads(cleaned_response)
            if isinstance(conversations, list) and len(conversations) > 0:
                logger.debug("Standard JSON on original cleaned response successful")
                return conversations
        except Exception as final_err:
            logger.debug(f"Final standard JSON parse failed: {final_err}")
        
        logger.warning("Could not parse conversations as list")
        logger.debug(f"Failed JSON string preview (first 500): {json_str[:500]}")
        logger.debug(f"Failed JSON string preview (last 200): ...{json_str[-200:]}")
        return []

    except Exception as e:
        logger.error(f"Failed to parse enhanced response: {e}")
        logger.debug(f"Response preview: {response[:500]}")
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
