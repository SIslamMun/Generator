#!/usr/bin/env python3
"""
Convert Jarvis datasets to FunctionGemma format for fine-tuning.

This script transforms:
1. CoT pairs → FunctionGemma with <think></think> blocks
2. Tool examples → FunctionGemma with function calls
3. QA pairs → Simple conversation format

Output: Single training dataset in FunctionGemma chat format
"""

import json
import re
from pathlib import Path
from typing import Any


# =============================================================================
# Jarvis-CD Tool Definitions (29 MCP tools)
# =============================================================================

JARVIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_pipeline",
            "description": "Create a new pipeline environment for data-centric workflows",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "Name/ID for the new pipeline"}
                },
                "required": ["pipeline_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_pipeline",
            "description": "Load an existing pipeline environment by ID, or the current one if not specified",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline to load"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_pipeline",
            "description": "Execute the pipeline, running all configured steps",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline to run"}
                },
                "required": ["pipeline_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "destroy_pipeline",
            "description": "Destroy a pipeline and clean up all associated files and resources",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline to destroy"}
                },
                "required": ["pipeline_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_pipeline",
            "description": "Re-apply environment and configuration to every package in a Jarvis pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline to update"}
                },
                "required": ["pipeline_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "build_pipeline_env",
            "description": "Build the pipeline execution environment for a given pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline to build"}
                },
                "required": ["pipeline_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "append_pkg",
            "description": "Add a package to a pipeline for execution or analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline"},
                    "pkg_type": {"type": "string", "description": "Type of package to add (e.g., ior, mdtest)"},
                    "pkg_id": {"type": "string", "description": "Unique identifier for the package"},
                    "do_configure": {"type": "boolean", "description": "Whether to configure after adding"},
                    "extra_args": {"type": "object", "description": "Additional configuration arguments"}
                },
                "required": ["pipeline_id", "pkg_type", "pkg_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remove_pkg",
            "description": "Remove a package and its files from a pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline"},
                    "pkg_id": {"type": "string", "description": "ID of the package to remove"}
                },
                "required": ["pipeline_id", "pkg_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "unlink_pkg",
            "description": "Unlink a package from a pipeline without deleting its files",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline"},
                    "pkg_id": {"type": "string", "description": "ID of the package to unlink"}
                },
                "required": ["pipeline_id", "pkg_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "configure_pkg",
            "description": "Configure a package in a pipeline with new settings",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline"},
                    "pkg_id": {"type": "string", "description": "ID of the package"},
                    "extra_args": {"type": "object", "description": "Configuration arguments"}
                },
                "required": ["pipeline_id", "pkg_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pkg_config",
            "description": "Retrieve the configuration of a specific package in a pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "ID of the pipeline"},
                    "pkg_id": {"type": "string", "description": "ID of the package"}
                },
                "required": ["pipeline_id", "pkg_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_list_pipelines",
            "description": "List all current pipelines under management",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_cd",
            "description": "Set the working pipeline context",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline_id": {"type": "string", "description": "Pipeline ID to set as current"}
                },
                "required": ["pipeline_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_reset",
            "description": "Reset manager to a clean state by destroying all pipelines and config",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_add_repo",
            "description": "Add a repository path to the manager",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repository path"},
                    "force": {"type": "boolean", "description": "Force add even if exists"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_remove_repo",
            "description": "Remove a repository from configuration",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string", "description": "Repository name to remove"}
                },
                "required": ["repo_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_list_repos",
            "description": "List all registered repositories",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_get_repo",
            "description": "Get detailed information about a repository",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string", "description": "Repository name"}
                },
                "required": ["repo_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_promote_repo",
            "description": "Promote a repository to higher priority",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_name": {"type": "string", "description": "Repository name to promote"}
                },
                "required": ["repo_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_create_config",
            "description": "Initialize manager directories and persist configuration",
            "parameters": {
                "type": "object",
                "properties": {
                    "config_dir": {"type": "string", "description": "Configuration directory path"},
                    "private_dir": {"type": "string", "description": "Private directory path"},
                    "shared_dir": {"type": "string", "description": "Shared directory path"}
                },
                "required": ["config_dir", "private_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_load_config",
            "description": "Load manager configuration from saved state",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_save_config",
            "description": "Save current configuration state to disk",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_bootstrap_list",
            "description": "List all bootstrap templates available",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_bootstrap_from",
            "description": "Bootstrap configuration based on a predefined machine template",
            "parameters": {
                "type": "object",
                "properties": {
                    "machine": {"type": "string", "description": "Machine template name"}
                },
                "required": ["machine"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_set_hostfile",
            "description": "Set and save the path to the hostfile for deployments",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to hostfile"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_construct_pkg",
            "description": "Generate a new package skeleton by type",
            "parameters": {
                "type": "object",
                "properties": {
                    "pkg_type": {"type": "string", "description": "Package type to construct"}
                },
                "required": ["pkg_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_graph_build",
            "description": "Construct or rebuild the graph with a given sleep delay",
            "parameters": {
                "type": "object",
                "properties": {
                    "net_sleep": {"type": "number", "description": "Sleep delay between operations"}
                },
                "required": ["net_sleep"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_graph_modify",
            "description": "Modify the current resource graph with a delay between operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "net_sleep": {"type": "number", "description": "Sleep delay between operations"}
                },
                "required": ["net_sleep"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "jm_graph_show",
            "description": "Print the resource graph to the console",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# System prompt for Jarvis-CD assistant
JARVIS_SYSTEM_PROMPT = """You are Jarvis, an expert HPC (High-Performance Computing) assistant specialized in the Jarvis-CD workflow management system. You help users with:

1. Pipeline Management: Creating, loading, running, and destroying pipelines
2. Package Management: Adding, configuring, and removing packages from pipelines
3. Repository Management: Managing code repositories and configurations
4. Environment Setup: Bootstrapping and configuring HPC environments
5. Workflow Optimization: Providing guidance on best practices for scientific computing

When you need to perform actions, use the available tools. Always think through problems step-by-step before taking action."""


def convert_cot_to_functiongemma(cot_data: list[dict]) -> list[dict]:
    """
    Convert CoT (Chain-of-Thought) pairs to FunctionGemma format with <think> blocks.
    
    Input format:
        {"question": "...", "reasoning": "...", "answer": "...", "type": "..."}
    
    Output format:
        {"messages": [...], "tools": [...]}
    """
    converted = []
    
    for item in cot_data:
        question = item.get("question", "").strip()
        reasoning = item.get("reasoning", "").strip()
        answer = item.get("answer", "").strip()
        
        # Skip invalid entries
        if not question or not answer:
            continue
            
        # Build messages
        messages = [
            {
                "role": "system",
                "content": JARVIS_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        # Build assistant response with thinking
        if reasoning:
            # Clean up reasoning text
            reasoning = reasoning.replace("Let me think through this step by step:\n\n", "")
            reasoning = reasoning.replace("Let me analyze this step by step:\n\n", "")
            assistant_content = f"<think>{reasoning}</think>\n{answer}"
        else:
            assistant_content = answer
            
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
        
        converted.append({
            "messages": messages,
            "tools": JARVIS_TOOLS
        })
    
    return converted


def convert_tool_to_functiongemma(tool_data: list[dict]) -> list[dict]:
    """
    Convert Tool examples to FunctionGemma format with function calls.
    
    Input format:
        {"instruction": "...", "solution": {"reasoning_path": [...], "final_answer": "..."}}
    
    Output format:
        {"messages": [...], "tools": [...]}
    """
    converted = []
    
    for item in tool_data:
        instruction = item.get("instruction", "").strip()
        solution = item.get("solution", {})
        reasoning_path = solution.get("reasoning_path", [])
        final_answer = solution.get("final_answer", "").strip()
        execution_result = item.get("execution_result", {})
        
        # Skip invalid entries
        if not instruction or not reasoning_path:
            continue
            
        # Build messages
        messages = [
            {
                "role": "system",
                "content": JARVIS_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": instruction
            }
        ]
        
        # Build thinking from reasoning path
        think_parts = []
        for step in reasoning_path:
            thought = step.get("thought", "")
            if thought:
                think_parts.append(f"Step {step.get('step', '')}: {thought}")
        
        think_text = "\n".join(think_parts) if think_parts else ""
        
        # Get tool calls from reasoning path
        tool_calls = []
        for step in reasoning_path:
            tool_name = step.get("tool", "")
            args = step.get("args", {})
            if tool_name and args:
                tool_calls.append({
                    "id": f"call_{len(tool_calls)+1}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": args
                    }
                })
        
        # Build assistant response
        if tool_calls:
            # Assistant with thinking and tool call
            assistant_msg = {
                "role": "assistant",
                "content": f"<think>{think_text}</think>" if think_text else "",
                "tool_calls": tool_calls
            }
            messages.append(assistant_msg)
            
            # Add tool response
            tool_result = execution_result.get("result", "Operation completed successfully")
            if isinstance(tool_result, dict):
                tool_result = json.dumps(tool_result)
            
            messages.append({
                "role": "tool",
                "name": tool_calls[0]["function"]["name"],
                "tool_call_id": tool_calls[0]["id"],
                "content": str(tool_result)
            })
            
            # Final assistant answer
            if final_answer:
                messages.append({
                    "role": "assistant",
                    "content": final_answer
                })
        else:
            # No tool calls, just thinking + answer
            if think_text:
                assistant_content = f"<think>{think_text}</think>\n{final_answer}"
            else:
                assistant_content = final_answer
                
            messages.append({
                "role": "assistant",
                "content": assistant_content
            })
        
        converted.append({
            "messages": messages,
            "tools": JARVIS_TOOLS
        })
    
    return converted


def convert_qa_to_functiongemma(qa_data: list[dict], sample_ratio: float = 0.3) -> list[dict]:
    """
    Convert QA pairs to FunctionGemma format (simple conversations without tool calls).
    Only use a subset to balance the dataset.
    
    Input format:
        {"question": "...", "answer": "...", "rating": ...}
    
    Output format:
        {"messages": [...], "tools": [...]}
    """
    import random
    
    # Sample a portion of QA data
    sample_size = int(len(qa_data) * sample_ratio)
    sampled = random.sample(qa_data, sample_size)
    
    converted = []
    
    for item in sampled:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        
        # Skip invalid entries
        if not question or not answer:
            continue
            
        # Build messages (no tool calls for simple QA)
        messages = [
            {
                "role": "system",
                "content": JARVIS_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
        
        converted.append({
            "messages": messages,
            "tools": JARVIS_TOOLS
        })
    
    return converted


def format_for_training(examples: list[dict], tokenizer=None) -> list[dict]:
    """
    Format examples for training using tokenizer.apply_chat_template.
    If tokenizer is None, return messages format for later processing.
    """
    if tokenizer is None:
        return examples
        
    formatted = []
    for ex in examples:
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tools=ex["tools"],
                add_generation_prompt=False,
                tokenize=False
            ).removeprefix("<bos>")
            
            formatted.append({"text": text})
        except Exception as e:
            print(f"Warning: Failed to format example: {e}")
            continue
            
    return formatted


def main():
    """Main conversion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Jarvis datasets to FunctionGemma format")
    parser.add_argument("--cot", default="jarvis_qa_cot_24w/jarvis_qa_cot_curated.json",
                        help="Path to CoT dataset")
    parser.add_argument("--tools", default="jarvis_tool_training_24w/jarvis_tool_curated.json",
                        help="Path to Tool dataset")
    parser.add_argument("--qa", default="jarvis_qa_full_24w_5pairs/jarvis_qa_curated.json",
                        help="Path to QA dataset")
    parser.add_argument("--output", default="jarvis_functiongemma_training.json",
                        help="Output file path")
    parser.add_argument("--qa-ratio", type=float, default=0.3,
                        help="Ratio of QA data to include (0.0-1.0)")
    parser.add_argument("--format-text", action="store_true",
                        help="Format as text using tokenizer (requires unsloth)")
    args = parser.parse_args()
    
    print("🔄 Loading datasets...")
    
    # Load datasets
    with open(args.cot) as f:
        cot_data = json.load(f)
    print(f"   CoT: {len(cot_data):,} examples")
    
    with open(args.tools) as f:
        tool_data = json.load(f)
    print(f"   Tools: {len(tool_data):,} examples")
    
    with open(args.qa) as f:
        qa_data = json.load(f)
    print(f"   QA: {len(qa_data):,} examples")
    
    print("\n🔄 Converting datasets...")
    
    # Convert each dataset
    cot_converted = convert_cot_to_functiongemma(cot_data)
    print(f"   CoT → {len(cot_converted):,} examples")
    
    tool_converted = convert_tool_to_functiongemma(tool_data)
    print(f"   Tools → {len(tool_converted):,} examples")
    
    qa_converted = convert_qa_to_functiongemma(qa_data, sample_ratio=args.qa_ratio)
    print(f"   QA → {len(qa_converted):,} examples (sampled {args.qa_ratio*100:.0f}%)")
    
    # Merge all
    all_examples = cot_converted + tool_converted + qa_converted
    print(f"\n📊 Total examples: {len(all_examples):,}")
    
    # Shuffle
    import random
    random.seed(42)
    random.shuffle(all_examples)
    
    # Format for training if requested
    if args.format_text:
        print("\n🔄 Formatting with tokenizer...")
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/functiongemma-270m-it",
                max_seq_length=4096,
                load_in_16bit=True
            )
            all_examples = format_for_training(all_examples, tokenizer)
            print(f"   Formatted: {len(all_examples):,} examples")
        except ImportError:
            print("   Warning: unsloth not installed, saving messages format")
    
    # Save
    print(f"\n💾 Saving to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(all_examples, f, indent=2)
    
    # Save tools separately
    tools_output = args.output.replace(".json", "_tools.json")
    with open(tools_output, "w") as f:
        json.dump(JARVIS_TOOLS, f, indent=2)
    print(f"   Tools saved to {tools_output}")
    
    print("\n✅ Conversion complete!")
    print(f"\nDataset composition:")
    print(f"   CoT (with thinking): {len(cot_converted):,} ({len(cot_converted)/len(all_examples)*100:.1f}%)")
    print(f"   Tool calls: {len(tool_converted):,} ({len(tool_converted)/len(all_examples)*100:.1f}%)")
    print(f"   Simple QA: {len(qa_converted):,} ({len(qa_converted)/len(all_examples)*100:.1f}%)")


if __name__ == "__main__":
    main()
