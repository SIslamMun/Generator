"""
Generate MCP tool definitions from domain/topic using LLM.

Creates plausible MCP tool definitions when no existing server is available,
using domain knowledge and documentation as input.
"""

import json
import json5
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..clients import get_client

console = Console()
logger = logging.getLogger(__name__)


class MCPGenerator:
    """Generate MCP tool definitions from domain/topic."""
    
    def __init__(self, llm_config: Dict[str, Any], prompts: Dict[str, str]):
        """
        Initialize MCP generator.
        
        Args:
            llm_config: LLM configuration with provider and settings
            prompts: Prompt templates dict (loaded from configs/prompts/tool_prompts.yaml)
        """
        self.prompts = prompts
        provider = llm_config.pop("provider", "ollama")
        self.llm = get_client(provider, llm_config)
        self.provider = provider
    
    def _get_prompt(self, key: str) -> str:
        """Get prompt template, raising error if not found."""
        if key not in self.prompts:
            raise ValueError(
                f"Missing prompt template '{key}'. "
                f"Add it to configs/prompts/tool_prompts.yaml"
            )
        return self.prompts[key]
    
    def load_domain_context(
        self,
        topic: str,
        docs_path: Optional[str] = None,
        max_samples: int = 50
    ) -> str:
        """
        Load domain context from topic and optional documentation.
        
        Args:
            topic: Domain/topic description
            docs_path: Optional path to documentation/QA data
            max_samples: Maximum samples to load
            
        Returns:
            Rich context string for LLM
        """
        context = f"Topic: {topic}\n\n"
        
        if not docs_path:
            return context
        
        docs_file = Path(docs_path)
        if not docs_file.exists():
            logger.warning(f"Documentation file not found: {docs_path}")
            return context
        
        console.print(f"[cyan]📚 Loading domain knowledge from {docs_file.name}...[/cyan]")
        
        try:
            # Load data
            with open(docs_file, 'r') as f:
                if docs_file.suffix == '.json':
                    data = json.load(f)
                elif docs_file.suffix == '.jsonl':
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    logger.warning(f"Unsupported format: {docs_file.suffix}")
                    return context
            
            # Extract relevant samples
            samples = data[:max_samples] if isinstance(data, list) else [data]
            
            context += "Domain Knowledge Samples:\n"
            for i, item in enumerate(samples[:10], 1):
                if isinstance(item, dict):
                    # Handle QA pairs
                    if 'question' in item and 'answer' in item:
                        context += f"\n{i}. Q: {item['question'][:200]}\n"
                        context += f"   A: {item['answer'][:200]}\n"
                    # Handle chunks
                    elif 'chunk_text' in item:
                        context += f"\n{i}. {item['chunk_text'][:300]}\n"
                    # Handle instruction/response
                    elif 'instruction' in item:
                        context += f"\n{i}. {item['instruction'][:200]}\n"
            
            console.print(f"[green]✓ Loaded {len(samples)} knowledge samples[/green]")
            
        except Exception as e:
            logger.error(f"Failed to load documentation: {e}")
        
        return context
    
    def generate_tools(
        self,
        topic: str,
        context: str,
        n_tools: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Generate MCP tool definitions using LLM.
        
        Args:
            topic: Domain/topic description
            context: Rich domain context
            n_tools: Number of tools to generate
            
        Returns:
            List of tool definitions
        """
        prompt_template = self._get_prompt("mcp_from_topic_generation")
        prompt = prompt_template.format(context=context, n_tools=n_tools)

        console.print("[cyan]🤖 Generating MCP tool definitions with LLM...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Generating {n_tools} tools...", total=None)
            
            response = self.llm.generate(prompt, temperature=0.7, max_tokens=8192)
            
            progress.update(task, completed=True)
        
        # Parse response
        try:
            # Remove markdown if present
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            
            # Try json5 first (more lenient)
            try:
                tools = json5.loads(cleaned)
            except:
                tools = json.loads(cleaned)
            
            if not isinstance(tools, list):
                logger.warning("Expected list, got dict. Wrapping...")
                tools = [tools]
            
            console.print(f"[green]✓ Generated {len(tools)} tool definitions[/green]")
            return tools
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response preview: {response[:500]}")
            return []
    
    def refine_and_validate(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Refine and validate generated tools.
        
        Args:
            tools: Raw tool definitions
            
        Returns:
            Cleaned and validated tools
        """
        console.print("[cyan]🔍 Refining and validating tools...[/cyan]")
        
        refined = []
        for i, tool in enumerate(tools):
            try:
                # Ensure required fields
                if 'name' not in tool or 'description' not in tool:
                    logger.warning(f"Tool {i}: Missing name/description, skipping")
                    continue
                
                # Normalize structure
                refined_tool = {
                    "name": tool['name'],
                    "description": tool['description'],
                    "parameters": tool.get('parameters', []),
                    "returns": tool.get('returns', {
                        "type": "object",
                        "description": "Operation result"
                    }),
                    "examples": tool.get('examples', []),
                }
                
                # Add optional fields
                if 'category' in tool:
                    refined_tool['category'] = tool['category']
                
                # Validate parameters
                valid = True
                for param in refined_tool['parameters']:
                    if 'name' not in param or 'type' not in param:
                        logger.warning(f"{tool['name']}: Invalid parameter, skipping tool")
                        valid = False
                        break
                    # Ensure required field exists
                    if 'required' not in param:
                        param['required'] = True
                
                if valid:
                    refined.append(refined_tool)
                    console.print(
                        f"[green]  ✓ {tool['name']}: "
                        f"{len(refined_tool['parameters'])} params[/green]"
                    )
            
            except Exception as e:
                logger.warning(f"Tool {i}: Validation error - {e}")
        
        console.print(f"[green]✓ Validated {len(refined)}/{len(tools)} tools[/green]")
        return refined
    
    def generate_mcp_from_topic(
        self,
        topic: str,
        n_tools: int = 20,
        docs_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Complete workflow: generate MCP tools from topic.
        
        Args:
            topic: Domain/topic description
            n_tools: Number of tools to generate
            docs_path: Optional path to documentation
            
        Returns:
            List of validated tool definitions
        """
        # Load domain context
        context = self.load_domain_context(topic, docs_path)
        
        # Generate tools
        tools = self.generate_tools(topic, context, n_tools)
        
        if not tools:
            logger.error("No tools generated")
            return []
        
        # Refine and validate
        refined_tools = self.refine_and_validate(tools)
        
        return refined_tools
