"""
Tool parser for OpenAPI, JSON Schema, and Python modules.

Extracts tool definitions from various documentation formats.
"""

import ast
import inspect
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import logging

from .tool_schemas import Tool, Parameter

logger = logging.getLogger(__name__)


class ToolParser:
    """Parse tool definitions from various formats."""
    
    # Type mappings
    OPENAPI_TYPE_MAP = {
        "string": "string",
        "number": "number",
        "integer": "integer",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }
    
    PYTHON_TYPE_MAP = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "List": "array",
        "Dict": "object",
        "Any": "any",
    }
    
    def parse_openapi(self, spec_path: str) -> List[Tool]:
        """
        Parse OpenAPI 3.0 specification.
        
        Extracts tools from paths with requestBody or parameters.
        
        Args:
            spec_path: Path to OpenAPI JSON/YAML file
            
        Returns:
            List of Tool objects
        """
        path = Path(spec_path)
        
        if path.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                with open(path) as f:
                    spec = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML files: pip install pyyaml")
        else:
            with open(path) as f:
                spec = json.load(f)
        
        tools = []
        paths = spec.get("paths", {})
        
        for endpoint, methods in paths.items():
            for method, details in methods.items():
                if method in ["get", "post", "put", "patch", "delete"]:
                    tool = self._parse_openapi_operation(
                        endpoint, method, details, spec
                    )
                    if tool:
                        tools.append(tool)
        
        return tools
    
    def _parse_openapi_operation(
        self, 
        endpoint: str, 
        method: str, 
        details: Dict, 
        spec: Dict
    ) -> Optional[Tool]:
        """Parse a single OpenAPI operation into a Tool."""
        operation_id = details.get("operationId", f"{method}_{endpoint.replace('/', '_')}")
        description = details.get("summary", details.get("description", ""))
        
        parameters = []
        
        # Parse path/query parameters
        for param in details.get("parameters", []):
            param_def = self._resolve_ref(param, spec) if "$ref" in param else param
            parameters.append(Parameter(
                name=param_def["name"],
                type=self.OPENAPI_TYPE_MAP.get(
                    param_def.get("schema", {}).get("type", "string"), "string"
                ),
                description=param_def.get("description", ""),
                required=param_def.get("required", False),
                default=param_def.get("schema", {}).get("default"),
                enum=param_def.get("schema", {}).get("enum"),
            ))
        
        # Parse requestBody
        request_body = details.get("requestBody", {})
        if request_body:
            content = request_body.get("content", {})
            json_content = content.get("application/json", {})
            schema = json_content.get("schema", {})
            
            if "$ref" in schema:
                schema = self._resolve_ref(schema, spec)
            
            body_params = self._parse_schema_properties(schema, spec)
            parameters.extend(body_params)
        
        # Parse response for returns
        responses = details.get("responses", {})
        returns = {"type": "object", "description": "API response"}
        if "200" in responses:
            resp = responses["200"]
            returns["description"] = resp.get("description", "Success response")
        
        # Determine complexity
        complexity = "simple"
        if len(parameters) > 5:
            complexity = "medium"
        if len(parameters) > 10:
            complexity = "complex"
        
        return Tool(
            tool_id=operation_id,
            name=operation_id,
            description=description,
            parameters=parameters,
            returns=returns,
            examples=[],
            category=details.get("tags", ["general"])[0] if details.get("tags") else "general",
            complexity=complexity,
        )
    
    def _resolve_ref(self, ref_obj: Dict, spec: Dict) -> Dict:
        """Resolve $ref to actual schema."""
        ref_path = ref_obj.get("$ref", "")
        if not ref_path.startswith("#/"):
            return ref_obj
        
        parts = ref_path[2:].split("/")
        result = spec
        for part in parts:
            result = result.get(part, {})
        return result
    
    def _parse_schema_properties(self, schema: Dict, spec: Dict) -> List[Parameter]:
        """Parse JSON Schema properties into Parameters."""
        parameters = []
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        for name, prop in properties.items():
            if "$ref" in prop:
                prop = self._resolve_ref(prop, spec)
            
            parameters.append(Parameter(
                name=name,
                type=self.OPENAPI_TYPE_MAP.get(prop.get("type", "string"), "string"),
                description=prop.get("description", ""),
                required=name in required,
                default=prop.get("default"),
                enum=prop.get("enum"),
            ))
        
        return parameters
    
    def parse_json_schema(self, schema_path: str) -> List[Tool]:
        """
        Parse tools from a JSON Schema file.
        
        Expects format:
        {
            "tools": [
                {"name": "...", "description": "...", "parameters": {...}}
            ]
        }
        
        Args:
            schema_path: Path to JSON file
            
        Returns:
            List of Tool objects
        """
        with open(schema_path) as f:
            data = json.load(f)
        
        tools = []
        tool_defs = data if isinstance(data, list) else data.get("tools", [data])
        
        for i, tool_def in enumerate(tool_defs):
            tool = self._parse_json_tool(tool_def, i)
            tools.append(tool)
        
        return tools
    
    def _parse_json_tool(self, tool_def: Dict, index: int) -> Tool:
        """Parse a single JSON tool definition."""
        name = tool_def.get("name", f"tool_{index}")
        tool_id = tool_def.get("tool_id", name.lower().replace(" ", "_"))
        
        parameters = []
        params_def = tool_def.get("parameters", {})
        
        # Handle both list and object formats
        if isinstance(params_def, list):
            for p in params_def:
                parameters.append(Parameter.from_dict(p))
        elif isinstance(params_def, dict):
            # JSON Schema format
            properties = params_def.get("properties", {})
            required = params_def.get("required", [])
            
            for pname, pdef in properties.items():
                parameters.append(Parameter(
                    name=pname,
                    type=pdef.get("type", "string"),
                    description=pdef.get("description", ""),
                    required=pname in required,
                    default=pdef.get("default"),
                    enum=pdef.get("enum"),
                ))
        
        return Tool(
            tool_id=tool_id,
            name=name,
            description=tool_def.get("description", ""),
            parameters=parameters,
            returns=tool_def.get("returns", {"type": "any", "description": ""}),
            examples=tool_def.get("examples", []),
            category=tool_def.get("category", "general"),
            complexity=tool_def.get("complexity", "simple"),
        )
    
    def parse_python_module(self, module_path: str) -> List[Tool]:
        """
        Parse tools from Python module with docstrings.
        
        Extracts functions with Google-style docstrings.
        
        Args:
            module_path: Path to Python file
            
        Returns:
            List of Tool objects
        """
        with open(module_path) as f:
            source = f.read()
        
        tree = ast.parse(source)
        tools = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private functions
                if node.name.startswith("_"):
                    continue
                
                tool = self._parse_python_function(node, source)
                if tool:
                    tools.append(tool)
        
        return tools
    
    def _parse_python_function(self, node: ast.FunctionDef, source: str) -> Optional[Tool]:
        """Parse a Python function into a Tool."""
        docstring = ast.get_docstring(node) or ""
        
        # Parse docstring for description and params
        description, param_docs = self._parse_docstring(docstring)
        
        # Parse function arguments
        parameters = []
        defaults = {
            arg.arg: ast.literal_eval(ast.unparse(default)) 
            for arg, default in zip(
                reversed(node.args.args), 
                reversed(node.args.defaults)
            )
        } if node.args.defaults else {}
        
        for arg in node.args.args:
            if arg.arg == "self":
                continue
            
            # Get type annotation
            arg_type = "any"
            if arg.annotation:
                ann = ast.unparse(arg.annotation)
                # Extract base type
                base_type = re.match(r"(\w+)", ann)
                if base_type:
                    arg_type = self.PYTHON_TYPE_MAP.get(base_type.group(1), "any")
            
            # Get description from docstring
            arg_desc = param_docs.get(arg.arg, "")
            
            parameters.append(Parameter(
                name=arg.arg,
                type=arg_type,
                description=arg_desc,
                required=arg.arg not in defaults,
                default=defaults.get(arg.arg),
            ))
        
        # Determine complexity
        complexity = "simple"
        if len(parameters) > 3:
            complexity = "medium"
        if len(parameters) > 6:
            complexity = "complex"
        
        return Tool(
            tool_id=node.name,
            name=node.name,
            description=description or f"Function {node.name}",
            parameters=parameters,
            returns={"type": "any", "description": ""},
            examples=[],
            category="python",
            complexity=complexity,
        )
    
    def _parse_docstring(self, docstring: str) -> tuple[str, Dict[str, str]]:
        """Parse Google-style docstring into description and param docs."""
        if not docstring:
            return "", {}
        
        lines = docstring.strip().split("\n")
        description_lines = []
        param_docs = {}
        current_section = "description"
        current_param = None
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.lower().startswith("args:") or stripped.lower().startswith("parameters:"):
                current_section = "args"
                continue
            elif stripped.lower().startswith("returns:"):
                current_section = "returns"
                continue
            elif stripped.lower().startswith("raises:"):
                current_section = "raises"
                continue
            
            if current_section == "description":
                description_lines.append(stripped)
            elif current_section == "args":
                # Parse param: description format
                match = re.match(r"(\w+)(?:\s*\([^)]*\))?:\s*(.*)", stripped)
                if match:
                    current_param = match.group(1)
                    param_docs[current_param] = match.group(2)
                elif current_param and stripped:
                    param_docs[current_param] += " " + stripped
        
        return " ".join(description_lines).strip(), param_docs
    
    def parse_functions(self, functions: List[Callable]) -> List[Tool]:
        """
        Parse tools from Python callable objects.
        
        Args:
            functions: List of Python functions
            
        Returns:
            List of Tool objects
        """
        tools = []
        
        for func in functions:
            sig = inspect.signature(func)
            doc = inspect.getdoc(func) or ""
            description, param_docs = self._parse_docstring(doc)
            
            parameters = []
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                
                # Get type annotation
                arg_type = "any"
                if param.annotation != inspect.Parameter.empty:
                    type_name = getattr(param.annotation, "__name__", str(param.annotation))
                    arg_type = self.PYTHON_TYPE_MAP.get(type_name, "any")
                
                # Get default
                default = None
                required = True
                if param.default != inspect.Parameter.empty:
                    default = param.default
                    required = False
                
                parameters.append(Parameter(
                    name=name,
                    type=arg_type,
                    description=param_docs.get(name, ""),
                    required=required,
                    default=default,
                ))
            
            tools.append(Tool(
                tool_id=func.__name__,
                name=func.__name__,
                description=description or f"Function {func.__name__}",
                parameters=parameters,
                returns={"type": "any", "description": ""},
                examples=[],
                category="python",
                complexity="simple" if len(parameters) <= 3 else "medium",
            ))
        
        return tools
    
    def validate_tool(self, tool: Tool) -> List[str]:
        """
        Validate a tool definition.
        
        Args:
            tool: Tool to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not tool.tool_id:
            errors.append("tool_id is required")
        
        if not tool.name:
            errors.append("name is required")
        
        if not tool.description:
            errors.append("description is required")
        
        for i, param in enumerate(tool.parameters):
            if not param.name:
                errors.append(f"Parameter {i}: name is required")
            
            if param.type not in ["string", "number", "integer", "boolean", "array", "object", "any"]:
                errors.append(f"Parameter {param.name}: invalid type '{param.type}'")
        
        # Check for duplicate parameter names
        names = [p.name for p in tool.parameters]
        if len(names) != len(set(names)):
            errors.append("Duplicate parameter names found")
        
        return errors
    
    def validate_tools(self, tools: List[Tool]) -> Dict[str, List[str]]:
        """
        Validate multiple tools.
        
        Args:
            tools: List of tools to validate
            
        Returns:
            Dict mapping tool_id to list of errors (only includes tools with errors)
        """
        results = {}
        for tool in tools:
            errors = self.validate_tool(tool)
            if errors:
                results[tool.tool_id] = errors
        return results
