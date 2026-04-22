"""Minimal stdio MCP client for the real Jarvis MCP server.

Speaks JSON-RPC over stdin/stdout to the FastMCP server at
`jarvis-env/clio-kit-mcp-servers/jarvis/src/server.py`. Exposes:
    - `list_tools()` → [{name, description, input_schema}]
    - `call_tool(name, arguments)` → parsed JSON result (or str)

Pages through `tools/list` cursors so all 29 tools are returned even with
FastMCP's default `list_page_size=10`.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from queue import Queue, Empty
from typing import Any


class MCPError(RuntimeError):
    pass


class JarvisMCP:
    def __init__(
        self,
        server_cmd: list[str] | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        startup_timeout: float = 15.0,
    ):
        self.server_cmd = server_cmd or [
            "/home/shazzadul/Illinois_Tech/Spring26/RA/clio-kit/clio-kit-mcp-servers/jarvis/.venv/bin/jarvis-mcp",
        ]
        self.cwd = cwd or "/home/shazzadul/Illinois_Tech/Spring26/RA/clio-kit/clio-kit-mcp-servers/jarvis"
        self.env = {**os.environ, **(env or {})}
        self.startup_timeout = startup_timeout
        self._proc: subprocess.Popen | None = None
        self._next_id = 1
        self._reader_thread: threading.Thread | None = None
        self._responses: Queue = Queue()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def start(self):
        self._proc = subprocess.Popen(
            self.server_cmd,
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()
        self._handshake()

    def close(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None

    def _reader(self):
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._responses.put(msg)

    def _send(self, payload: dict):
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict | None = None, timeout: float = 60.0) -> Any:
        rid = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        import time as _t
        deadline = _t.time() + timeout
        while _t.time() < deadline:
            try:
                msg = self._responses.get(timeout=max(0.1, deadline - _t.time()))
            except Empty:
                continue
            if msg.get("id") != rid:
                continue
            if "error" in msg:
                raise MCPError(msg["error"])
            return msg.get("result")
        raise MCPError(f"timeout waiting for `{method}`")

    def _notify(self, method: str, params: dict | None = None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _handshake(self):
        self._request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "generator-inference", "version": "0.1"},
            },
            timeout=self.startup_timeout,
        )
        self._notify("notifications/initialized")

    def list_tools(self) -> list[dict]:
        """Paginated so the full 29-tool catalog is returned."""
        tools: list[dict] = []
        cursor: str | None = None
        for _ in range(20):
            params = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params)
            for t in result.get("tools", []):
                tools.append(
                    {
                        "name": t.get("name"),
                        "description": t.get("description", ""),
                        "input_schema": t.get("inputSchema") or t.get("input_schema") or {},
                    }
                )
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict | None = None, timeout: float = 120.0) -> str:
        """Invoke a tool. Returns the raw content string (JSON when applicable)."""
        try:
            result = self._request(
                "tools/call",
                {"name": name, "arguments": arguments or {}},
                timeout=timeout,
            )
        except MCPError as e:
            return json.dumps({"error": str(e)})
        if isinstance(result, dict):
            if result.get("isError"):
                content = result.get("content", [])
                if content and isinstance(content, list):
                    first = content[0]
                    if isinstance(first, dict) and "text" in first:
                        return json.dumps({"error": first["text"]})
                return json.dumps({"error": str(result)})
            content = result.get("content", [])
            if content and isinstance(content, list):
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
            if "structuredContent" in result:
                return json.dumps(result["structuredContent"])
        return json.dumps(result)
