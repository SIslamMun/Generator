"""Build a LanceDB index from a directory of source/markdown files.

Schema: id, source, source_file, content, type
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from rich.console import Console

console = Console()


def _is_text_file(path: Path) -> bool:
    if path.suffix.lower() in {".py", ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".cfg", ".ini"}:
        return True
    return False


def _file_type(path: Path) -> str:
    if path.suffix.lower() in {".py"}:                       return "code"
    if path.suffix.lower() in {".md", ".rst", ".txt"}:       return "text"
    return "config"


def _chunk_text(text: str, max_chars: int = 1500, overlap: int = 100) -> list[str]:
    """Cheap newline-aware chunker."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        # try to break on a newline near the end
        if end < len(text):
            nl = text.rfind("\n", i + max_chars - 200, end)
            if nl > i + max_chars - 400:
                end = nl
        chunks.append(text[i:end])
        i = max(i + 1, end - overlap)
    return chunks


def ingest(
    source_dir: str | Path,
    output_db:  str | Path,
    *,
    table:       str = "code_chunks",
    max_chars:   int = 1500,
    skip_dirs:   tuple = (".git", ".venv", "venv", "__pycache__", "node_modules",
                          ".pytest_cache", "build", "dist", ".idea"),
    max_files:   int | None = None,
):
    """Walk source_dir, chunk each text file, write to LanceDB."""
    import lancedb
    import pyarrow as pa

    source_dir = Path(source_dir).resolve()
    output_db  = Path(output_db).resolve()
    output_db.mkdir(parents=True, exist_ok=True)

    rows = []
    n_files = 0
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for fn in files:
            p = Path(root) / fn
            if not _is_text_file(p):
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            text = text.strip()
            if not text:
                continue
            n_files += 1
            if max_files and n_files > max_files:
                break
            rel = str(p.relative_to(source_dir))
            ftype = _file_type(p)
            for j, chunk in enumerate(_chunk_text(text, max_chars=max_chars)):
                cid = hashlib.sha1(f"{rel}:{j}".encode()).hexdigest()[:12]
                rows.append({
                    "id":          f"{p.name}:{j}:{cid}",
                    "source":      str(p),
                    "source_file": rel,
                    "content":     chunk,
                    "type":        ftype,
                })
        if max_files and n_files > max_files:
            break

    if not rows:
        raise RuntimeError(f"no rows produced from {source_dir}")

    db = lancedb.connect(str(output_db))
    if table in db.table_names():
        db.drop_table(table)
    db.create_table(table, data=rows)
    console.print(f"[ingest] wrote {len(rows)} chunks from {n_files} files → {output_db} (table: {table})")
    return {"rows": len(rows), "files": n_files, "db": str(output_db), "table": table}
