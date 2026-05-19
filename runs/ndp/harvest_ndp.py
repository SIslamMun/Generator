"""Harvest real NDP state into a cache used to ground the dataset builder.

Hits the live NDP API (the same endpoints ndp_mcp wraps) and records:
  - every organization slug, per server (global / pre_ckan / local)
  - a wide pool of real datasets (id, name, title, owner_org, resource formats)
  - which owner_orgs and resource formats actually return results

Output: runs/ndp/data/ndp_harvest.json  — the single source of real values
the deterministic dataset builder draws from. Re-run to refresh.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

BASE = "http://155.101.6.191:8003"
HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "ndp_harvest.json"

SERVERS = ["global", "pre_ckan", "local"]

# Broad topic sweep — picks up datasets across the whole catalog.
TOPICS = [
    "climate", "fire", "wildfire", "water", "air quality", "satellite",
    "temperature", "ocean", "soil", "agriculture", "forest", "health",
    "genomics", "energy", "weather", "flood", "drought", "vegetation",
    "wildlife", "carbon", "emissions", "precipitation", "snow", "ice",
    "land use", "crop", "dairy", "ecology", "biodiversity", "hydrology",
    "elevation", "landslide", "treatment", "connectivity", "nasa", "usgs",
]


def _get(path: str, params: list[tuple]) -> tuple[int, object]:
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{BASE}{path}?{qs}" if qs else f"{BASE}{path}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()[:200]
        except Exception as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            return -1, f"{type(e).__name__}: {e}"
    return -1, "unreachable"


def _post(path: str, params: list[tuple], body: dict) -> tuple[int, object]:
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{BASE}{path}?{qs}" if qs else f"{BASE}{path}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()[:200]
        except Exception as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            return -1, f"{type(e).__name__}: {e}"
    return -1, "unreachable"


def main() -> None:
    harvest: dict = {"base": BASE, "servers": {}, "datasets": {}, "topics": {}}

    # ── organizations, per server ────────────────────────────────────
    for server in SERVERS:
        code, body = _get("/organization", [("server", server)])
        orgs = body if (code == 200 and isinstance(body, list)) else []
        harvest["servers"][server] = {"http": code, "n_orgs": len(orgs), "orgs": orgs}
        print(f"[orgs] {server:9s} -> HTTP {code}  n={len(orgs)}")

    # ── dataset pool: topic sweep on the global server ───────────────
    pool: dict[str, dict] = {}          # id -> dataset record
    topic_hits: dict[str, int] = {}
    for topic in TOPICS:
        code, body = _get("/search", [("server", "global"), ("terms", topic)])
        if code == 200 and isinstance(body, list):
            topic_hits[topic] = len(body)
            for d in body:
                if isinstance(d, dict) and d.get("id"):
                    pool[d["id"]] = d
            print(f"[search] {topic:18s} -> {len(body):4d} datasets")
        else:
            topic_hits[topic] = 0
            print(f"[search] {topic:18s} -> HTTP {code} (skipped)")
        time.sleep(0.2)

    harvest["topics"] = topic_hits
    harvest["datasets"]["global"] = list(pool.values())
    print(f"[pool] global: {len(pool)} unique datasets")

    # ── owner_org / resource_format frequency from the pool ──────────
    org_freq: Counter = Counter()
    fmt_freq: Counter = Counter()
    for d in pool.values():
        if d.get("owner_org"):
            org_freq[d["owner_org"]] += 1
        for res in d.get("resources") or []:
            f = (res.get("format") or "").strip()
            if f:
                fmt_freq[f] += 1
    harvest["owner_org_freq"] = dict(org_freq.most_common())
    harvest["resource_format_freq"] = dict(fmt_freq.most_common())
    print(f"[pool] owner_orgs with data: {len(org_freq)}  formats: {dict(fmt_freq.most_common(8))}")

    # ── verify advanced search works for the top owner_orgs ──────────
    adv_ok = {}
    for org, _ in org_freq.most_common(15):
        code, body = _post("/search", [], {"owner_org": org, "server": "global"})
        n = len(body) if (code == 200 and isinstance(body, list)) else 0
        adv_ok[org] = {"http": code, "n": n}
        print(f"[adv] owner_org={org:35s} -> HTTP {code}  n={n}")
        time.sleep(0.2)
    harvest["advanced_owner_org"] = adv_ok

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(harvest, indent=1))
    print(f"\n[done] wrote {OUT}")


if __name__ == "__main__":
    main()
