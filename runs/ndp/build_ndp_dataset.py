"""Deterministically build a clean NDP tool-calling dataset.

Goal: a dataset where EVERY tool call is schema-perfect and EVERY tool result
is real JSON captured from the live National Data Platform — so the fine-tuned
model learns flawless JSON-in / JSON-out tool calling against the 3 ndp-mcp
tools (list_organizations, search_datasets, get_dataset_details).

How it stays correct:
  - tool calls are CONSTRUCTED from a coverage matrix, never written by an LLM,
    so names / params / enums / types are valid by construction;
  - list_organizations and get_dataset_details results are built offline from
    the harvest cache (runs/ndp/data/ndp_harvest.json) — exact real records;
  - search_datasets results are fetched live (retry + skip on NDP 5xx);
  - every result mirrors the ndp_mcp server's exact return shape;
  - long results are trimmed so training sequences stay within 4096 tokens.

Output: runs/ndp/data/ndp_tool_examples_curated.json — consumed unchanged by
finetuned_unsloth/models/nemotron_nano_4b/prepare_data.py (--types tool).

Run on any node that can reach NDP (login node is fine): python3.11 build_ndp_dataset.py
"""
from __future__ import annotations

import html
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://155.101.6.191:8003"
HERE = Path(__file__).resolve().parent
HARVEST = HERE / "data" / "ndp_harvest.json"
OUT = HERE / "data" / "ndp_tool_examples_curated.json"
SEED = 3407
RESULT_MAX_DATASETS = 5      # trim search results to keep sequences short
RESULT_MAX_RESOURCES = 3     # trim each dataset's resources
NOTES_MAX = 280              # trim long dataset notes

rng = random.Random(SEED)

# ─────────────────────────── live NDP (search only) ─────────────────

def _get(path: str, params: list[tuple]) -> tuple[int, object]:
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{BASE}{path}?{qs}" if qs else f"{BASE}{path}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return e.code, e.read().decode()[:160]
        except Exception as e:
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return -1, f"{type(e).__name__}: {e}"
    return -1, "unreachable"


def _post(path: str, body: dict) -> tuple[int, object]:
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return e.code, e.read().decode()[:160]
        except Exception as e:
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            return -1, f"{type(e).__name__}: {e}"
    return -1, "unreachable"


# ─────────────────────────── result shaping ─────────────────────────

def _clean_text(s: str) -> str:
    """Strip HTML tags / entities and collapse whitespace — NDP dataset notes
    are often raw HTML, which must never leak into a tool result or answer."""
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = html.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


def _trim_dataset(d: dict) -> dict:
    """Shrink a dataset record to the fields the model needs, short enough
    to keep training sequences inside 4096 tokens."""
    res = []
    for r in (d.get("resources") or [])[:RESULT_MAX_RESOURCES]:
        res.append({k: r.get(k) for k in ("name", "format", "url") if r.get(k)})
    notes = _clean_text(d.get("notes") or "")
    if len(notes) > NOTES_MAX:
        notes = notes[:NOTES_MAX].rstrip() + "…"
    return {
        "id": d.get("id"),
        "name": d.get("name"),
        "title": d.get("title"),
        "owner_org": d.get("owner_org"),
        "notes": notes,
        "resources": res,
    }


def wrap_list_orgs(orgs: list[str], server: str, name_filter) -> dict:
    """Mirror ndp_mcp.list_organizations' return shape."""
    return {
        "organizations": orgs,
        "count": len(orgs),
        "server": server,
        "name_filter": name_filter,
        "_meta": {"tool": "list_organizations", "status": "success"},
    }


def wrap_search(datasets: list[dict], server: str, params: dict) -> dict:
    """Mirror ndp_mcp.search_datasets' return shape (with our small trim)."""
    total = len(datasets)
    limited = datasets[:RESULT_MAX_DATASETS]
    was_limited = total > len(limited)
    trimmed = [_trim_dataset(d) for d in limited]
    return {
        "datasets": trimmed,
        "count": len(trimmed),
        "total_found": f"{len(trimmed)} of {total}" if was_limited else total,
        "server": server,
        "search_parameters": params,
        "_meta": {"tool": "search_datasets", "status": "success"},
    }


def wrap_details(d: dict, ident: str, itype: str, server: str) -> dict:
    """Mirror ndp_mcp.get_dataset_details' return shape."""
    td = _trim_dataset(d)
    return {
        "dataset": td,
        "identifier_used": {"type": itype, "value": ident},
        "server": server,
        "resource_count": len(td["resources"]),
        "_meta": {"tool": "get_dataset_details", "status": "success"},
    }


# ─────────────────────────── query templates ────────────────────────

T_LIST_ALL = [
    "List all organizations on the National Data Platform.",
    "What organizations are available on NDP?",
    "Show me every organization that publishes data on NDP.",
    "Who are the data publishers on the National Data Platform?",
    "Give me the full list of NDP organizations.",
    "Which organizations contribute datasets to NDP?",
]
T_LIST_FILTER = [
    "Show me organizations with '{x}' in their name.",
    "Which NDP organizations match '{x}'?",
    "List organizations whose name contains {x}.",
    "Find all organizations related to {x} on the platform.",
    "I'm looking for {x} organizations — list them.",
    "Get the organizations that have {x} in their name.",
    "Are there any organizations named like {x} on NDP?",
]
T_SEARCH = [
    "Find datasets about {q}.",
    "Search NDP for {q} data.",
    "I need datasets related to {q}.",
    "What {q} datasets are available on the platform?",
    "Look up {q} datasets.",
    "Show me data on {q}.",
    "Are there datasets covering {q}?",
    "Help me discover {q} datasets.",
]
T_SEARCH_ORG = [
    "What datasets does {org} publish?",
    "Show me datasets owned by {org}.",
    "List the datasets from the organization {org}.",
    "Find all datasets belonging to {org}.",
    "I want to see {org}'s datasets.",
]
T_SEARCH_FMT = [
    "Find datasets that provide {fmt} files.",
    "Show me datasets available in {fmt} format.",
    "Which datasets have {fmt} resources?",
    "I need datasets with {fmt} data.",
    "Search for datasets distributed as {fmt}.",
]
T_SEARCH_TITLE = [
    "Find datasets whose title mentions {q}.",
    "Search for datasets with '{q}' in the title.",
    "Look up datasets titled around {q}.",
]
T_DETAILS_NAME = [
    "Give me the full details of the dataset '{name}'.",
    "Show me the metadata for the dataset named {name}.",
    "Tell me everything about the dataset {name}.",
    "I want all the details for '{name}'.",
    "Look up the dataset called {name}.",
    "What's in the dataset {name}?",
]
T_DETAILS_ID = [
    "Get the details for the dataset with id {id}.",
    "Show me the dataset whose id is {id}.",
    "Retrieve the metadata for dataset id {id}.",
    "Look up the dataset {id} by its id.",
]
T_CHAIN = [
    "Find datasets about {q} and then give me the full details of the top result.",
    "Search for {q} datasets and show me the metadata of the first match.",
    "I'm researching {q} — find datasets and then tell me more about the best one.",
    "Look up {q} data, then get the full details for the most relevant dataset.",
    "Search NDP for {q} and then describe the first dataset in full.",
    "Find {q} datasets and pull complete metadata for the top one.",
    "What {q} datasets are there? Then give me everything about the leading result.",
    "Discover {q} datasets and fetch the details of the most relevant match.",
]

# ─────────────────────────── think templates ────────────────────────

def think_list(name_filter, server):
    if name_filter:
        opts = [
            f"The user wants organizations matching '{name_filter}'. list_organizations "
            f"with name_filter does a substring match.",
            f"Filtering the organization list by the substring '{name_filter}' — "
            f"that is exactly what list_organizations' name_filter does.",
        ]
    else:
        opts = [
            "The user wants the full organization list; list_organizations returns it.",
            "This is a request for all organizations — call list_organizations with no filter.",
        ]
    t = rng.choice(opts)
    if server != "global":
        t += f" The {server} catalog is requested, so server='{server}'."
    return t


def think_search(kind, val, server):
    if kind == "terms":
        opts = [
            f"Keyword discovery — search_datasets with search_terms covers all fields.",
            f"A topical search; pass the keyword(s) to search_datasets via search_terms.",
        ]
    elif kind == "owner_org":
        opts = [
            f"The user wants one organization's datasets — use search_datasets' owner_org field.",
            f"Filtering datasets by publisher; search_datasets owner_org='{val}' does that.",
        ]
    elif kind == "resource_format":
        opts = [
            f"The user wants a specific file format — search_datasets' resource_format field.",
            f"Format-based discovery; pass resource_format to search_datasets.",
        ]
    else:
        opts = [
            f"Title-scoped discovery — search_datasets' dataset_title field.",
        ]
    t = rng.choice(opts)
    if server != "global":
        t += f" Server '{server}' was requested."
    return t


def think_details(ident, itype):
    return rng.choice([
        f"The user named a specific dataset; get_dataset_details with identifier_type='{itype}' "
        f"returns its full metadata.",
        f"This is a single-dataset lookup — call get_dataset_details on '{ident}' "
        f"as a {itype}.",
    ])


# ─────────────────────────── answer templates ───────────────────────

def ans_list(res):
    orgs = res["organizations"]
    n = res["count"]
    flt = f" matching '{res['name_filter']}'" if res.get("name_filter") else ""
    srv = res["server"]
    if n == 0:
        return f"No organizations{flt} were found on the {srv} NDP catalog."
    sample = ", ".join(orgs[:6])
    more = f", and {n - 6} more" if n > 6 else ""
    return (f"The {srv} NDP catalog has {n} organization(s){flt}: "
            f"{sample}{more}.")


def ans_search(res, scope):
    ds = res["datasets"]
    if not ds:
        return f"No datasets were found for {scope} on the {res['server']} NDP catalog."
    names = "; ".join(f"{d['title']} ({d['name']})" for d in ds[:3])
    tf = res["total_found"]
    tail = f" ({tf} total)" if isinstance(tf, str) else ""
    return (f"I found {res['count']} dataset(s) for {scope} on the {res['server']} "
            f"catalog{tail}. Top results: {names}.")


def ans_details(res):
    d = res["dataset"]
    notes = f" {d['notes']}" if d.get("notes") else ""
    return (f"'{d['title']}' ({d['name']}) is published by {d['owner_org']} and has "
            f"{res['resource_count']} resource(s).{notes}")


# ─────────────────────────── example assembly ───────────────────────

def make_example(query, steps, final_answer):
    """steps: list of (thought, tool, args, result_dict)."""
    rp = []
    for i, (thought, tool, args, result) in enumerate(steps, 1):
        rp.append({
            "step": i, "thought": thought, "tool": tool, "args": args,
            "status": "success", "expected_result": result,
        })
    return {
        "instruction": query,
        "solution": {
            "instruction": query,
            "reasoning_path": rp,
            "final_answer": final_answer,
            "execution_validated": True,
        },
    }


def fill(templates, count, **kw):
    """Pick `count` distinct query phrasings from a template bank."""
    pool = list(templates)
    rng.shuffle(pool)
    out = []
    for tpl in pool[:count]:
        out.append(tpl.format(**kw))
    return out


# ─────────────────────────── builders ───────────────────────────────

def substr_filters(orgs: list[str]) -> list[str]:
    """Real name substrings that match >=2 organizations (so name_filter
    examples reflect genuine NDP behaviour)."""
    cand = ["nasa", "climate", "california", "cal", "data", "forest", "fire",
            "health", "ai", "wildfire", "landscape", "library", "energy",
            "water", "research", "lab", "earth", "national", "science",
            "geospatial", "test", "metrics", "observatory"]
    keep = []
    for c in cand:
        hits = sum(1 for o in orgs if c in o.lower())
        if hits >= 2:
            keep.append(c)
    return keep


def build_list_org(harvest, examples):
    servers = harvest["servers"]
    # all-orgs, per server
    for server in ("global", "pre_ckan", "local"):
        orgs = servers[server]["orgs"]
        res = wrap_list_orgs(orgs, server, None)
        args = {} if server == "global" else {"server": server}
        for q in fill(T_LIST_ALL, 4):
            if server != "global":
                q = q.rstrip(".?") + f" on the {server} catalog."
            examples.append(make_example(
                q, [(think_list(None, server), "list_organizations", dict(args), res)],
                ans_list(res)))
    # name_filter, on global + a few on pre_ckan
    for server in ("global", "pre_ckan"):
        orgs = servers[server]["orgs"]
        filters = substr_filters(orgs)
        for x in filters:
            matched = [o for o in orgs if x in o.lower()]
            if not matched:
                continue
            res = wrap_list_orgs(matched, server, x)
            args = {"name_filter": x}
            if server != "global":
                args["server"] = server
            n_q = 5 if server == "global" else 2
            for q in fill(T_LIST_FILTER, n_q, x=x):
                if server != "global":
                    q = q.rstrip(".?") + f" on the {server} server."
                examples.append(make_example(
                    q, [(think_list(x, server), "list_organizations", dict(args), res)],
                    ans_list(res)))


def _search_live(args: dict) -> dict | None:
    """Execute one search_datasets call against live NDP; return wrapped result
    or None if NDP failed."""
    server = args.get("server", "global")
    if args.get("search_terms"):
        params = [("server", server)]
        for t in args["search_terms"]:
            params.append(("terms", t))
        for k in args.get("search_keys") or []:
            params.append(("keys", k))
        code, body = _get("/search", params)
    else:
        body_fields = {k: v for k, v in args.items() if k not in ("limit",)}
        code, body = _post("/search", body_fields)
    if code != 200 or not isinstance(body, list):
        return None
    sp = {k: args.get(k) for k in
          ("search_terms", "search_keys", "dataset_name", "dataset_title",
           "owner_org", "resource_format", "search_term", "filter_list", "limit")}
    return wrap_search(body, server, sp)


def build_search(harvest, examples, log):
    topics = [t for t, n in harvest["topics"].items() if n > 0]
    rng.shuffle(topics)
    # ── simple keyword search ────────────────────────────────────────
    for topic in topics:
        for server in (["global"] if rng.random() > 0.25 else ["global", "pre_ckan"]):
            args = {"search_terms": [topic]}
            if server != "global":
                args["server"] = server
            res = _search_live(args)
            if res is None:
                log.append(f"skip search terms={topic} server={server} (NDP 5xx)")
                continue
            scope = f"'{topic}'"
            for q in fill(T_SEARCH, 5, q=topic):
                examples.append(make_example(
                    q, [(think_search("terms", topic, server),
                         "search_datasets", dict(args), res)],
                    ans_search(res, scope)))
    # ── multi-term search ────────────────────────────────────────────
    multi = [["satellite", "fire"], ["air", "quality"], ["sea", "level"],
             ["forest", "carbon"], ["climate", "model"], ["snow", "water"]]
    for terms in multi:
        args = {"search_terms": terms}
        res = _search_live(args)
        if res is None:
            continue
        scope = " ".join(terms)
        for q in fill(T_SEARCH, 4, q=" ".join(terms)):
            examples.append(make_example(
                q, [(think_search("terms", scope, "global"),
                     "search_datasets", dict(args), res)],
                ans_search(res, f"'{scope}'")))
    # ── advanced: owner_org ──────────────────────────────────────────
    adv = harvest.get("advanced_owner_org", {})
    for org, info in adv.items():
        if info.get("http") != 200 or info.get("n", 0) == 0:
            continue
        args = {"owner_org": org}
        res = _search_live(args)
        if res is None:
            continue
        for q in fill(T_SEARCH_ORG, 4, org=org):
            examples.append(make_example(
                q, [(think_search("owner_org", org, "global"),
                     "search_datasets", dict(args), res)],
                ans_search(res, f"organization '{org}'")))
    # ── advanced: resource_format ────────────────────────────────────
    for fmt in ("CSV", "GeoTIFF", "GeoJSON", "TIFF", "ZIP", "HTML"):
        args = {"resource_format": fmt}
        res = _search_live(args)
        if res is None:
            continue
        for q in fill(T_SEARCH_FMT, 4, fmt=fmt):
            examples.append(make_example(
                q, [(think_search("resource_format", fmt, "global"),
                     "search_datasets", dict(args), res)],
                ans_search(res, f"{fmt} format")))
    # ── advanced: dataset_title ──────────────────────────────────────
    for title in ("climate", "fire", "elevation", "water quality", "vegetation"):
        args = {"dataset_title": title}
        res = _search_live(args)
        if res is None:
            continue
        for q in fill(T_SEARCH_TITLE, 3, q=title):
            examples.append(make_example(
                q, [(think_search("dataset_title", title, "global"),
                     "search_datasets", dict(args), res)],
                ans_search(res, f"title '{title}'")))
    # ── search + limit ───────────────────────────────────────────────
    for topic, lim in [("climate", 3), ("fire", 5), ("water", 10), ("forest", 3)]:
        args = {"search_terms": [topic], "limit": lim}
        res = _search_live(args)
        if res is None:
            continue
        for q in fill([f"Find the top {lim} datasets about {topic}.",
                       f"Show me {lim} {topic} datasets.",
                       f"Give me at most {lim} datasets on {topic}."], 3):
            examples.append(make_example(
                q, [(think_search("terms", topic, "global"),
                     "search_datasets", dict(args), res)],
                ans_search(res, f"'{topic}'")))


def build_details(harvest, examples):
    pool = [d for d in harvest["datasets"]["global"]
            if d.get("id") and d.get("name") and d.get("title")]
    rng.shuffle(pool)
    chosen = pool[:110]
    for i, d in enumerate(chosen):
        # ~1 in 3 lookups by id, the rest by name (name is the common case)
        if i % 3 == 0:
            itype, ident, tmpl = "id", d["id"], T_DETAILS_ID
        else:
            itype, ident, tmpl = "name", d["name"], T_DETAILS_NAME
        res = wrap_details(d, ident, itype, "global")
        args = {"dataset_identifier": ident, "identifier_type": itype}
        key = "id" if itype == "id" else "name"
        for q in fill(tmpl, 4, **{key: ident, "name": ident, "id": ident}):
            examples.append(make_example(
                q, [(think_details(ident, itype),
                     "get_dataset_details", dict(args), res)],
                ans_details(res)))


def build_chains(harvest, examples, log):
    """Two-step chains: search_datasets → get_dataset_details, where the detail
    identifier is taken from the search RESULT — never invented. This is the
    core JSON-input skill: read a JSON result, feed a value into the next call."""
    pool_by_id = {d["id"]: d for d in harvest["datasets"]["global"] if d.get("id")}

    # ── shape 1: keyword search → details of the top match ───────────
    topics = [t for t, n in harvest["topics"].items() if n > 0]
    rng.shuffle(topics)
    for topic in topics:
        args1 = {"search_terms": [topic], "limit": 5}
        res1 = _search_live(args1)
        if res1 is None or not res1["datasets"]:
            log.append(f"skip chain topic={topic}")
            continue
        top = res1["datasets"][0]
        full = pool_by_id.get(top["id"], top)
        args2 = {"dataset_identifier": top["name"], "identifier_type": "name"}
        res2 = wrap_details(full, top["name"], "name", "global")
        step1 = (think_search("terms", topic, "global"),
                 "search_datasets", args1, res1)
        step2 = (f"The search returned '{top['name']}' as the top match; "
                 f"fetching its full metadata with get_dataset_details.",
                 "get_dataset_details", args2, res2)
        final = f"The most relevant {topic} dataset: " + ans_details(res2)
        for q in fill(T_CHAIN, 6, q=topic):
            examples.append(make_example(q, [step1, step2], final))

    # ── shape 2: owner_org search → details of one of its datasets ───
    adv = harvest.get("advanced_owner_org", {})
    orgs = [o for o, i in adv.items() if i.get("http") == 200 and i.get("n", 0) > 0]
    rng.shuffle(orgs)
    for org in orgs:
        args1 = {"owner_org": org, "limit": 5}
        res1 = _search_live(args1)
        if res1 is None or not res1["datasets"]:
            continue
        top = res1["datasets"][0]
        full = pool_by_id.get(top["id"], top)
        args2 = {"dataset_identifier": top["name"], "identifier_type": "name"}
        res2 = wrap_details(full, top["name"], "name", "global")
        step1 = (think_search("owner_org", org, "global"),
                 "search_datasets", args1, res1)
        step2 = (f"Taking the first dataset '{top['name']}' published by {org} "
                 f"and retrieving its full details.",
                 "get_dataset_details", args2, res2)
        final = f"From {org}'s datasets — " + ans_details(res2)
        ctmpl = [f"Find datasets from {org} and give me details on the first one.",
                 f"Show me {org}'s datasets, then the full metadata of one.",
                 f"List datasets owned by {org} and detail the top result.",
                 f"What does {org} publish — and tell me more about one of them."]
        for q in fill(ctmpl, 4):
            examples.append(make_example(q, [step1, step2], final))


def build_search_extra(harvest, examples, log):
    """Coverage for the remaining advanced search fields so the model has seen
    every parameter: more owner_orgs, dataset_name, dataset_description,
    filter_list, and search_term (kept light — search_terms is preferred)."""
    pool = harvest["datasets"]["global"]

    # ── more owner_orgs beyond the pre-verified set ──────────────────
    seen = set(harvest.get("advanced_owner_org", {}))
    for org in list(harvest.get("owner_org_freq", {}))[:34]:
        if org in seen:
            continue
        res = _search_live({"owner_org": org})
        if res is None or res["count"] == 0:
            continue
        for q in fill(T_SEARCH_ORG, 4, org=org):
            examples.append(make_example(
                q, [(think_search("owner_org", org, "global"),
                     "search_datasets", {"owner_org": org}, res)],
                ans_search(res, f"organization '{org}'")))

    # ── dataset_name (exact / partial name match) ────────────────────
    named = [d for d in pool if d.get("name")]
    rng.shuffle(named)
    for d in named[:26]:
        name = d["name"]
        res = _search_live({"dataset_name": name})
        if res is None or res["count"] == 0:
            continue
        for q in fill(["Search for the dataset named {n}.",
                       "Find datasets matching the name '{n}'.",
                       "Look up datasets called {n}."], 3, n=name):
            examples.append(make_example(
                q, [("Matching on the dataset_name field with search_datasets.",
                     "search_datasets", {"dataset_name": name}, res)],
                ans_search(res, f"name '{name}'")))

    # ── dataset_description (free text in descriptions) ──────────────
    for desc in ("wildfire risk", "water quality", "land cover",
                 "carbon emissions", "snow cover", "air pollution"):
        res = _search_live({"dataset_description": desc})
        if res is None:
            continue
        for q in fill(["Find datasets described as being about {d}.",
                       "Search dataset descriptions for {d}.",
                       "Which datasets mention {d} in their description?"], 3, d=desc):
            examples.append(make_example(
                q, [("Searching the dataset_description field with search_datasets.",
                     "search_datasets", {"dataset_description": desc}, res)],
                ans_search(res, f"description '{desc}'")))

    # ── filter_list (key:value field filters) ────────────────────────
    fl_specs = [["resource_format:CSV"], ["resource_format:GeoTIFF"],
                ["resource_format:GeoJSON"], ["resource_format:ZIP"]]
    for org in list(harvest.get("owner_org_freq", {}))[:4]:
        fl_specs.append([f"owner_org:{org}"])
    for fl in fl_specs:
        res = _search_live({"filter_list": fl})
        if res is None or res["count"] == 0:
            continue
        label = fl[0]
        for q in fill(["Filter datasets where {f}.",
                       "Find datasets with the field filter {f}.",
                       "Apply the filter '{f}' to the dataset search."], 3, f=label):
            examples.append(make_example(
                q, [("Applying a key:value field filter via search_datasets' filter_list.",
                     "search_datasets", {"filter_list": list(fl)}, res)],
                ans_search(res, f"filter {label}")))

    # ── search_term (singular, comma-separated) — light coverage ─────
    for term in ("climate", "wildfire", "ocean temperature"):
        res = _search_live({"search_term": term})
        if res is None or res["count"] == 0:
            continue
        for q in fill(["Do a general search for {t}.",
                       "Search across all dataset fields for {t}."], 2, t=term):
            examples.append(make_example(
                q, [("Using search_datasets' general search_term field.",
                     "search_datasets", {"search_term": term}, res)],
                ans_search(res, f"'{term}'")))


def build_notool(examples):
    """~10%: inputs that need NO tool call — a direct answer or a clarifying
    question. Teaches the model when NOT to call a tool. Each item carries
    several phrasings of the question sharing one thought + answer."""
    items = [
        # ── facts about NDP (answer directly) ────────────────────────
        (["What is the National Data Platform?", "What is NDP?",
          "Tell me about the National Data Platform.", "Explain what NDP is."],
         "A general question about NDP itself, not a data request — answer directly.",
         "The National Data Platform (NDP) is an open platform for discovering and "
         "accessing scientific datasets contributed by many organizations. I can "
         "search its catalogs, list publishing organizations, and retrieve detailed "
         "metadata for any dataset."),
        (["What kind of data is on NDP?", "What sort of datasets does NDP have?",
          "What topics does the National Data Platform cover?"],
         "A general question about NDP's contents — answerable without a tool call.",
         "The National Data Platform hosts scientific datasets across many domains — "
         "climate, wildfire, water, air quality, satellite imagery, ecology and more — "
         "published by dozens of research organizations."),
        (["Which servers can I query?", "What catalogs does NDP have?",
          "What are the NDP server options?"],
         "A factual question about NDP servers — answerable without a tool call.",
         "The National Data Platform exposes three catalogs: 'global', 'pre_ckan', and "
         "'local'. Most datasets live on the 'global' catalog, which is the default."),
        (["What does identifier_type mean?", "What's the difference between id and name lookups?"],
         "A question about how the tools work — explain rather than call one.",
         "When fetching dataset details you can identify a dataset either by its 'id' "
         "(a UUID) or its 'name' (a slug). identifier_type tells the tool which one "
         "you're passing."),
        # ── capability / meta ────────────────────────────────────────
        (["What can you help me with?", "What can you do?", "How can you help me?",
          "What are you able to do?"],
         "A capability question — describe the tools rather than calling one.",
         "I can help you explore the National Data Platform: list the organizations "
         "that publish data, search datasets by keyword or by fields like owner or "
         "format, and retrieve full metadata for a specific dataset."),
        (["How many tools do you have?", "What tools can you use?",
          "Which functions are available to you?"],
         "A meta question about capabilities — answer directly.",
         "I have three National Data Platform tools: list_organizations (lists "
         "publishing organizations), search_datasets (finds datasets), and "
         "get_dataset_details (returns full metadata for one dataset)."),
        (["Can you delete a dataset for me?", "Please remove a dataset from NDP.",
          "Can you upload a new dataset?", "Can you edit a dataset's metadata?"],
         "NDP tools are read-only; create/update/delete are not supported — explain "
         "instead of calling a tool.",
         "I can only read from the National Data Platform — searching datasets, listing "
         "organizations, and fetching metadata. I can't create, modify, upload, or "
         "delete datasets."),
        (["Can you download a file for me?", "Download this dataset to my computer."],
         "Downloading files is outside the tools' scope — they return metadata only.",
         "I can't download files, but I can retrieve a dataset's metadata — including "
         "its resource URLs — so you can download the files yourself."),
        # ── greetings / closings ─────────────────────────────────────
        (["Hi there!", "Hello", "Hey", "Good morning!"],
         "A greeting — no tool needed.",
         "Hi! I can help you explore the National Data Platform. Tell me a topic, an "
         "organization, or a dataset name to get started."),
        (["Thanks, that's all I needed.", "Thank you!", "Thanks for the help."],
         "A closing remark — no tool needed.",
         "You're welcome! Ask any time you need to explore National Data Platform datasets."),
        (["Who are you?", "What are you?"],
         "An identity question — answer directly.",
         "I'm an assistant for the National Data Platform. I can search datasets, list "
         "organizations, and pull dataset metadata for you."),
        # ── vague queries → ask for clarification ────────────────────
        (["Find me some data.", "I need data.", "Search for datasets.",
          "Can you find datasets?"],
         "The request is too vague to choose search terms — ask for a topic before "
         "calling search_datasets.",
         "I'd be glad to help. What topic, keyword, or organization should I search "
         "the National Data Platform for?"),
        (["Get the dataset details.", "Show me the dataset.", "Look up a dataset.",
          "Tell me about the dataset."],
         "get_dataset_details needs a dataset_identifier and none was given — ask for it.",
         "Sure — which dataset? Give me its name or id and I'll pull the full metadata."),
        (["Show me the organization.", "Tell me about the org.",
          "Which organization?"],
         "Ambiguous — unclear which organization, or whether to list all of them.",
         "Could you clarify? I can list every organization on NDP, or filter them by a "
         "name keyword — which would you like?"),
        (["Search it.", "Look that up.", "Find it for me."],
         "No subject was given — there is nothing concrete to search for yet.",
         "Happy to — could you tell me what to search for? A topic, an organization, or "
         "a dataset name all work."),
        (["I'm doing research.", "I'm working on a project."],
         "A statement with no actionable request — invite a concrete query.",
         "Great — I can help with the data side. What subject are you researching? "
         "I can search the National Data Platform for relevant datasets."),
        (["What should I search for?", "What do you recommend I look at?"],
         "An open-ended question — no tool call is warranted without a topic.",
         "It depends on your interest. The National Data Platform is rich in climate, "
         "wildfire, water, and satellite data — tell me your subject and I'll search."),
        # ── more facts ───────────────────────────────────────────────
        (["Is the National Data Platform free to use?",
          "Does NDP cost money?", "Is NDP open access?"],
         "A general question about NDP access — answer directly.",
         "The National Data Platform is an open platform for discovering scientific "
         "datasets. I can search its public catalogs and return dataset metadata for you."),
        (["What's the default server?", "Which catalog is used by default?"],
         "A factual question about defaults — answerable without a tool call.",
         "Searches default to the 'global' catalog, which holds the bulk of NDP's "
         "datasets. You can also query the 'pre_ckan' or 'local' catalogs."),
        (["What does a dataset record contain?",
          "What metadata does a dataset have?"],
         "A question about NDP's data model — explain directly.",
         "Each NDP dataset has an id, a name, a title, an owning organization, "
         "descriptive notes, and a list of downloadable resources (files)."),
        (["What's the difference between search_datasets and get_dataset_details?"],
         "A question comparing two tools — explain rather than call one.",
         "search_datasets finds datasets matching a keyword or field filter and returns "
         "a list; get_dataset_details takes one dataset's id or name and returns its "
         "complete metadata."),
        # ── more capability / scope ──────────────────────────────────
        (["Can you visualize the data for me?", "Can you make a chart?",
          "Can you plot this dataset?"],
         "Visualization is outside the tools' scope — they return metadata only.",
         "I can't create charts, but I can find datasets and return their metadata and "
         "resource URLs so you can visualize the data with your own tools."),
        (["Can you analyze a dataset's contents?", "Can you summarize the actual data?"],
         "The tools return metadata, not file contents — clarify the limitation.",
         "I can retrieve a dataset's metadata and resource links, but I don't read the "
         "file contents themselves. I can point you to the resources to analyze."),
        (["Are you ChatGPT?", "What model are you?"],
         "An identity question — answer briefly without a tool call.",
         "I'm an assistant specialized in the National Data Platform — I search datasets, "
         "list organizations, and fetch dataset metadata."),
        # ── more vague / clarification ───────────────────────────────
        (["Help", "I need help", "Can you help me?"],
         "A bare request for help — invite a concrete query.",
         "Of course. I can search the National Data Platform for datasets, list its "
         "organizations, or pull metadata for a dataset. What would you like to do?"),
        (["Give me everything.", "Show me all of it.", "Just give me the data."],
         "Too broad to act on — ask the user to narrow the request.",
         "NDP is large, so let's narrow it down. What topic or organization are you "
         "interested in? I can then search for matching datasets."),
        (["Details please.", "More info.", "Tell me more."],
         "No subject in context — get_dataset_details needs an identifier first.",
         "Sure — about which dataset? Give me a dataset name or id and I'll fetch its "
         "full details."),
        (["organizations", "datasets", "search"],
         "A single bare word with no clear request — ask what the user wants.",
         "Could you say a bit more? I can list NDP organizations, search for datasets, "
         "or fetch a specific dataset's details — what would you like?"),
        (["Find datasets owned by that org.", "Show datasets from this organization."],
         "No organization was named — owner_org is unknown, so ask for it.",
         "Which organization? Give me its name and I'll search for the datasets it "
         "publishes on the National Data Platform."),
        (["Can you recommend a good dataset?"],
         "Open-ended with no topic — invite a subject before searching.",
         "Happy to point you to one — what subject are you interested in? Once I know "
         "the topic I can search the National Data Platform for strong matches."),
    ]
    for questions, thought, ans in items:
        for q in questions:
            examples.append({
                "instruction": q,
                "solution": {
                    "instruction": q,
                    "reasoning_path": [],      # no tool — prepare_data handles this
                    "final_answer": ans,
                    "no_tool_thought": thought,
                    "execution_validated": True,
                },
            })


# ─────────────────────────── main ───────────────────────────────────

def main() -> None:
    harvest = json.loads(HARVEST.read_text())
    examples: list[dict] = []
    log: list[str] = []

    print("[build] list_organizations …")
    build_list_org(harvest, examples)
    n1 = len(examples)
    print(f"  -> {n1} examples")

    print("[build] search_datasets (live) …")
    build_search(harvest, examples, log)
    build_search_extra(harvest, examples, log)
    n2 = len(examples)
    print(f"  -> {n2 - n1} examples")

    print("[build] get_dataset_details …")
    build_details(harvest, examples)
    n3 = len(examples)
    print(f"  -> {n3 - n2} examples")

    print("[build] search→details chains (live) …")
    build_chains(harvest, examples, log)
    n4 = len(examples)
    print(f"  -> {n4 - n3} examples")

    print("[build] no-tool examples …")
    build_notool(examples)
    n5 = len(examples)
    print(f"  -> {n5 - n4} examples")

    rng.shuffle(examples)
    OUT.write_text(json.dumps(examples, indent=1))

    print()
    for line in log:
        print("  NOTE:", line)
    print(f"\n[done] wrote {len(examples)} examples → {OUT}")
    tool_ex = sum(1 for e in examples if e["solution"]["reasoning_path"])
    notool = len(examples) - tool_ex
    print(f"        tool examples: {tool_ex}  |  no-tool: {notool}")


if __name__ == "__main__":
    main()
