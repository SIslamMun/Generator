"""Offline evaluation for jarvis-qa-v1 on the held-out val set.

No MCP, no Ollama. We load the merged_16bit HF checkpoint, render each
validation prompt with the Gemma3 chat template, generate greedily, then
grade the output structurally and against the gold answer.

Metrics per example:
  - parse_success         : output contains both `**Reasoning:**` and `**Answer:**` markers
  - has_reasoning_steps   : the reasoning section uses step-by-step phrasing
  - answer_keyword_recall : fraction of gold-answer content words present in pred answer
  - reasoning_len / answer_len / total_len (chars)
  - finished              : generation ended with EOS, not hit max_new_tokens

Also prints 5 side-by-side samples at the end so you can eyeball quality.

Usage:
    python eval_qa.py \
        --model  /work/hdd/bekn/sislam3/jarvis_qa_v1_lora/merged_16bit \
        --val    /u/sislam3/Generator/finetuned_unsloth/data/qa_v1/jarvis_qa_v1_cot.val.jsonl \
        --out    /u/sislam3/Generator/inference/results/qa_v1_eval \
        [--max-new-tokens 700] [--max-examples 0] [--baseline unsloth/gemma-3-270m-it]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from collections import Counter


STOPWORDS = {
    "the","a","an","and","or","but","is","are","was","were","be","been","being",
    "of","to","in","on","at","for","with","by","from","as","it","its","this",
    "that","these","those","if","then","when","which","what","will","would",
    "can","could","should","may","might","must","do","does","did","has","have",
    "had","you","your","we","our","they","their","not","no","yes","so","up",
    "out","about","into","over","under","between","through","here","there",
    "how","why","who","whom","whose","i","he","she","him","her","them","me","us",
}


def content_words(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def split_reasoning_answer(out: str) -> tuple[str, str, bool]:
    """Extract reasoning + answer blocks from the generated text.

    Returns (reasoning, answer, parse_success).
    """
    m_reason = re.search(r"\*\*Reasoning:\*\*\s*(.*?)(?=\*\*Answer:\*\*|\Z)", out, re.DOTALL)
    m_ans    = re.search(r"\*\*Answer:\*\*\s*(.*)", out, re.DOTALL)
    reasoning = m_reason.group(1).strip() if m_reason else ""
    answer    = m_ans.group(1).strip()    if m_ans    else ""
    parse_success = bool(m_reason and m_ans)
    return reasoning, answer, parse_success


def has_step_by_step(reasoning: str) -> bool:
    r = reasoning.lower()
    return ("step by step" in r or "let me" in r or
            bool(re.search(r"^\s*\d+\.\s", r, re.MULTILINE)))


def grade_one(gold_assistant: str, pred_out: str) -> dict:
    g_reason, g_answer, _ = split_reasoning_answer(gold_assistant)
    p_reason, p_answer, parse_ok = split_reasoning_answer(pred_out)

    g_ans_words = content_words(g_answer)
    p_ans_words = set(content_words(p_answer))
    if g_ans_words:
        recall = sum(1 for w in g_ans_words if w in p_ans_words) / len(g_ans_words)
    else:
        recall = 0.0

    return {
        "parse_success":         parse_ok,
        "has_reasoning_steps":   has_step_by_step(p_reason),
        "answer_keyword_recall": recall,
        "reasoning_len":         len(p_reason),
        "answer_len":            len(p_answer),
        "total_len":             len(pred_out),
        "pred_reasoning":        p_reason,
        "pred_answer":           p_answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--max-examples", type=int, default=0, help="0 = all")
    ap.add_argument("--baseline", default=None,
                    help="Optional HF model id to compare against (e.g. unsloth/gemma-3-270m-it)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.val)]
    if args.max_examples:
        rows = rows[: args.max_examples]
    print(f"[data] {len(rows)} validation examples")

    # Lazy imports so --help works without torch.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def load_model(name):
        print(f"[model] loading {name}")
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, device_map="cuda",
        )
        mdl.eval()
        return tok, mdl

    def run_model(name, tag):
        tok, mdl = load_model(name)
        # Confirm chat template.
        if tok.chat_template is None:
            raise RuntimeError(f"{name} has no chat_template")

        eos_id = tok.eos_token_id
        results = []
        t0 = time.time()
        for i, row in enumerate(rows):
            convo = row["conversations"]
            prompt_msgs = convo[:2]  # system + user only
            gold_assistant = convo[2]["content"]

            rendered = tok.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True,
            ).removeprefix("<bos>")

            inputs = tok(rendered, return_tensors="pt").to("cuda")
            input_len = inputs["input_ids"].shape[1]

            t_gen = time.time()
            with torch.inference_mode():
                out = mdl.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    eos_token_id=eos_id,
                    pad_token_id=tok.pad_token_id or eos_id,
                )
            gen_ids = out[0, input_len:]
            finished = gen_ids[-1].item() == eos_id if len(gen_ids) > 0 else False
            text = tok.decode(gen_ids, skip_special_tokens=True).strip()

            m = grade_one(gold_assistant, text)
            m.update({
                "case_idx":   i,
                "category":   row.get("type", "unknown"),
                "question":   convo[1]["content"],
                "gold":       gold_assistant,
                "pred":       text,
                "finished":   finished,
                "gen_latency_s": round(time.time() - t_gen, 3),
            })
            results.append(m)

            if (i + 1) % 20 == 0:
                dt = time.time() - t0
                ps = sum(r["parse_success"] for r in results) / len(results)
                kr = sum(r["answer_keyword_recall"] for r in results) / len(results)
                print(f"  [{i+1:>4}/{len(rows)}] parse={100*ps:.0f}%  kw_recall={100*kr:.0f}%  t={dt:.0f}s", flush=True)
        total = time.time() - t0

        ps  = sum(r["parse_success"] for r in results) / len(results)
        rs  = sum(r["has_reasoning_steps"] for r in results) / len(results)
        kr  = sum(r["answer_keyword_recall"] for r in results) / len(results)
        fin = sum(r["finished"] for r in results) / len(results)
        rl  = sum(r["reasoning_len"] for r in results) / len(results)
        al  = sum(r["answer_len"] for r in results) / len(results)
        tl  = sum(r["total_len"] for r in results) / len(results)
        lat = sum(r["gen_latency_s"] for r in results) / len(results)

        summary = {
            "tag": tag, "model": name, "n": len(results),
            "parse_success":         round(ps, 4),
            "has_reasoning_steps":   round(rs, 4),
            "answer_keyword_recall": round(kr, 4),
            "finished":              round(fin, 4),
            "reasoning_len_mean":    round(rl, 1),
            "answer_len_mean":       round(al, 1),
            "total_len_mean":        round(tl, 1),
            "gen_latency_s_mean":    round(lat, 3),
            "elapsed_s":             round(total, 1),
        }

        # write jsonl + summary
        with (out_dir / f"{tag}_predictions.jsonl").open("w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[{tag}] summary: {summary}")

        # free gpu
        del mdl
        torch.cuda.empty_cache()
        return summary, results

    ft_summary, ft_results = run_model(args.model, "jarvis_qa_v1")
    base_summary = None
    if args.baseline:
        base_summary, base_results = run_model(args.baseline, "gemma3_270m_base")

    # write a combined markdown report
    lines = [
        "# jarvis-qa-v1 — offline evaluation on held-out val set",
        "",
        f"- Validation set: `{args.val}` ({len(rows)} rows)",
        f"- Model:         `{args.model}`",
        f"- Greedy decoding, max_new_tokens={args.max_new_tokens}",
        "",
        "## Summary metrics",
        "",
        "| metric | jarvis-qa-v1 (FT) | base gemma-3-270m-it |",
        "|---|---|---|",
    ]
    if base_summary:
        for k in ["parse_success","has_reasoning_steps","answer_keyword_recall","finished"]:
            lines.append(f"| {k} | {100*ft_summary[k]:.1f}% | {100*base_summary[k]:.1f}% |")
        for k in ["reasoning_len_mean","answer_len_mean","total_len_mean","gen_latency_s_mean"]:
            lines.append(f"| {k} | {ft_summary[k]} | {base_summary[k]} |")
    else:
        for k in ["parse_success","has_reasoning_steps","answer_keyword_recall","finished"]:
            lines.append(f"| {k} | {100*ft_summary[k]:.1f}% | — |")
        for k in ["reasoning_len_mean","answer_len_mean","total_len_mean","gen_latency_s_mean"]:
            lines.append(f"| {k} | {ft_summary[k]} | — |")

    lines += ["", "## Sample side-by-side (first 5 val examples)", ""]
    for r in ft_results[:5]:
        lines += [
            f"### Case {r['case_idx']} ({r['category']})",
            "",
            f"**Question:** {r['question']}",
            "",
            "**Gold:**",
            "```",
            r["gold"],
            "```",
            "",
            "**jarvis-qa-v1 prediction:**",
            "```",
            r["pred"],
            "```",
            "",
            f"parse_success={r['parse_success']}  step_reasoning={r['has_reasoning_steps']}  "
            f"kw_recall={r['answer_keyword_recall']:.2f}  "
            f"lens={r['reasoning_len']}/{r['answer_len']}  finished={r['finished']}",
            "",
        ]

    (out_dir / "eval_report.md").write_text("\n".join(lines))
    print(f"\nreport → {out_dir / 'eval_report.md'}")
    print(f"preds  → {out_dir}/jarvis_qa_v1_predictions.jsonl")


if __name__ == "__main__":
    main()
