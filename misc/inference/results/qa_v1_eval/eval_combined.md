# jarvis-qa-v1 — offline evaluation vs base gemma-3-270m-it

- Validation set: `qa_v1/jarvis_qa_v1_cot.val.jsonl` (365 held-out rows, never seen in training)
- Greedy decoding, max_new_tokens=700, no MCP, pure text grading

## Headline metrics

| metric | jarvis-qa-v1 (FT 270M) | base gemma-3-270m-it | Δ |
|---|---|---|---|
| parse_success            (both `**Reasoning:**` and `**Answer:**` present) | 100.0%   | 9.9%   | **+90 pp** |
| has_reasoning_steps      (numbered / "step by step" / "let me" in reasoning)   | 100.0%   | 1.6%   | **+98 pp** |
| answer_keyword_recall    (content-word overlap with gold answer)              | 24.5% | 1.7% | **14.8× more** |
| finished cleanly (EOS)                                                        | 100.0% | 98.9% | — |
| reasoning length (chars, mean)                                                | 601.5 | 334.7 | — |
| answer length (chars, mean)                                                   | 321.4 | 31.0 | — |
| total length (chars, mean)                                                    | 951.9 | 386.5 | — |
| gen latency (s/example)                                                       | 5.453 | 2.072 | — |

## Interpretation

1. **Structure learned completely.** 100% of held-out prompts produce the exact `**Reasoning:** ... **Answer:** ...` format. Base model produces it ~10% of the time (random markdown bolding).
2. **CoT reasoning learned.** 100% of FT outputs contain numbered steps or "let me think" phrasing. Base: 1.6%.
3. **Content quality is real, not structural artifact.** Answer-keyword recall is 14.8× higher than base (24.5% vs 1.7%). Not a ceiling, but clearly the FT taught the model domain-specific terminology.
4. **Output length matches gold.** FT reasoning mean=601 chars (gold 691), answer mean=321 (gold 383). Base produces much shorter outputs (answer mean 31 chars = truncated single-sentence).
5. **Latency cost.** FT takes ~2.6× longer per generation because it actually produces a ~950-char structured response; base produces ~390 chars of mostly unstructured text.

## Conclusion

The fine-tuning pipeline works. A 270M model trained on 6,950 CoT examples for 3 epochs on a single GH200 (6.6 min) learns both the expected output structure (100%) and domain-specific answer content (14.8× baseline).