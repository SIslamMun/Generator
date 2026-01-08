# Documentation Index

**Author:** Shazzadul  
**Complete guide to all Generator documentation - Start here!**

---

## 🎯 Quick Start

**New to Generator?** Start here:
1. Read [README.md](../README.md) - Quick start guide
2. Skim [OVERVIEW.md](OVERVIEW.md) - 10-minute overview
3. Try a command: `uv run generator generate /path/to/lancedb -o test.json --target-pairs 50`

**Want to understand the design?** 
→ Read [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md)

**Want to understand how data flows?**
→ Read [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md)

**Want to see which papers were used?**
→ Read [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md)

---

## 📚 Documentation Overview

### Level 1: Quick Reference (Read First)

| Document | Length | Purpose | Read When |
|----------|--------|---------|-----------|
| [README.md](../README.md) | 550 lines | Quick start, commands, examples | Starting out |
| [OVERVIEW.md](OVERVIEW.md) | ~600 lines | Key points, design summary | Want overview |
| [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) | Visual | System diagrams, data flows | Visual learner |

### Level 2: Comprehensive Guides (Deep Dive)

| Document | Length | Purpose | Read When |
|----------|--------|---------|-----------|
| [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) | 12,000+ words | Complete design rationale | Want to understand *why* |
| [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) | 10,000+ words | How extraction works | Want to understand *how* |
| [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md) | 5,000+ words | Paper → code mapping | Want research references |



### By Topic

#### **Design & Architecture**
- **Design rationale:** [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) → Section 8 "Implementation Decisions"
- **System overview:** [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) → "System Overview"
- **Research foundation:** [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) → Section 1 "Research Foundation"

#### **How Things Work**
- **QA extraction:** [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) → Section 3 "QA Extraction Process"
- **CoT generation:** [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) → Section 4 "CoT Extraction Process"
- **Tool-use generation:** [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) → Section 5 "Tool-Use Extraction Process"
- **Quality control:** [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) → Section 6 "Quality Control Pipeline"

#### **Research Papers**
- **Which papers:** [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md) → "Paper → Implementation Mapping"
- **What was extracted:** [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) → Section 1.1 "Core Papers"
- **Code locations:** [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md) → Summary Table

#### **Usage & Commands**
- **Quick start:** [README.md](README.md) → "Quick Start"
- **All commands:** [README.md](README.md) → "QA Pipeline Commands" + "Tool-Use Pipeline Commands"
- **Configuration:** [README.md](README.md) → "Configuration"
- **Examples:** [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) → Section 7 "Summary"

---

## 📖 Reading Paths

### For Different Audiences

#### **Researcher**
*"I want to understand design decisions and research basis"*

1. [OVERVIEW.md](OVERVIEW.md) - Get overview (10 min)
2. [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md) - See paper connections (15 min)
3. [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) - Understand rationale (45 min)

**Total:** ~70 minutes for complete understanding

#### **Developer / Contributor**
*"I want to understand the code and add features"*

1. [README.md](README.md) - Quick start (10 min)
2. [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md) - System structure (15 min)
3. [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) - How it works (30 min)
4. Browse code: `src/generator/*.py` with docstrings

**Total:** ~55 minutes + code exploration

#### **User / Data Scientist**
*"I just want to generate training data"*

1. [README.md](../README.md) - Installation + Quick Start (15 min)
2. [OVERVIEW.md](OVERVIEW.md) → "Example Usage" (5 min)
3. Try commands, read outputs
4. If issues → [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) → Relevant section

**Total:** ~20 minutes to get started

#### **Reviewer / Evaluator**
*"I need to evaluate this work"*

1. [OVERVIEW.md](OVERVIEW.md) - Overview (10 min)
2. [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) → Section 1 "Research Foundation" (20 min)
3. [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md) → Summary Table (5 min)
4. [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md) → "Example Results" (5 min)
5. Look at generated data: `output/*/`

**Total:** ~40 minutes for evaluation

---

## 🔍 Find Information Quickly

### Common Questions → Where to Look

| Question | Document | Section |
|----------|----------|---------|
| How do I install and run? | [README.md](../README.md) | Installation, Quick Start |
| What papers were used? | PAPER_IMPLEMENTATIONS.md | Paper → Implementation Mapping |
| Why was X designed this way? | DESIGN_DOCUMENTATION.md | Section 8: Implementation Decisions |
| How does QA generation work? | EXTRACTION_METHODOLOGY.md | Section 3: QA Extraction Process |
| How does tool-use work? | EXTRACTION_METHODOLOGY.md | Section 5: Tool-Use Extraction Process |
| What's the data flow? | ARCHITECTURE_DIAGRAMS.md | Data Flow diagram |
| How to configure providers? | README.md | Configuration section |

### Search by Keyword

**Instruction Backtranslation:**
- [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) → Section 1.1 "Paper 2"
- [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) → Section 3
- [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md) → Section 2

**LIMA / Quality:**
- [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) → Section 2.1, 6.1
- [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md) → Section 1

**Chain-of-Thought / CoT:**
- [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) → Section 4
- [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md) → Section 3

**Tool-Use / Agentic:**
- [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md) → Section 5
- [PHASE3_TOOL_USE_SPEC.md](PHASE3_TOOL_USE_SPEC.md)
- [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md) → Sections 5-7

**LLM Providers:**
- [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) → Section 7
- [README.md](README.md) → Cloud Providers section

---

## 📊 Documentation Statistics

**Total Documentation:**
- **27,000+** words across comprehensive guides
- **8** main documents
- **7** research papers fully documented
- **2** major pipelines (QA + Tool-Use)
- **6** LLM providers supported
- **4** export formats

**Coverage:**
- ✅ Design rationale for every major decision
- ✅ Research paper → implementation mapping
- ✅ Step-by-step extraction methodology
- ✅ Complete command reference
- ✅ Architecture diagrams
- ✅ Code examples throughout

---

## 🎓 Learning Path

### Beginner → Expert

**Week 1: Understanding**
- Day 1: [README.md](../README.md) + [OVERVIEW.md](OVERVIEW.md)
- Day 2: [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)
- Day 3: [PAPER_IMPLEMENTATIONS.md](PAPER_IMPLEMENTATIONS.md)
- Day 4-5: [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md)

**Week 2: Deep Dive**
- Day 1-2: [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md)
- Day 3: Code exploration (`src/generator/*.py`)
- Day 4-5: Hands-on experimentation

**Week 3: Mastery**
- Implement new features
- Write tests
- Generate custom datasets
- Contribute improvements

---

## 🔗 External Resources

### Research Papers (ArXiv)
- [LIMA](https://arxiv.org/abs/2305.11206) - Meta AI, NeurIPS 2023
- [Instruction Backtranslation](https://arxiv.org/abs/2308.06259) - Meta AI, ICLR 2024
- [Distilling Step-by-Step](https://arxiv.org/abs/2305.02301) - Google, 2023
- [AlpaGasus](https://arxiv.org/abs/2307.08701) - UMD, ICLR 2024
- [Toolformer](https://arxiv.org/abs/2302.04761) - Meta AI, NeurIPS 2023
- [Gorilla](https://arxiv.org/abs/2305.15334) - UC Berkeley, NeurIPS 2024
- [ToolLLM](https://arxiv.org/abs/2307.16789) - Tsinghua, ICLR 2024

### LLM Provider Docs
- [Ollama](https://ollama.com/docs)
- [Anthropic Claude](https://docs.anthropic.com/)
- [Google Gemini](https://ai.google.dev/docs)
- [vLLM](https://docs.vllm.ai/)
- [OpenAI](https://platform.openai.com/docs)

---

## 📝 Quick Reference Cards

### Commands Cheatsheet

```bash
# QA Pipeline
uv run generator generate <db> -o qa.json --target-pairs 300
uv run generator enrich qa.json -o enriched.json
uv run generator curate enriched.json -o curated.json --threshold 7.0
uv run generator enhance-cot curated.json -o cot.json
uv run generator export cot.json -o training.jsonl -f chatml

# Or full pipeline
uv run generator pipeline <db> -o training.jsonl

# Tool-Use Pipeline
uv run generator tool-parse api.json -o tools.json
uv run generator tool-generate tools.json -o examples.json --mode auto
uv run generator tool-execute examples.json -o verified.json --mode simulated
uv run generator tool-curate verified.json -o curated.json --threshold 7.0

# Or full pipeline
uv run generator tool-pipeline api.json -o tool_training.json
```

### File Locations

```
Key Code:
- QA Generation: src/generator/qa_generator.py
- CoT Generation: src/generator/cot_generator.py
- Curation: src/generator/curate.py
- Tool Generation: src/generator/tool_generator.py

Config:
- Providers: configs/config.yaml
- Prompts: configs/prompts/*.yaml

Output:
- Generated data: output/*/
- Tests: tests/test_*.py
```

---

## 🚀 Next Steps After Reading

### For Researchers
1. Understand the methodology
2. Evaluate generated datasets
3. Compare with baseline approaches
4. Suggest improvements

### For Developers
1. Set up development environment
2. Run tests: `pytest tests/`
3. Try generating data
4. Explore extending functionality

### For Users
1. Install: `uv pip install -e .`
2. Configure provider: Edit `configs/config.yaml`
3. Generate data: Follow Quick Start
4. Fine-tune your model

---

## 📞 Need Help?

**Documentation not clear?** 
- Check if answer is in a different document (use this index)
- Look at code examples in [EXTRACTION_METHODOLOGY.md](EXTRACTION_METHODOLOGY.md)
- Read the relevant paper for deeper context

**Found an issue?**
- Review error handling in code
- Check provider-specific documentation

**Want to contribute?**
- Read [DESIGN_DOCUMENTATION.md](DESIGN_DOCUMENTATION.md) first
- Understand architecture in [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md)
- Review existing tests in `tests/`
- Follow code patterns in `src/generator/`

---

**Document Version:** 1.0  
**Last Updated:** January 8, 2026

**Total Documentation Pages:** 6 comprehensive guides + this index  
**Total Words:** 30,000+ across all documents  
**Coverage:** Complete system documentation
