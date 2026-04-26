"""Universal training pipeline: LanceDB → curated dataset → fine-tuned model.

Public entry points are wired into generator.cli as `train-init`, `train-chat`,
`train-tool`, and `ingest`. The internals live here:

    config.py    — YAML schema + interactive wizard
    runner.py    — Stage execution with idempotency
    hardware.py  — GPU/CPU detection
    stages/      — one module per pipeline stage
"""

__version__ = "0.1.0"
