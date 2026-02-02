#!/bin/bash
set -e

cd /home/cc/Generator

echo "========================================================================"
echo "Starting CoT Enhancement for Jarvis QA Datasets"
echo "========================================================================"
echo ""
echo "Timestamp: $(date)"
echo ""

# First: Enhance CURATED dataset
echo "========================================================================"
echo "1. Enhancing CURATED dataset (3911 pairs)"
echo "========================================================================"
echo ""

START_CURATED=$(date +%s)

uv run generator enhance-cot \
  jarvis_qa_new_prompts/jarvis_qa_curated.json \
  -o jarvis_qa_new_prompts/jarvis_qa_curated_cot.json \
  --provider ollama \
  --model gpt-oss:20b \
  --workers 24 \
  --batch-size 5

END_CURATED=$(date +%s)
DURATION_CURATED=$((END_CURATED - START_CURATED))

echo ""
echo "✓ Curated dataset enhanced in ${DURATION_CURATED}s ($(($DURATION_CURATED / 60))m)"
echo ""

# Second: Enhance RAW dataset
echo "========================================================================"
echo "2. Enhancing RAW dataset (7520 pairs)"
echo "========================================================================"
echo ""

START_RAW=$(date +%s)

uv run generator enhance-cot \
  jarvis_qa_new_prompts/jarvis_qa.json \
  -o jarvis_qa_new_prompts/jarvis_qa_raw_cot.json \
  --provider ollama \
  --model gpt-oss:20b \
  --workers 24 \
  --batch-size 5

END_RAW=$(date +%s)
DURATION_RAW=$((END_RAW - START_RAW))

echo ""
echo "✓ Raw dataset enhanced in ${DURATION_RAW}s ($(($DURATION_RAW / 60))m)"
echo ""

# Summary
echo "========================================================================"
echo "CoT Enhancement Complete!"
echo "========================================================================"
echo ""
echo "Curated dataset: ${DURATION_CURATED}s ($(($DURATION_CURATED / 60))m)"
echo "Raw dataset:     ${DURATION_RAW}s ($(($DURATION_RAW / 60))m)"
echo "Total time:      $((DURATION_CURATED + DURATION_RAW))s ($((($DURATION_CURATED + $DURATION_RAW) / 60))m)"
echo ""
echo "Output files:"
echo "  - jarvis_qa_new_prompts/jarvis_qa_curated_cot.json"
echo "  - jarvis_qa_new_prompts/jarvis_qa_raw_cot.json"
echo ""
echo "Completed at: $(date)"
echo ""
