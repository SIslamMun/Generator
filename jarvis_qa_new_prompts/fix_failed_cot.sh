#!/bin/bash
set -e

cd /home/cc/Generator

echo "========================================================================"
echo "Fixing Failed CoT Pairs for Jarvis QA Datasets"
echo "========================================================================"
echo ""
echo "Timestamp: $(date)"
echo ""

# First: Fix CURATED dataset
echo "========================================================================"
echo "1. Fixing CURATED CoT dataset"
echo "========================================================================"
echo ""

START_CURATED=$(date +%s)

uv run generator fix-cot \
  jarvis_qa_new_prompts/jarvis_qa_curated_cot.json \
  -o jarvis_qa_new_prompts/jarvis_qa_curated_cot_fixed.json \
  --provider ollama \
  --model gpt-oss:20b \
  --workers 24 \
  --batch-size 5

END_CURATED=$(date +%s)
DURATION_CURATED=$((END_CURATED - START_CURATED))

echo ""
echo "✓ Curated dataset fixed in ${DURATION_CURATED}s ($(($DURATION_CURATED / 60))m)"
echo ""

# Second: Fix RAW dataset
echo "========================================================================"
echo "2. Fixing RAW CoT dataset"
echo "========================================================================"
echo ""

START_RAW=$(date +%s)

uv run generator fix-cot \
  jarvis_qa_new_prompts/jarvis_qa_raw_cot.json \
  -o jarvis_qa_new_prompts/jarvis_qa_raw_cot_fixed.json \
  --provider ollama \
  --model gpt-oss:20b \
  --workers 24 \
  --batch-size 5

END_RAW=$(date +%s)
DURATION_RAW=$((END_RAW - START_RAW))

echo ""
echo "✓ Raw dataset fixed in ${DURATION_RAW}s ($(($DURATION_RAW / 60))m)"
echo ""

# Summary
echo "========================================================================"
echo "CoT Fix Complete!"
echo "========================================================================"
echo ""
echo "Curated dataset: ${DURATION_CURATED}s ($(($DURATION_CURATED / 60))m)"
echo "Raw dataset:     ${DURATION_RAW}s ($(($DURATION_RAW / 60))m)"
echo "Total time:      $((DURATION_CURATED + DURATION_RAW))s ($((($DURATION_CURATED + $DURATION_RAW) / 60))m)"
echo ""
echo "Output files:"
echo "  - jarvis_qa_new_prompts/jarvis_qa_curated_cot_fixed.json"
echo "  - jarvis_qa_new_prompts/jarvis_qa_raw_cot_fixed.json"
echo ""
echo "Completed at: $(date)"
echo ""
