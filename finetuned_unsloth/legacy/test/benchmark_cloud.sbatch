#!/bin/bash
#SBATCH --job-name=jarvis-cloud-bench
#SBATCH --account=bekn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.out
#SBATCH --error=/u/sislam3/Generator/finetuned_unsloth/logs/%x-%j.err

# Benchmark cloud-hosted LLMs on the real Jarvis MCP server.
#   - Claude: haiku / sonnet / opus (via claude-agent-sdk, uses local Claude Code session)
#   - Gemini: 2.5-flash / 2.5-pro   (via google-adk, needs GOOGLE_API_KEY)
#
# Both SDKs spawn their own jarvis-mcp subprocess internally, but we point them
# at the SAME binary / config as the Ollama runs so outcomes are comparable.
# --case-timeout 180 prevents a single flaky case from eating the whole job.

set -eu
module load python/anaconda3/2.10.0
source /u/sislam3/Generator/.venv-delta/bin/activate

# No local GPU model needs to load — this is all API / CLI calls.
export CLAUDE_AGENT_SDK=1
export GOOGLE_ADK=1
export GOOGLE_API_KEY="AIzaSyDb0SXpta5CHx6qcQ5q98NQEDPTSFYrxgA"

# Make sure the claude CLI is on PATH (installed via nvm under /u/sislam3)
export PATH="/u/sislam3/.nvm/versions/node/v24.14.1/bin:$PATH"

cd /u/sislam3/Generator

echo "=== env ==="
which claude && claude --version
python3 -c "from google.adk.agents import Agent; from claude_agent_sdk import query; print('SDK imports OK')"

echo ""
echo "=== reset jarvis MCP state (retry on cold-start init timeout) ==="
python3 -c "
import time
from inference.mcp_client import JarvisMCP, MCPError
last = None
for attempt in range(1, 4):
    try:
        with JarvisMCP(startup_timeout=60.0) as m:
            print(m.call_tool('jm_reset', {})[:80])
        break
    except MCPError as e:
        print(f'MCP init attempt {attempt} failed: {e}', flush=True)
        last = e
        time.sleep(5)
else:
    raise SystemExit(f'MCP never came up: {last}')
"

echo ""
echo "=== benchmark cloud models (Claude haiku/sonnet/opus + Gemini flash/pro) ==="
python finetuned_unsloth/test/benchmark_all.py \
    --models claude-haiku-4-5 claude-sonnet-4-6 claude-opus-4-7 gemini-2.5-flash gemini-2.5-pro \
    --tag cloud_v2 \
    --case-timeout 180 \
    "$@"

echo ""
echo "=== DONE — $(date) ==="
