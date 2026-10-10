#!/usr/bin/env bash
# Start from nothing and run the whole thing: services, a clean graph, the build, a sample question,
# the benchmark, scoring and the submission files.
#
#   bash scripts/fresh_run.sh                  everything
#   bash scripts/fresh_run.sh --no-benchmark   stop after the sample question
#   SAMPLE=100 bash scripts/fresh_run.sh       build a 100-document sample instead of the full corpus
#
# It drops the graph named by TG_GRAPH in .env and deletes the earlier results, so every number
# in the log comes from this run. Everything printed is also saved to results/fresh_run_<time>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
COMPOSE="docker compose -f infra/docker-compose.yml"
GADMIN=/home/tigergraph/tigergraph/app/cmd/gadmin
GSQL=/home/tigergraph/tigergraph/app/cmd/gsql
BENCHMARK=1
[[ "${1:-}" == "--no-benchmark" ]] && BENCHMARK=0

mkdir -p results
LOG="results/fresh_run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "==================== $* ===================="; date "+%H:%M:%S"; }
T0=$(date +%s)

step "0. Checking settings"
KEY_LEN=$($PY -c "from agr2 import config; import os; print(len(os.getenv('OPENAI_API_KEY', '')))")
if [[ "$KEY_LEN" -lt 20 ]]; then
  echo "OPENAI_API_KEY in .env is missing or still the sk-... placeholder. Put your real key in .env first."
  exit 1
fi
GRAPH=$($PY -c "from agr2 import config; print(config.TG_GRAPH)")
echo "graph: $GRAPH"
echo "model: $($PY -c "from agr2 import config; print(config.LLM_MODEL, '/', config.EMBED_MODEL)")"

step "1. Starting TigerGraph and the GraphRAG service"
# v1 runs its own TigerGraph on the same port, so it has to be stopped first
if docker ps --format '{{.Names}}' | grep -qx tigergraph; then
  echo "stopping the v1 TigerGraph container, which uses the same port"
  docker stop tigergraph
fi
$PY infra/render_config.py
$COMPOSE up -d
$COMPOSE exec -T tigergraph $GADMIN start all

# wait_for <name> <command>: retry the command every 5 seconds for up to 5 minutes
wait_for() {
  echo -n "waiting for $1"
  for _ in $(seq 1 60); do
    if eval "$2" >/dev/null 2>&1; then echo " ready"; return 0; fi
    echo -n "."; sleep 5
  done
  echo " not ready after 5 minutes"; exit 1
}
TG_PORT=$($PY -c "from agr2 import config; print(config.TG_PORT)")
wait_for "TigerGraph on port $TG_PORT" \
  "[[ \$(curl -s -o /dev/null -w '%{http_code}' -u tigergraph:tigergraph localhost:$TG_PORT/restpp/echo) == 200 ]]"
# the GraphRAG containers connect to TigerGraph when they start, so restart them now that it is up
$COMPOSE restart graphrag graphrag-ecc chat-history
wait_for "the GraphRAG service" "curl -s localhost:8000/health | grep -q '\"status\":\"healthy\"'"

step "2. Removing the old graph and old results"
# GSQL comes up later than the REST endpoint, and an empty answer from it would look like "no graph".
# We use grep without -q on gsql output: -q stops reading at the first match, gsql then fails on the
# closed pipe, and with pipefail the whole check counts as failed even though the text was there.
wait_for "GSQL" "$COMPOSE exec -T tigergraph $GSQL ls | grep 'Graphs:'"
if $COMPOSE exec -T tigergraph $GSQL "ls" | grep "Graph $GRAPH(" >/dev/null; then
  $COMPOSE exec -T tigergraph $GSQL "DROP GRAPH $GRAPH CASCADE"
else
  echo "no graph called $GRAPH yet"
fi
if $COMPOSE exec -T tigergraph $GSQL "ls" | grep "Graph $GRAPH(" >/dev/null; then
  echo "the graph $GRAPH is still there after dropping it, stopping"; exit 1
fi
rm -rf results/graph results/public results/general results/hidden results/submission results/summary.json

step "3. Building the graph"
if [[ -n "${SAMPLE:-}" ]]; then
  $PY scripts/build_graph.py --sample "$SAMPLE"
else
  $PY scripts/build_graph.py
fi

step "4. Waiting for GraphRAG to chunk and embed the documents"
$PY scripts/graph_status.py --watch

step "5. Linking records to their chunks"
$PY scripts/finalize_graph.py
$PY scripts/check_vectors.py

step "6. One question through all three approaches"
$PY scripts/ask.py --pipelines rag,graphrag,agentic "Who directed the film The English Patient?"

if [[ "$BENCHMARK" == 1 ]]; then
  step "7. Benchmark: public questions"
  $PY scripts/run_benchmark.py --split public --pipelines rag,graphrag,agentic
  step "8. Benchmark: general questions (not about the Olympics)"
  $PY scripts/run_benchmark.py --split general --pipelines rag,graphrag,agentic
  step "9. Benchmark: hidden questions"
  $PY scripts/run_benchmark.py --split hidden --pipelines rag,graphrag,agentic
  step "10. Scoring"
  $PY scripts/evaluate.py --pipelines rag,graphrag,agentic
  step "11. Submission files"
  $PY scripts/export_submission.py
fi

step "Done in $(( ($(date +%s) - T0) / 60 )) minutes"
echo "log:      $LOG"
if [[ "$BENCHMARK" == 1 ]]; then
  echo "scores:   results/summary.json"
  echo "submit:   results/submission/"
fi
