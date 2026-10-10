# Agentic GraphRAG on TigerGraph

Our entry for the TigerGraph Agentic GraphRAG Hackathon. The repository holds two versions of the system. Both
answer every question three ways, with RAG, GraphRAG and Agentic GraphRAG, use `gpt-4.1-mini` for all three, and
report accuracy, tokens, latency and a full agent trace for every answer.

| Folder | What it is |
|---|---|
| [firstround_v1](firstround_v1/) | Our own agent harness: an orchestrator that picks the next action, specialised agents (entity linker, graph traversal, temporal reasoner, aggregator, similarity search, document retriever) and an evidence evaluator that checks every answer before it is accepted. Built for the Olympic part of the corpus, which is what the benchmark asks about. |
| [firstround_v2](firstround_v2/) | A topic-agnostic version built on TigerGraph's own GraphRAG engine and planning agent. Our record layer reads any corpus, works out record types, numeric fields and relations from its structured blocks, and gives them to TigerGraph as the graph schema. It also documents four problems we found in TigerGraph's agent and how we fixed them without changing its code. |
| [hidden_results.json](hidden_results.json) | The 50 hidden questions with the answers from both versions and all three approaches, the tokens used, the latency, and the full agentic trace. |
| [hidden_results.csv](hidden_results.csv) | The same answers and token counts as a table, without the traces. |

Each folder has its own README with the architecture diagram, setup instructions and dependencies
(`requirements.txt`).

## Results on the 100 public questions

| | v1 RAG | v1 GraphRAG | v1 Agentic | v2 RAG | v2 GraphRAG | v2 Agentic |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 50.5% | 66.0% | **100%** | 49.0% | 48.0% | **81.0%** |
| Tokens per question | 2,939 | 4,585 | 3,007 | 2,334 | 2,334 | 33,493 |
| Latency per question | 3.87 s | 7.88 s | 2.46 s | 1.93 s | 1.92 s | 13.09 s |

v2 was also tested on 33 questions we wrote about the non-Olympic part of the corpus (films and officeholders).
There its agent scores 88%, against 42% for RAG.

In both versions the agent matters for counting, ranking and chained questions, where a few retrieved paragraphs
cannot hold the answer. For a question about one named thing, plain RAG already gets 95% for far fewer tokens.
v1 shows that an agent does not have to be expensive when its tools return small, exact evidence. v2 shows the
same pattern holds on any topic, at a higher token cost that comes from TigerGraph GraphRAG's query tool (see its
README for where the tokens go).

## Running it

Both versions need Docker, Python 3.10 or newer and an OpenAI key, and the hackathon dataset. Put
`corpus.jsonl` from the dataset in `data/corpus/` of the version you run. The question files are already in
`data/question/`.

v1 and v2 both run TigerGraph on port 14240, so run one at a time.

**v1** needs a TigerGraph Community Edition container:

```bash
docker run -d --name tigergraph -p 14240:14240 \
  -v tigergraph_data:/home/tigergraph/tigergraph/data tigergraph/community:4.2.2
docker exec tigergraph /home/tigergraph/tigergraph/app/cmd/gadmin start all
cd firstround_v1
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # then add your OPENAI_API_KEY
.venv/bin/python scripts/build_graph.py
.venv/bin/python scripts/install_queries.py
.venv/bin/python -m agr.ask --pipeline all "Who won the gold medal in the men's 20 kilometres walk at the Summer Olympics held immediately before 2016?"
```

**v2** brings up TigerGraph and the GraphRAG service with Docker Compose and builds everything in one command:

```bash
cd firstround_v2
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # then add your OPENAI_API_KEY
bash scripts/fresh_run.sh
```

## Built with

TigerGraph Community Edition 4.2 with TigerVector, TigerGraph GraphRAG
([github.com/tigergraph/graphrag](https://github.com/tigergraph/graphrag)), which v2 runs from its published
Docker images and which we studied while designing both versions, OpenAI `gpt-4.1-mini` and
`text-embedding-3-small`.
