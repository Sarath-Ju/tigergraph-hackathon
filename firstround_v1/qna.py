from pyTigerGraph import TigerGraphConnection

host = "http://localhost"

username = "tigergraph"
password = "tigergraph"

# We first create a connection to the database
conn = TigerGraphConnection(
    host=host,
    username=username, 
    password=password,
    restppPort=14240
    )

conn.graphname = "corpus2"

# And then add GraphRAG's address to the connection. This address
# is the host's address where the GraphRAG container is running.
conn.ai.configureGraphRAGHost(f"{host}:8000")
conn.ai.forceConsistencyUpdate("graphrag")


query = "Who won the gold medal in the women's 200 metre freestyle swimming event at the Summer Olympics held immediately before 2016?"
print(f"""Fetching answer for question: {query}""")

resp = conn.ai.answerQuestion(
    query,
    method="hybrid",
    method_parameters = {
        "indices": ["Document", "DocumentChunk"],
        "top_k": 2,
        "num_hops": 2,
        "num_seen_min": 2,
        "verbose": True
    })

print(f"""\nAnswer using Hybrid Search:\n{resp["response"]}""")

resp = conn.ai.answerQuestion(
    query,
    method="community",
    method_parameters={
        "community_level": 2,
        "combine": False,
        "top_k": 5,
        "verbose": True
    })

print(f"""\nAnswer using Community Search:\n{resp["response"]}""")

# Uses the graph's configured engine (agentic by default; falls back to
# classic if the chat model can't tool-call).
# Override (ex 2.0.5+): conn.ai.query(query, mode="agentic", rag_method="planned")
# print(help(conn.ai.query))
# agentic = conn.ai.query(query)
agentic =conn.ai.query(query, mode="agentic", rag_method="planned")

print(f"""\nAnswer using the Agentic engine:\n{agentic["natural_language_response"]}""")
