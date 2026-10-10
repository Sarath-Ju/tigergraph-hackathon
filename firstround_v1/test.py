import os
import json
from pyTigerGraph import TigerGraphConnection

def create_graph(conn: TigerGraphConnection):
    # conn.gsql(f"""CREATE GRAPH {conn.graphname}()""")
    conn.ai.initializeGraphRAG()
    print(f"Graph {conn.graphname} created successfully.")

def load_data(conn: TigerGraphConnection,source_path="/data/corpus.jsonl"):
    res = conn.ai.createDocumentIngest(
        data_source="server",
        data_source_config={"data_path": source_path,},
        file_format="json",
    ) 

    print("CREATE INGEST RESULT:")
    print(res)
    print(res)
    print("load_job_id   =", res.get("load_job_id"))
    print("data_source_id=", res.get("data_source_id"))
    print("data_path     =", res.get("data_path"))
    print("data_source   =", res.get("data_source"))
    print("load_job_info =", res.get("load_job_info"))
    # print(help(conn.ai.runDocumentIngest))
    
    result = conn.ai.runDocumentIngest(
        load_job_id=res.get("load_job_id"),
        data_source_id=res.get("data_source_id"),
        data_path=res.get("data_path"),
        
    )

    print(result)



def update_graphrag(conn: TigerGraphConnection):
    conn.ai.forceConsistencyUpdate("graphrag")
    print("GraphRAG updated successfully.")

if __name__ == "__main__":

    # We first create a connection to the database
    conn = TigerGraphConnection(
        host="http://localhost",
        username="tigergraph",
        password="tigergraph",
        restppPort="14240",
    )
    conn.graphname = "corpus5"  # Set the graph name to "corpus2"

    # And then add GraphRAG's address to the connection. This address
    # is the host's address where the GraphRAG container is running.
    conn.ai.configureGraphRAGHost("http://localhost:8000")

    create_graph(conn)
    load_data(conn)
    update_graphrag(conn)
