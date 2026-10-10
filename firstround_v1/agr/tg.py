"""TigerGraph connection helper."""
from functools import lru_cache

from pyTigerGraph import TigerGraphConnection

from . import config


@lru_cache(maxsize=4)
def get_conn(graph: str | None = None) -> TigerGraphConnection:
    conn = TigerGraphConnection(
        host=config.TG_HOST,
        username=config.TG_USERNAME,
        password=config.TG_PASSWORD,
        restppPort=config.TG_PORT,
        gsPort=config.TG_PORT,
        graphname=graph or config.TG_GRAPH,
    )
    if graph != "":
        try:
            conn.getToken()
        except Exception:
            pass  # auth may be disabled on community edition
    return conn


def run_query(name: str, params: dict | None = None, timeout: int = 60000):
    return get_conn().runInstalledQuery(name, params or {}, timeout=timeout)
