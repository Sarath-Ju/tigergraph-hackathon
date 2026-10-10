"""Install the GSQL queries used by all three pipelines."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agr.tg import get_conn  # noqa: E402

out = get_conn().gsql((Path(__file__).resolve().parent.parent / "gsql" / "queries.gsql").read_text())
print(out[-3000:])
