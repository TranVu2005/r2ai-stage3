"""S40 rehearsal input: 100 random public queries (numpy default_rng(2026), sorted by original id), renumbered 5001..5100 so that any read
of a file keyed by the public ids shows up as a mismatch. Writes <out>/queries-rh100.parquet (id, query) and <out>/id-map-rh100.parquet."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from r2ai import paths  # noqa: E402

out = paths.RUNS_DIR / "s40" / "rehearsal-input"
paths.assert_writable(out)
out.mkdir(parents=True, exist_ok=True)
q = pd.read_parquet(paths.DATA_DIR / "raw" / "query.parquet")
orig = np.sort(np.random.default_rng(2026).choice(q.id.to_numpy(), 100, replace=False))
mapping = pd.DataFrame({"id": np.arange(5001, 5101), "orig_id": orig})
text = dict(zip(q.id.astype(int), q["query"]))
pd.DataFrame({"id": mapping.id, "query": [text[int(i)] for i in orig]}).to_parquet(out / "queries-rh100.parquet", index=False)
mapping.to_parquet(out / "id-map-rh100.parquet", index=False)
print(out, orig[:10].tolist())
