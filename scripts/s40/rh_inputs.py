"""S40 rehearsal inputs, pinned (Vu 26/10): a missing file or a different SHA256 stops; nothing is searched elsewhere. The directory was moved at
26/10 12:14:54 from out/runs/s40/rehearsal-input to out/runs/s40/abl/rehearsal-input by a process that is not this session (docs/s40-private.md)."""
import hashlib
from pathlib import Path

import pandas as pd

ID_MAP = Path("D:/GitHub/r2ai-stage3/out/runs/s40/abl/rehearsal-input/id-map-rh100.parquet")
ID_MAP_SHA256 = "dfe02ee79c725ad310b10e431ccf64d10ecba69e676e3c220035b4697f7cdc47"
QUERIES = Path("D:/GitHub/r2ai-stage3/out/runs/s40/abl/rehearsal-input/queries-rh100.parquet")
QUERIES_SHA256 = "181819a3291aed7f12b6b9a1859d27fb54462f25b24a6d393938fb0aa9c2dfda"


def id_map() -> pd.DataFrame:
    if not ID_MAP.exists() or hashlib.sha256(ID_MAP.read_bytes()).hexdigest() != ID_MAP_SHA256:
        raise SystemExit(f"{ID_MAP}: missing or SHA256 differs from {ID_MAP_SHA256}")
    return pd.read_parquet(ID_MAP)
