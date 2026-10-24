"""S38 compose: carriers override, base bundle from a docsel (S37 mode) and the 120ask registry body sources."""
import json
from pathlib import Path

import numpy as np

import pandas as pd

from r2ai.s38 import compose, registry
from r2ai.s38.score import choose


def rows_of(chunks):
    return [{"id": 1, "relevant_docs": [10, 11, 12, 13], "relevant_chunks": [{"doc_id": d, "chunk_text": f"body{d}"} for d in chunks]},
            {"id": 2, "relevant_docs": [20, 21], "relevant_chunks": [{"doc_id": 20, "chunk_text": "body20"}]}]


def docsel_frame(carriers):
    rows = [(1, d, k + 1, 0.9 - 0.1 * k) for k, d in enumerate([10, 11, 12, 13])] + [(2, 20, 1, 0.8), (2, 21, 2, 0.1)]
    frame = pd.DataFrame(rows, columns=["query_id", "doc_id", "rank", "p4b"])
    frame["chunk"] = [(q, d) in carriers for q, d in zip(frame.query_id, frame.doc_id)]
    return frame


def test_compose_swaps_drops_a_chunk_added_then_removed():
    first = {"removed": [[1, 10]], "added": [{"query_id": 1, "doc_id": 12, "body": "x"}, {"query_id": 1, "doc_id": 13, "body": "y"}]}
    second = {"removed": [[1, 12], [2, 20]], "added": [{"query_id": 1, "doc_id": 11, "body": "z"}]}
    out = compose.compose_swaps(first, second)
    assert sorted(map(tuple, out["removed"])) == [(1, 10), (2, 20)]               # (1, 12) never reaches the base bundle
    assert sorted(a["doc_id"] for a in out["added"]) == [11, 13]


def test_carrier_swap_moves_the_carriers_to_the_target(monkeypatch):
    rows = rows_of([10, 11])
    docsel = docsel_frame({(1, 10), (1, 11), (2, 20)})
    monkeypatch.setattr(compose, "s29_chunk_texts", lambda pairs: {(1, 12): ("body", "body12"), (1, 13): (None, "")})
    swap = compose.carrier_swap(rows, docsel, pd.DataFrame({"query_id": [1, 1, 1, 2], "doc_id": [11, 12, 13, 20]}))
    assert swap["removed"] == [[1, 10]]
    assert [a["doc_id"] for a in swap["added"]] == [12] and swap["added"][0]["rank"] == 3
    assert swap["skipped_no_source"] == [[1, 13]]
    out = compose.apply_swap(rows, swap)
    assert [c["doc_id"] for c in out[0]["relevant_chunks"]] == [11, 12]           # chunks follow the docs order
    assert out[0]["relevant_docs"] == [10, 11, 12, 13]                            # docs untouched


def test_carrier_swap_refuses_a_carrier_outside_the_docsel():
    try:
        compose.carrier_swap(rows_of([10]), docsel_frame({(1, 10)}), pd.DataFrame({"query_id": [1], "doc_id": [99]}))
    except ValueError as err:
        assert "not docs of the docsel" in str(err)
    else:
        raise AssertionError("no error")


def test_virtual_base_roundtrip(tmp_path, monkeypatch):
    docsel = docsel_frame({(1, 10), (1, 12), (2, 20)})
    monkeypatch.setattr(compose, "s29_chunk_texts", lambda pairs: {(q, d): ("body", f"body{d}") if d != 12 else (None, "")
                                                                   for q, d in zip(pairs.query_id, pairs.doc_id)})
    rows, info = compose.virtual_base(tmp_path, "t", docsel, None)
    assert info["carriers"] == 2 and info["skipped_no_source"] == 1 and info["skipped_examples"] == [[1, 12]]
    assert [r["relevant_docs"] for r in rows] == [[10, 11, 12, 13], [20, 21]]
    assert [[c["doc_id"] for c in r["relevant_chunks"]] for r in rows] == [[10], [20]]
    docsel_path = tmp_path / "docsel.parquet"
    docsel.to_parquet(docsel_path)
    prep = {"base_zip": None, "base_parquet": info["base_parquet"], "base_parquet_sha256": info["base_parquet_sha256"],
            "docsel": str(docsel_path), "docsel_sha256": compose._sha256(docsel_path)}
    assert compose.load_base(prep) == rows
    docsel_path.write_bytes(docsel_path.read_bytes() + b"x")
    try:
        compose.load_base(prep)
    except ValueError:
        pass
    else:
        raise AssertionError("a changed docsel must be refused")


def make_segment(tmp_path):
    store = tmp_path / "store"
    seg = store / "seg1"
    seg.mkdir(parents=True)
    text = "Tieu de\n\nPhan mot cua bai.\n\nPhan hai cua bai."
    a, b = text.index("Phan mot"), text.index("Phan hai")
    pd.DataFrame({"doc_id": [900001], "lang": ["zh"], "doc_text": [text]}).to_parquet(seg / "docs.parquet")
    pd.DataFrame({"chunk_id": [1, 2, 3], "doc_id": [900001] * 3, "field": ["title", "body", "body"],
                  "char_start": [0, a, b], "char_end": [7, a + 17, len(text)], "text": [text[0:7], text[a:a + 17], text[b:]]}
                 ).to_parquet(seg / "chunks_t256.parquet")
    pd.DataFrame({"pair_id": [5, 5, 5], "rank": [1, 2, 3], "chunk_id": [3, 2, 1], "chunk_cosine": [0.9, 0.5, 0.1]}).to_parquet(seg / "cmax_top3.parquet")
    reg = tmp_path / "pairs_new.parquet"
    pd.DataFrame({"pair_id": [5], "query_id": [7], "doc_id": [900001], "segment": ["seg1"]}).to_parquet(reg)
    return store, reg, text


def test_registry_docs_and_best_chunk(tmp_path, monkeypatch):
    store, reg, text = make_segment(tmp_path)
    monkeypatch.setattr(registry, "STORE", store)
    monkeypatch.setattr(registry, "REGISTRY", reg)
    index = registry.registry_index()
    assert index == {900001: "seg1"} and registry.registry_index(tmp_path / "missing.parquet") == {}
    docs = registry.load_docs(index, [900001, 5])
    assert docs.doc_text.tolist() == [text]
    spans = registry.body_chunk_spans(index, [900001])
    assert compose._body_span(text, "vi", spans[900001]) == (text.index("Phan mot"), len(text))      # the body span, title excluded
    pairs = pd.DataFrame({"query_id": [7], "doc_id": [900001]})
    # no chunk score: the rank-1 chunk by chunk cosine of cmax_top3 (chunk 3)
    best, pieces = registry.best_spans(pairs, {900001: text}, index, choose, scores={})
    assert best[(7, 900001)] == (text.index("Phan hai"), len(text)) and list(pieces) == [900001]
    # a 0.6B chunk score wins over the cosine
    best, _ = registry.best_spans(pairs, {900001: text}, index, choose, scores={(7, 900001, 2): 0.8, (7, 900001, 3): 0.3})
    assert best[(7, 900001)][0] == text.index("Phan mot")
    # chunk spans that do not index the text: the doc is left out
    best, pieces = registry.best_spans(pairs, {900001: "x" + text}, index, choose, scores={})
    assert best == {} and pieces == {}


def test_registry_chunk_scores_read_every_cmax_run(tmp_path):
    for name, pair_id, doc, score in (("cmax", 1, 30, 0.7), ("cmax-b0004", 2, 31, 0.4)):
        shard = tmp_path / name / "pool" / "shards"
        shard.mkdir(parents=True)
        pd.DataFrame({"pair_id": [pair_id, pair_id + 10], "query_id": [3, 3], "doc_id": [doc, doc + 100],
                      "chunk_id": [6.0, None]}).to_parquet(shard / registry.SHARD)
        (tmp_path / name / registry.SCORES).parent.mkdir(parents=True)
        pd.DataFrame({"pair_id": [pair_id, pair_id + 10], "score": [score, 0.1]}).to_parquet(tmp_path / name / registry.SCORES)
    # the doc_text fallback rows have no chunk and are left out
    assert registry.chunk_scores(tmp_path) == {(3, 30, 6): 0.7, (3, 31, 6): 0.4}


def test_t_for_count_follows_the_body_token_quantiles():
    bt = np.array([1600, 1700, 1800, 1900, 2000])
    assert compose.t_for_count(bt, 2) == 1800 and compose.t_for_count(bt, 5) == compose.FULL_MIN_T and compose.t_for_count(bt, 0) == 2000
    assert compose.t_for_count(np.array([1540, 1550]), 1) == 1540 and compose.t_for_count(np.array([1500]), 0) == compose.FULL_MIN_T


def fake_write(tmp_path, name, limit, zip_of):
    """A stand-in for the ``write`` subprocess: the ZIP size is a function of T, the ceiling is ``limit``."""
    class Proc:
        def __init__(self, cmd, **_):
            t = int(cmd[cmd.index("--threshold") + 1])
            z = tmp_path / f"{name}-t{t}.zip"
            z.write_bytes(b"z")
            size = zip_of(t)
            entry = {"zip": str(z), "zip_bytes": size, "zip_sha256": compose._sha256(z), "limit_bytes": limit, "under_ceiling": size < limit,
                     "second_chunks": 3535 - t, "chunks": 0, "validator": {"ok": True}}
            (tmp_path / f"{name}-t{t}.compose.json").write_text(json.dumps(entry), encoding="utf-8")
            self.returncode = 0

        def communicate(self):
            return b"", b""
    return Proc


def test_run_full_reestimates_from_the_slack_when_every_candidate_is_under(tmp_path, monkeypatch):
    name, limit = "n", 1_000_000
    (tmp_path / f"{name}.prepare.json").write_text("{}", encoding="utf-8")
    bt = np.arange(1536, 3536)                                   # windows at T = #(bt > T) = 3535 - T
    zip_of = lambda t: 900_000 + int((3535 - t) * compose.RATE_FULL)           # lower T -> more windows -> bigger ZIP
    monkeypatch.setattr(compose.subprocess, "Popen", fake_write(tmp_path, name, limit, zip_of))
    res = compose.run_full(tmp_path, name, [3000, 3100], 4, bt=bt)
    assert res["lowest_t_under_ceiling"] is not None and res["exact_to_one_token"]
    best = res["lowest_t_under_ceiling"]
    assert zip_of(best) < limit <= zip_of(best - 1)


def test_run_full_reuses_a_fresh_measurement_and_ignores_a_changed_zip(tmp_path, monkeypatch):
    name = "n"
    (tmp_path / f"{name}.prepare.json").write_text("{}", encoding="utf-8")
    calls = []
    base = fake_write(tmp_path, name, 10, lambda t: 5)

    class Counting(base):
        def __init__(self, cmd, **kw):
            calls.append(cmd[cmd.index("--threshold") + 1])
            super().__init__(cmd, **kw)

    monkeypatch.setattr(compose.subprocess, "Popen", Counting)
    compose.run_full(tmp_path, name, [1600], 1)
    assert calls == ["1600"]
    compose.run_full(tmp_path, name, [1600], 1)                  # second run: reused
    assert calls == ["1600"]
    (tmp_path / f"{name}-t1600.zip").write_bytes(b"changed")
    compose.run_full(tmp_path, name, [1600], 1)                  # the ZIP no longer matches its measurement
    assert calls == ["1600", "1600"]


def fake_fit_write(tmp_path, name, zip_of):
    """Stand-in for ``write --threshold NO_WINDOWS --drop N``: the ZIP without windows shrinks 580 B per dropped carrier."""
    class Proc:
        def __init__(self, cmd, **_):
            n = int(cmd[cmd.index("--drop") + 1])
            label = compose.label_of(name, n, compose.NO_WINDOWS)
            z = tmp_path / f"{label}.zip"
            z.write_bytes(b"z")
            (tmp_path / f"{label}.compose.json").write_text(json.dumps({"zip_bytes": zip_of(n), "zip_sha256": compose._sha256(z)}), encoding="utf-8")
            self.returncode = 0

        def communicate(self):
            return b"", b""
    return Proc


def test_fit_carriers_finds_the_smallest_drop_under_the_ceiling(tmp_path, monkeypatch):
    name, limit = "n", compose.BTC_CEILING - compose.RESERVE
    (tmp_path / f"{name}.prepare.json").write_text("{}", encoding="utf-8")
    base = limit + 2_000_000
    zip_of = lambda n: int(base - n * 612.0)                                    # the real slope is not the RATE_CARRIER estimate
    monkeypatch.setattr(compose.subprocess, "Popen", fake_fit_write(tmp_path, name, zip_of))
    fit = compose.fit_carriers(tmp_path, name, base, compose.RESERVE, 6, 100_000, step=10)
    assert zip_of(fit["drop"]) < limit and fit["drop"] - fit["highest_drop_over"] <= 10 and fit["highest_drop_over"] is not None
    assert fit["zip_bytes_without_windows"] == zip_of(fit["drop"])
    # nothing to drop when the estimate and the probe say the base fits
    (tmp_path / "again").mkdir()
    (tmp_path / "again" / f"{name}.prepare.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(compose.subprocess, "Popen", fake_fit_write(tmp_path / "again", name, lambda n: limit - 1000))
    assert compose.fit_carriers(tmp_path / "again", name, limit - 1000, compose.RESERVE, 3, 100_000)["drop"] == 0


def test_lowest_p_carriers_and_without(tmp_path):
    docsel = pd.DataFrame({"query_id": [1, 1, 2, 2], "doc_id": [10, 11, 20, 21], "rank": [1, 2, 1, 2], "p4b": [0.9, 0.3, 0.3, 0.8]})
    path = tmp_path / "docsel.parquet"
    docsel.to_parquet(path)
    frame = pd.DataFrame({"query_id": [1, 1, 2, 2, 1], "doc_id": [10, 11, 20, 21, 11], "rank": [1, 1, 1, 1, 2]})
    assert compose.lowest_p_carriers({"docsel": str(path)}, frame, 1) == {(2, 20)}             # the tie at 0.3 goes to the later query
    assert compose.lowest_p_carriers({"docsel": str(path)}, frame, 0) == set()
    assert compose.lowest_p_carriers({"docsel": str(path)}, frame, 2) == {(2, 20), (1, 11)}
    kept = compose.without(frame, {(1, 11)})
    assert len(kept) == 3 and (kept[["query_id", "doc_id"]].values.tolist().count([1, 11]) == 0)           # rank 1 and rank 2 rows both go
    assert compose.label_of("n", 0, 1700) == "n-t1700" and compose.label_of("n", 40, 1700) == "n-d40-t1700" and compose.label_of("n", 40, None) == "n-d40"
