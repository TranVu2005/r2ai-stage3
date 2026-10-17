from r2ai.raw_recovery import core


def test_cjk_char_count_counts_han_unified_and_extension_ideographs():
    assert core.cjk_char_count("中A，病" + chr(0x20000)) == 3


def test_zh_threshold_is_50_han_and_preserves_existing_200_codepoint_docs():
    assert not core.good_text_for_lang("病" * 49, "zh")
    assert core.good_text_for_lang("病" * 50, "zh")
    assert core.good_text_for_lang("x" * 200, "zh")
    assert not core.good_text_for_lang("病" * 50, "vi")
    assert core.good_text_for_lang("x" * 200, "vi")


def test_chunker_minimum_tokens_is_language_specific():
    assert core.cc_min_body_tokens_for_lang("zh") == 1
    assert core.cc_min_body_tokens_for_lang("vi") == core.CC_MIN_BODY_TOKENS == 50


def test_doc_row_uses_language_threshold_without_downgrading_existing_long_text():
    from types import SimpleNamespace
    from r2ai.raw_recovery.pipeline import _doc_row

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1] * len(text)

    doc = SimpleNamespace(body="病" * 50, title="病", description="", question="", answer="",
                          paragraphs=["病" * 50], extractor="fixture", headings=[])
    record = {"doc_ids": [7], "url": "https://example.test", "domain": "example.test"}
    assert _doc_row(record, doc, "zh", {7}, Tokenizer())["status"] == "ok"
    assert _doc_row(record, doc, "vi", {7}, Tokenizer())["status"] == "thin"


def test_chunker_filter_accepts_zh_short_body_at_zh_minimum():
    import pyarrow as pa
    from r2ai.index.chunk import _keep_mask

    rows = pa.Table.from_pylist([
        {"doc_ids": [7], "status": "ok", "n_tokens_bge_m3": 12, "body": "病" * 50},
        {"doc_ids": [8], "status": "ok", "n_tokens_bge_m3": 12, "body": "病" * 50},
    ])
    mask = _keep_mask(rows, core.cc_min_body_tokens_for_lang("zh"))
    assert rows.filter(mask)["doc_ids"].to_pylist() == [[7], [8]]
    assert rows.filter(_keep_mask(rows, core.cc_min_body_tokens_for_lang("vi")))["doc_ids"].to_pylist() == []
