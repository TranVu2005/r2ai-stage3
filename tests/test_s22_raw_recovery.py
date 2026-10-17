from r2ai.raw_recovery import core


def test_cleaned_text_threshold_collapses_unicode_whitespace():
    assert core.cleaned_length("病" * 199 + "\u3000\n") == 199
    assert not core.good_text("病" * 199)
    assert core.good_text("病" * 200)


def test_s20_failure_uses_primary_doc_id_and_120ask_extract_ok_proxy():
    ids = {7}
    assert core.is_s20_failure(
        {"doc_ids": [7, 900], "body": "短", "status": "thin"},
        "120ask.com",
        ids,
    )
    assert not core.is_s20_failure(
        {"doc_ids": [7, 900], "body": "短", "status": "ok"},
        "120ask.com",
        ids,
    )
    assert core.is_s20_failure(
        {"doc_ids": [7, 900], "body": "病" * 199, "status": "ok"},
        "a-hospital.com",
        ids,
    )
    assert not core.is_s20_failure(
        {"doc_ids": [900, 7], "body": "病" * 199, "status": "thin"},
        "a-hospital.com",
        ids,
    )


def test_sample_is_stable_and_has_at_most_twenty_docs():
    rows = [{"doc_id": i} for i in range(40)]
    a = core.select_sample("a-hospital.com", rows, limit=20)
    b = core.select_sample("a-hospital.com", list(reversed(rows)), limit=20)
    assert len(a) == 20
    assert [r["doc_id"] for r in a] == [r["doc_id"] for r in b]


def test_s15_cc_reextract_provenance_uses_source_and_raw_shard_fields():
    row = core.provenance_row(
        doc_id=7,
        url="https://example.test/doc",
        raw_shard="data/raw_vi/example.test/20261017.jsonl.zst",
        status="ok",
        body="x" * 200,
        extractor="generic",
        extractor_name="generic",
        fetched_at=0,
        n_tokens=50,
    )
    assert row["source"] == "reextract-raw"
    assert row["raw_shard"] == "data/raw_vi/example.test/20261017.jsonl.zst"
    assert set(row) == {field.name for field in core.PROV_SCHEMA}


def test_batch_label_uses_s15_cc_prefix_and_domain():
    assert core.batch_label("ask.39.net") == "cc-s22-ask.39.net"


def test_ask39_nested_answer_paragraphs_remain_in_answer_field_in_order():
    from r2ai.zh_v2.extract import extract

    raw = """<html><head><title>症状咨询</title></head><body>
    <div class="ask_cont"><p>症状咨询</p><p>补充说明：持续两天</p><section class="sys-box">相关问题</section></div>
    <p class="sele_txt"><p>第一位医生回答。</p><p>建议观察。</p></p>
    <p class="sele_txt"><p>第二位医生回答。</p></p>
    </body></html>"""
    doc = extract(raw, "ask.39.net")
    assert doc.question == "补充说明：持续两天"
    assert doc.answer == "第一位医生回答。\n\n建议观察。\n\n第二位医生回答。"


def test_familydoctor_selector_drops_recommendation_and_copyright_boilerplate():
    from r2ai.zh_v2.extract import extract

    raw = """<html><body><h1>头痛的原因</h1><div id="viewContent">
    <p>第一段正文说明头痛的常见原因和处理办法。</p>
    <p>相关文章：更多相关疾病介绍</p>
    <p>家庭医生在线（www.familydoctor.com.cn）原创内容，内容合作请联系编辑部。</p>
    </div></body></html>"""
    doc = extract(raw, "familydoctor.com.cn")
    assert doc.body == "第一段正文说明头痛的常见原因和处理办法。"


def test_raw_diagnosis_signatures_do_not_treat_cloudflare_asset_as_a_block_page():
    assert not core.looks_blocked('<script src="https://static.cloudflare.net/site.js"></script>', 200)
    assert core.looks_blocked('<title>Just a moment...</title><div id="cf-chl-123"></div>', 200)
    assert core.encoding_suspect('文章正文�乱码')
    assert core.strip_null_bytes('有\x00\x00必要') == '有必要'
