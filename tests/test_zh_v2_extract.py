from r2ai.zh_v2 import extract as x

ASK = """<html><head><title>你好，乳房发胀_39问医生</title></head><body><h1>你好，乳房发胀</h1>
<div class="ask_cont"><p>你好，乳房发胀</p><p>发病时间：最近一个星期</p><p>补充说明：你好，乳房发胀</p>
<p>a******W 2012-11-30 19:44</p><p>乳房发胀 副乳 彩超</p><section class="sys-box"><div class="sys-label">其他人还在看 治疗人乳头留病毒效果好</div></section>
<p>我要咨询</p></div>
<div class="selected"><div class="sele_all"><p>尹文迪 医师 某某卫生所 擅长：临床全科</p><p class="sele_txt">病情分析：根据描述多见于乳痛症。</p></div>
<div class="sele_all"><p class="sele_txt">建议做乳腺彩超。</p></div></div>
<footer>声明：39问医生</footer></body></html>"""

ART = """<html><body><h1>小儿干咳当心支原体肺炎</h1><div class="art_wrap"><div class="art_content">
<p>在省妇幼儿保科就诊的朵朵今年5岁，近一周的时间一直有点干咳，家人以为是入冬孩子不适应天气变化。</p>
<h2>孩子咳嗽要先去痰</h2><p>秋冬气候变化大，寒冷的空气可引起呼吸道的黏液分泌增加，孩子很容易咳嗽并反复发作。</p>
<p>39健康网(www.39.net)专稿，未经书面授权请勿转载。</p><p>（实习编辑：刘茵莹）</p></div><div class="art_rec">相关推荐 很多链接</div></div></body></html>"""

FD = """<html><body><h1>儿童患风疹接种法</h1><div class="endContent"><div id="viewContent" class="viewContent">
<p>有时全身感到不适，有时颈部的淋巴结还会肿大。发烧2-3天即退。</p></div><div class="hotyy">热门医院 推荐文章</div></div></body></html>"""


def test_ask_39_keeps_question_answer_pairs_without_page_furniture():
    d = x.extract(ASK, 'ask.39.net')
    assert d.title == '你好，乳房发胀'
    assert d.question == '发病时间：最近一个星期\n\n乳房发胀 副乳 彩超'          # title repeats, asker line, button, related box dropped
    assert d.answer == '病情分析：根据描述多见于乳痛症。\n\n建议做乳腺彩超。'     # every selected answer, doctor line dropped
    assert d.paragraphs == [*d.question.split('\n\n'), *d.answer.split('\n\n')]
    assert d.extractor == 'zhv2_qa_ask.39.net' and x.doc_status(d.body, 'ask.39.net') == 'ok'


def test_39_net_article_keeps_only_art_content_and_drops_credits():
    d = x.extract(ART, 'baby.39.net')
    assert d.title == '小儿干咳当心支原体肺炎' and d.question == d.answer == ''
    assert len(d.paragraphs) == 3 and '专稿' not in d.body and '编辑' not in d.body and '相关推荐' not in d.body
    assert d.headings == ['孩子咳嗽要先去痰']


def test_familydoctor_uses_view_content_only():
    d = x.extract(FD, 'familydoctor.com.cn')
    assert d.paragraphs == ['有时全身感到不适，有时颈部的淋巴结还会肿大。发烧2-3天即退。']


def test_missing_container_and_other_hosts_fall_back_to_the_zh_full_extractor():
    from r2ai.zh_full_ingest import extract as base
    html = '<html><body><h1>标题</h1><div class="other"><p>' + '内容' * 200 + '</p></div></body></html>'
    assert x.extract(html, 'baby.39.net').body == base.extract(html, 'baby.39.net').body
    assert x.extract(html, '120ask.com').body == base.extract(html, '120ask.com').body


def test_version_changes_with_rules():
    assert len(x.extractor_version()) == 64
