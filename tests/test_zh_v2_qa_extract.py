from r2ai.zh_v2 import qa_extract as q


def row(text):
    return {'doc_id': 1, 'url': 'u', 'doc_text': text}


def test_ask_pair_with_answer_is_ok_and_question_only_is_flagged():
    ok = '胃痛怎么办\n\n发病时间：不清楚\n\n2021-06-23 11:42\n\n胃痛 胃炎\n\n你好，建议先查幽门螺杆菌，注意饮食规律，少吃辛辣刺激食物。'
    assert q.check('ask.39.net', row(ok))['flags'] == []
    only_q = '胃痛怎么办\n\n发病时间：不清楚\n\n胃痛 胃炎 幽门螺杆菌 饮食 辛辣 刺激 检查 治疗 药物 症状 疼痛'
    assert 'qa_no_answer' in q.check('ask.39.net', row(only_q))['flags']


def test_section_markers_count_only_as_short_heading_lines():
    body = '正文' * 40 + '\n\n推荐阅读：\n\n' + '正文' * 40
    assert 'related_block' in q.check('heart.39.net', row(body))['flags']
    sentence = '这篇文章的相关文章很多，作者在文中多次提到相关文章与相关阅读的区别，建议读者仔细阅读全文内容并对照。' * 2
    assert q.check('heart.39.net', row(sentence * 2))['flags'] == []


def test_error_and_short_pages_are_flagged():
    flags = q.check('baby.39.net', row('404 页面不存在'))['flags']
    assert 'error_page' in flags and 'too_short' in flags
