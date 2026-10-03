from r2ai.paths import ROOT

import tempfile
import unittest
import random
from pathlib import Path

from r2ai.gold_check.gold_check_common import normalize_url, normalize_text, match_excerpt, lcs_length, CorpusIndex
from r2ai.gold_check.gold_check_sample import make_tasks
from r2ai.gold_check.gold_check_report import wilson


class GoldCheckTests(unittest.TestCase):
    def test_normalization_preserves_query_and_path_case(self):
        self.assertEqual(normalize_url('HTTP://WWW.Example.COM/A/?x=1#z'), 'example.com/A?x=1')
        self.assertNotEqual(normalize_url('https://example.com/a?x=1'), normalize_url('https://example.com/a?x=2'))

    def test_match_offsets_are_original_text(self):
        text = 'Tiêu đề. Tôi bị sốt, 3 ngày? Bác sĩ trả lời.'
        result = match_excerpt('TÔI BỊ SỐT 3 NGÀY', text)
        self.assertTrue(result['verbatim_hit'])
        self.assertEqual(text[result['match_start']:result['match_end']], 'Tôi bị sốt, 3 ngày')
        self.assertTrue(result['after_match'].startswith('? Bác sĩ'))
        self.assertEqual(normalize_text('Sốt!'), 'sốt')

    def test_verbatim_in_title_is_not_lost_by_main_text_extraction(self):
        from r2ai.gold_check.gold_check_common import extracted_page_text
        body = ('<html><head><meta charset="utf-8"><title>Giảm đau hạch ở cổ như thế nào? | Vinmec</title></head><body><article><p>Hỏi: Ở cổ em bị nổi hạch rất đau.</p><p>' + ('Trả lời: Đây là phần thân bài không lặp lại câu hỏi ở tiêu đề. ' * 8) + '</p></article></body></html>').encode('utf-8')
        text, body_start = extracted_page_text(body)
        result = match_excerpt('Giảm đau hạch ở cổ như thế nào?', text)
        self.assertTrue(result['verbatim_hit'])
        self.assertLess(result['match_start'], body_start)

    def test_body_match_preferred_for_doctor_answer_excerpt(self):
        from r2ai.gold_check.gold_check_common import page_match
        title = 'Câu hỏi sốt | Site\n'
        body = 'Hỏi\nCâu hỏi sốt?\nTrả lời: câu trả lời bác sĩ.'
        result = page_match('Câu hỏi sốt', title + body, len(title))
        self.assertEqual(result['match_source'], 'body')
        self.assertGreaterEqual(result['match_start'], len(title))
        self.assertTrue(result['body_verbatim_hit'])
        self.assertIn('Trả lời:', result['after_match'])

    def test_faq_question_omitted_by_extractor_is_preserved_once(self):
        from unittest.mock import patch
        from r2ai.gold_check.gold_check_common import extracted_page_text
        question = 'Làm thế nào để trị dứt điểm bệnh hạ canxi?'
        body = ('<html><div class="faq-item"><div class="faq-item-desc">'+question+'</div><div class="faq-item-answer"><div class="faq-item-desc">Chào bạn, trả lời...</div></div></div></html>').encode('utf-8')
        with patch('r2ai.probe.fetcher.extract', return_value=('Chào bạn, trả lời...', 'Tư vấn online', '')):
            text, start = extracted_page_text(body)
        self.assertTrue(match_excerpt(question, text[start:])['verbatim_hit'])
        self.assertEqual(text.count(question), 1)

    def test_lcs_is_subsequence_not_contiguous(self):
        self.assertEqual(lcs_length([1, 2, 3], [8, 1, 9, 2, 3]), 3)
        self.assertEqual(lcs_length([1, 2, 1], [1, 1, 2]), 2)
        self.assertEqual(lcs_length([], [1]), 0)

    def test_stratification_and_exact_search(self):
        rows = [{'id': f'{n}-{i}', 'query': ' '.join(['Sốt'] * n)} for n in (8, 20, 45) for i in range(12)]
        tasks = make_tasks(rows)
        self.assertEqual(tasks, make_tasks(rows))
        self.assertEqual(len(tasks), 30)
        self.assertEqual([sum(t['stratum'] == s for t in tasks) for s in ('short', 'medium', 'long')], [10, 10, 10])
        for t in tasks:
            self.assertIn(len(t['searches']), (2, 3))
            for s in t['searches']:
                self.assertTrue(s['phrase'].startswith('"'))
                self.assertIn('%22', s['links']['google'])

    def test_index_groups_and_loose_is_only_suggestion(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            pq.write_table(pa.Table.from_pylist([
                {'id': 1, 'url': 'http://www.Example.com/a/?x=1'},
                {'id': 2, 'url': 'https://example.com/a?x=1'},
            ]), p / 'corpus.parquet')
            idx = CorpusIndex(p / 'corpus.parquet', p / 'index.sqlite')
            self.assertEqual(idx.lookup('https://example.com/a?x=1')['doc_ids'], ['1', '2'])
            r = idx.lookup('https://example.com/a?x=2')
            self.assertFalse(r['in_corpus'])
            self.assertEqual(r['suggested_doc_ids'], ['1', '2'])
            self.assertEqual(idx.domain_counts(), {'example.com': 2})
            self.assertEqual(CorpusIndex(p / 'corpus.parquet', p / 'index.sqlite').lookup('http://example.com/a?x=1')['doc_ids'], ['1', '2'])

    def test_wilson_boundaries(self):
        self.assertEqual(wilson(0, 0), None)
        lo, hi = wilson(0, 30)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, .11)
        self.assertEqual(wilson(30, 30)[1], 1)

    def test_lcs_matches_dynamic_program_reference(self):
        rng = random.Random(42)
        for _ in range(100):
            a = [rng.randrange(5) for _ in range(rng.randrange(20))]
            b = [rng.randrange(5) for _ in range(rng.randrange(40))]
            dp = [0] * (len(a) + 1)
            for t in b:
                old = dp[:]
                for i, q in enumerate(a, 1):
                    dp[i] = old[i-1] + 1 if t == q else max(old[i], dp[i-1])
            self.assertEqual(lcs_length(a, b), dp[-1])

    def test_search_domains_blocked_including_redirect(self):
        from r2ai.gold_check.gold_check_common import validate_url, make_fetcher
        for url in ('https://www.google.com/search?q=abc', 'https://google.co.uk/search', 'https://cn.bing.com', 'https://bing.com./search', 'https://coccoc.com./search', 'https://bing\u3002com/search', 'https://search.coccoc.com', 'file:///tmp/a', 'https://user:pass@example.com'):
            with self.assertRaises(ValueError):
                validate_url(url)
        with self.assertRaises(ValueError):
            make_fetcher().request('https://www.bing.com/search?q=abc')
        for url in ('https://%62ing.com/search', 'https://bing.com%2e/search', 'https://%63occoc.com/search'):
            with self.assertRaises(ValueError):
                validate_url(url)

    def test_long_search_avoids_generic_preface(self):
        from r2ai.gold_check.gold_check_sample import search_phrases
        q = 'bác sĩ cho em hỏi có sao không ạ em cảm ơn bác sĩ tôi xét nghiệm HBsAg 250 IU/ml Helicobacter pylori dương tính sau 14 ngày điều trị amoxicillin'
        phrases = search_phrases(q, {})
        self.assertGreaterEqual(len(phrases), 2)
        self.assertNotIn('bác sĩ cho em hỏi', phrases[0])
        self.assertTrue('HBsAg' in phrases[0] or '250' in phrases[0])

    def test_app_save_restart_and_report(self):
        import json
        import pyarrow as pa
        import pyarrow.parquet as pq
        from r2ai.gold_check.gold_check_app import create_app
        from r2ai.gold_check.gold_check_report import generate_report
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            task = dict(id='q1', query='Sốt ba ngày', word_count=3, stratum='short', searches=[])
            (p/'tasks.json').write_text(json.dumps({'tasks':[task]}), encoding='utf-8')
            pq.write_table(pa.Table.from_pylist([{'id':'d1', 'url':'https://example.com/a'}]), p/'corpus.parquet')
            def app():
                return create_app(p/'tasks.json', p/'results.csv', p/'corpus.parquet', p/'index.sqlite', p/'pages')
            a = app()
            client = a.test_client()
            headers = {'X-Gold-CSRF':a.extensions['gold_csrf']}
            self.assertEqual(client.get('/').status_code, 200)
            annotation = dict(query_id='q1', found='verbatim', page_type='hỏi đáp', urls='http://www.example.com/a/\nhttps://example.com/b', notes='review', url_annotations={'https://example.com/b':{'found':'partial','page_type':'bài viết'}})
            self.assertEqual(client.post('/api/save', json=annotation).status_code, 403)
            res = client.post('/api/save', json=annotation, headers=headers)
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.json['rows'][0]['in_corpus'])
            self.assertFalse(res.json['rows'][1]['in_corpus'])
            self.assertEqual(res.json['rows'][1]['url_found'], 'partial')
            self.assertEqual(res.json['rows'][1]['url_page_type'], 'bài viết')
            reopened = app()
            self.assertEqual(len(reopened.extensions['gold_store'].for_query('q1')), 2)
            report = generate_report(p/'tasks.json', p/'results.csv', {'example.com':1}, p/'report.md')
            content = report.read_text(encoding='utf-8')
            self.assertIn('1/2: 50.0%', content)
            self.assertIn('chưa đo/lỗi: 2/2', content)
            self.assertIn('0/1: 0.0%', content)
            annotation.update(found='none', urls='')
            self.assertEqual(client.post('/api/save', json=annotation, headers=headers).status_code, 200)
            self.assertEqual(len(app().extensions['gold_store'].for_query('q1')), 1)

    def test_mixed_urls_do_not_credit_partial_domain_with_gold(self):
        import json
        from r2ai.gold_check.gold_check_common import ResultsStore
        from r2ai.gold_check.gold_check_report import generate_report
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'tasks.json').write_text(json.dumps({'tasks':[dict(id='q', stratum='short')]}), encoding='utf-8')
            base = dict(query_id='q', found='verbatim', page_type='hỏi đáp', stratum='short', verification_status='ok', query='q')
            a = dict(base, url='https://gold.example/a', domain='gold.example', url_found='verbatim', in_corpus=False, verbatim_hit=True, lcs_ratio=1)
            b = dict(base, url='https://candidate.example/b', domain='candidate.example', url_found='partial', in_corpus=True, verbatim_hit=False, lcs_ratio=.9)
            ResultsStore(p/'results.csv').replace_query('q', [a,b])
            report = generate_report(p/'tasks.json', p/'results.csv', {'candidate.example':100}, p/'report.md').read_text(encoding='utf-8')
            self.assertIn('Query human-verbatim và có URL trong corpus: 0/1:', report)
            density = report.split('## Mật độ gold theo domain')[1].split('## Bảng quyết định')[0]
            self.assertIn('gold.example', density)
            self.assertNotIn('candidate.example', density)
            self.assertIn('Bất đồng/cần rà trên mọi URL đã đo: 0/2:', report)


if __name__ == '__main__':
    unittest.main()
