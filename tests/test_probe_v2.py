from r2ai.paths import ROOT

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path


class ClassifierTests(unittest.TestCase):
    def test_fetcher_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('r2ai.probe.fetcher'))

    def setUp(self):
        if importlib.util.find_spec('r2ai.probe.fetcher') is None:
            self.skipTest('fetcher not implemented yet')
        from r2ai.probe.fetcher import classify
        self.classify = classify

    def test_cookie_precedes_spa(self):
        r = self.classify({'url': 'https://x.test/a', 'http_status': 200}, b'<script>document.cookie="D1N=abc123"+"; path=/";window.location.reload(true);</script>')
        self.assertEqual(r['state'], 'cookie_challenge')

    def test_502_and_521_are_dead(self):
        for code in (502, 521, 522, 523):
            self.assertEqual(self.classify({'http_status': code}, b'')['state'], 'dead_origin')

    def test_spa_only_when_explicit_and_empty(self):
        self.assertEqual(self.classify({'http_status': 200}, b'<div id="app"></div><script src="bundle.js"></script>')['state'], 'needs_js_real')
        self.assertEqual(self.classify({'http_status': 200}, b'<p>Short real article.</p>')['state'], 'thin')

    def test_challenge_beats_403_but_article_captcha_does_not(self):
        self.assertEqual(self.classify({'http_status': 403}, b'<title>Just a moment...</title><script src="/cdn-cgi/challenge-platform/a"></script>')['state'], 'bot_challenge')
        article = ('<html><title>Captcha treatment study</title><article><p>' + 'Medical study discusses captcha. ' * 60 + '</p></article></html>').encode()
        self.assertEqual(self.classify({'http_status': 200}, article)['state'], 'ok')

    def test_home_redirect_not_article(self):
        article = ('<title>Hospital home</title><p>' + 'Navigation news medicine. ' * 50 + '</p>').encode()
        self.assertEqual(self.classify({'url': 'https://x.test/article', 'final_url': 'https://x.test/', 'http_status': 200}, article)['state'], 'soft404_or_home')

    def test_article_with_recaptcha_widget_is_ok(self):
        article = ('<html><title>Medical evidence</title><body><article><p>' + 'Evidence for treatment in patients. ' * 25 + '</p></article><div class="g-recaptcha"></div></body></html>').encode()
        self.assertEqual(self.classify({'http_status': 200}, article)['state'], 'ok')

    def test_all_common_statuses_classified(self):
        for code, state in [(200, 'thin'), (204, 'thin'), (404, 'soft404_or_home'), (403, 'blocked_4xx'), (429, 'blocked_4xx'), (401, 'blocked_4xx'), (503, 'dead_origin')]:
            self.assertEqual(self.classify({'http_status': code}, b'')['state'], state)
        self.assertEqual(self.classify({'error': 'timeout'}, b'')['state'], 'network_error')


class InfrastructureTests(unittest.TestCase):
    def test_common_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('r2ai.probe.probe_common'))

    def test_normalization_preserves_query_and_trailing_slash(self):
        if importlib.util.find_spec('r2ai.probe.probe_common') is None:
            self.skipTest('not implemented')
        from r2ai.probe.probe_common import normalize_url
        self.assertEqual(normalize_url('http://WWW.X.TEST/A?q=1#top'), 'x.test/A?q=1')
        self.assertNotEqual(normalize_url('https://x.test/A?q=1'), normalize_url('https://x.test/A?q=2'))
        self.assertNotEqual(normalize_url('https://x.test/A/'), normalize_url('https://x.test/A'))

    def test_checkpoint_tolerates_only_torn_final_line(self):
        if importlib.util.find_spec('r2ai.probe.probe_common') is None:
            self.skipTest('not implemented')
        from r2ai.probe.probe_common import Checkpoint
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'rows.jsonl'
            p.write_text('{"key":"a","state":"ok"}\n{"key":', encoding='utf-8')
            c = Checkpoint(p)
            c.add({'key': 'b', 'state': 'thin'})
            self.assertEqual(set(Checkpoint(p).rows), {'a', 'b'})

    def test_archive_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('r2ai.probe.step1_archive_coverage'))

    def test_weighted_unmeasured_domains_are_unknown(self):
        from r2ai.probe.profile_report_v2 import weighted_coverage
        stats = [{'domain': 'a', 'n_urls': 80}, {'domain': 'b', 'n_urls': 20}]
        r = weighted_coverage(stats, {'a': [{'state': 'ok', 'expected_n': 2}, {'state': 'thin', 'expected_n': 2}]}, {}, {})
        self.assertEqual(r['direct'], '40.0%')
        self.assertEqual(r['unknown'], '60.0%')
        self.assertEqual(r['lost'], '0.0%')

    def test_partial_direct_sample_remains_unknown(self):
        from r2ai.probe.profile_report_v2 import weighted_coverage
        r = weighted_coverage([{'domain': 'a', 'n_urls': 100}], {'a': [{'state': 'ok'}]}, {}, {})
        self.assertEqual(r['direct'], '0.0%')
        self.assertEqual(r['unknown'], '100.0%')

    def test_html_robots_fallback_allows_but_bot_challenge_does_not(self):
        from r2ai.probe.fetcher import Fetcher
        import requests
        from unittest.mock import patch
        def get(session, url, **kwargs):
            r = requests.Response()
            r.url, r.status_code, r._content_consumed = url, 200, True
            r._content = b'<html><title>Hospital homepage</title><body>Welcome</body></html>' if 'fallback' in url else b'<html><title>Just a moment...</title><body>Verify you are human</body></html>'
            return r
        with patch.object(requests.Session, 'get', get):
            f = Fetcher()
            f.pacer.wait = lambda url: None
            self.assertTrue(f.allowed('https://fallback.test/a')[0])
            self.assertFalse(f.allowed('https://challenge.test/a')[0])

    def test_archive_payload_error_cannot_prove_loss(self):
        from r2ai.probe.profile_report_v2 import usable_archive
        self.assertIsNone(usable_archive([{'content_sample': True, 'cc_status': 'found', 'cc_content_status': 'error', 'wb_status': 'missing'}]))

    def test_partial_archive_cannot_prove_domain_loss(self):
        from r2ai.probe.profile_report_v2 import usable_archive
        self.assertIsNone(usable_archive([{'content_sample': True, 'cc_status': 'missing', 'wb_status': 'missing'}]))

    def test_recovered_archive_hit_invalidates_unavailable_content(self):
        from r2ai.probe.step1_archive_coverage import content_cache_valid
        self.assertFalse(content_cache_valid({'cc_status': 'found'}, 'cc', {'cc_content_status': 'not_available'}, False))

    def test_archive_retry_success_clears_previous_timeout(self):
        import requests
        from unittest.mock import patch
        from r2ai.probe.probe_common import Checkpoint
        from r2ai.probe.step1_archive_coverage import Service
        with tempfile.TemporaryDirectory() as d:
            s = Service('test', Checkpoint(Path(d) / 'api.jsonl'))
            s.pacer.wait = lambda url: None
            r = requests.Response()
            r.status_code, r._content, r._content_consumed = 200, b'[]', True
            with patch.object(s.session, 'get', side_effect=[requests.Timeout('slow'), r]), patch('r2ai.probe.step1_archive_coverage.time.sleep'):
                result = s.get('https://archive.test/cdx')
            self.assertNotIn('error', result)
            self.assertFalse(s.open)

    def test_warc_wrong_range_is_rejected(self):
        from r2ai.probe.step1_archive_coverage import cc_payload
        class Server:
            def get(self, *a, **kw):
                return {'http_status': 206, 'body': b'1234', 'response_headers': {'Content-Range': 'bytes 20-23/100'}}
        with self.assertRaisesRegex(ValueError, 'Content-Range'):
            cc_payload({'offset': '10', 'length': '4', 'filename': 'crawl-data/test.warc.gz', 'url': 'https://x.test/a'}, Server())

    def test_warc_wrong_target_is_rejected(self):
        from r2ai.probe.step1_archive_coverage import cc_payload
        from warcio.warcwriter import WARCWriter
        from warcio.statusandheaders import StatusAndHeaders
        buf = io.BytesIO()
        writer = WARCWriter(buf, gzip=True)
        record = writer.create_warc_record('https://x.test/wrong', 'response', payload=io.BytesIO(b'<p>wrong article</p>'), http_headers=StatusAndHeaders('200 OK', [('Content-Type', 'text/html')], protocol='HTTP/1.1'))
        writer.write_record(record)
        body = buf.getvalue()
        record.raw_stream.close()
        class Server:
            def get(self, *a, **kw):
                return {'http_status': 206, 'body': body, 'response_headers': {'Content-Range': f'bytes 10-{9+len(body)}/9999'}}
        with self.assertRaisesRegex(ValueError, 'target'):
            cc_payload({'offset': '10', 'length': str(len(body)), 'filename': 'crawl-data/test.warc.gz', 'url': 'https://x.test/a'}, Server())

    def test_archive_domain_join_ignores_transport_www_trailing_slash(self):
        from r2ai.probe.step1_archive_coverage import archive_key
        self.assertEqual(archive_key('http://www.x.test/a/?q=1#top'), archive_key('https://x.test/a?q=1'))
        self.assertNotEqual(archive_key('http://x.test/a?q=1'), archive_key('http://x.test/a?q=2'))

    def test_wilson_95_zero_hits_has_nonzero_upper_bound(self):
        from r2ai.probe.profile_report_v2 import wilson95
        lo, hi = wilson95(0, 50)
        self.assertAlmostEqual(lo, 0)
        self.assertAlmostEqual(hi, .0713476, places=5)

    def test_partial_domain_index_unmatched_url_is_unknown(self):
        from r2ai.probe.step1_archive_coverage import join_capture
        result = join_capture('https://x.test/article', 'cc', {}, False, 'page_limit')
        self.assertEqual(result['cc_status'], 'not_observed')
        self.assertFalse(result['cc_search_complete'])

    def test_domain_archive_one_hit_does_not_imply_full_recovery(self):
        from r2ai.probe.profile_report_v2 import archive_outcomes
        rows = [{'method':'domain_index_local_join', 'expected_index_n':50, 'expected_content_n':1, 'observed_hit_n':1, 'cc_status':'not_observed', 'wb_status':'not_observed', 'content_sample':False} for _ in range(50)]
        rows[0].update(wb_status='found', content_sample=True, wb_text_ok=True, wb_content_status='extracted')
        good, lost, unknown = archive_outcomes(rows)
        self.assertAlmostEqual(good, .02)
        self.assertAlmostEqual(lost, 0)
        self.assertAlmostEqual(unknown, .98)

    def test_archive_deadline_returns_even_if_transport_stalls(self):
        from r2ai.probe.step1_archive_coverage import Service
        from r2ai.probe.probe_common import Checkpoint
        import threading
        import time
        from unittest.mock import patch
        blocked = threading.Event()
        with tempfile.TemporaryDirectory() as d:
            s = Service('stall', Checkpoint(Path(d)/'api.jsonl'), deadline=time.monotonic()+.05)
            def transport(*args, **kwargs):
                blocked.wait(1)
                return {'http_status':200}
            start = time.monotonic()
            try:
                with patch.object(s, '_get_impl', transport):
                    r = s.get('https://archive.test/slow')
                self.assertEqual(r['error'], 'time_budget_exhausted')
                self.assertLess(time.monotonic()-start, .5)
            finally:
                blocked.set()

    def test_ua_suspected_requires_actual_pairs(self):
        from r2ai.probe.profile_report_v2 import diagnosis
        self.assertEqual(diagnosis([{'state': 'blocked_4xx', 'http_status': '403'}], [], []), 'chưa xác định')


class FetchTests(unittest.TestCase):
    def test_crawl_delay_uses_matching_agent_only(self):
        from r2ai.probe.fetcher import Fetcher
        import requests
        from unittest.mock import patch
        def get(session, url, **kwargs):
            r = requests.Response()
            r.url, r.status_code, r._content_consumed = url, 200, True
            r._content = b'User-agent: *\nAllow: /\n\nUser-agent: Yahoo! Slurp China\nCrawl-delay: 1000'
            return r
        with patch.object(requests.Session, 'get', get):
            f = Fetcher()
            f.pacer.wait = lambda url: None
            allowed, why, delay = f.allowed('https://x.test/a')
        self.assertTrue(allowed)
        self.assertIsNone(delay)

    def test_cookie_retry_once_and_session_persists(self):
        from r2ai.probe.fetcher import Fetcher
        import requests
        from unittest.mock import patch
        calls = []
        def get(session, url, **kwargs):
            calls.append(session.cookies.get('D1N'))
            r = requests.Response()
            r.status_code, r.url = 200, url
            r._content = b'<script>document.cookie="D1N=abc123";</script>' if len(calls) == 1 else ('<html><title>Medical article</title><body><article><p>' + 'Medical article body. ' * 60 + '</p></article></body></html>').encode()
            r._content_consumed = True
            return r
        with patch.object(requests.Session, 'get', get), patch('r2ai.probe.fetcher.time.sleep'):
            f = Fetcher(respect_robots=False)
            f.pacer.wait = lambda url: None
            rec, _ = f.fetch('https://x.test/a')
        self.assertEqual(calls, [None, 'abc123'])
        self.assertTrue(rec['cookie_solved'])
        self.assertEqual(rec['request_count'], 2)

    def test_5xx_retry_is_bounded(self):
        from r2ai.probe.fetcher import Fetcher
        import requests
        from unittest.mock import patch
        def get(session, url, **kwargs):
            r = requests.Response()
            r.status_code, r.url, r._content, r._content_consumed = 502, url, b'gateway unavailable', True
            return r
        with patch.object(requests.Session, 'get', get), patch('r2ai.probe.fetcher.time.sleep'):
            f = Fetcher(respect_robots=False)
            f.pacer.wait = lambda url: None
            rec, _ = f.fetch('https://x.test/a')
        self.assertEqual(rec['request_count'], 3)
        self.assertEqual(rec['retry_count'], 2)
        self.assertEqual(rec['state'], 'dead_origin')

    def test_redirect_destination_robots_checked(self):
        from r2ai.probe.fetcher import Fetcher
        import requests
        from unittest.mock import patch
        calls = []
        def get(session, url, **kwargs):
            calls.append(url)
            r = requests.Response()
            r.url, r._content_consumed = url, True
            if url.endswith('/robots.txt'):
                r.status_code = 200
                r._content = b'User-agent: *\nDisallow: /' if 'b.test' in url else b'User-agent: *\nAllow: /'
            else:
                r.status_code, r._content = 302, b''
                r.headers['Location'] = 'https://b.test/article'
            return r
        with patch.object(requests.Session, 'get', get):
            f = Fetcher()
            f.pacer.wait = lambda url: None
            rec, _ = f.fetch('https://a.test/article')
        self.assertEqual(calls, ['https://a.test/robots.txt', 'https://a.test/article', 'https://b.test/robots.txt'])
        self.assertEqual(rec['reason'], 'robots_disallowed')

    def test_cookie_challenge_on_robots_is_solved_before_policy(self):
        from r2ai.probe.fetcher import Fetcher
        import requests
        from unittest.mock import patch
        def get(session, url, **kwargs):
            r = requests.Response()
            r.url, r.status_code, r._content_consumed = url, 200, True
            if url.endswith('/robots.txt'):
                r._content = b'User-agent: *\nDisallow: /private' if session.cookies.get('D1N') else b'<script>document.cookie="D1N=abc123";window.location.reload();</script>'
            else:
                r._content = ('<html><title>Medical study</title><article><p>'+'Study evidence and treatment. '*60+'</p></article></html>').encode()
            return r
        with patch.object(requests.Session, 'get', get):
            f = Fetcher()
            f.pacer.wait = lambda url: None
            rec, _ = f.fetch('https://x.test/article')
        self.assertEqual(rec['state'], 'ok')
        self.assertEqual(rec['robots'], 'robots_ok')
        self.assertTrue(rec['robots_cookie_solved'])


if __name__ == '__main__':
    unittest.main()
