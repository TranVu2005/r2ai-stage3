"""Deterministic local site used by the crash / offline integration tests.

Pages /p/<n>: article (n % 23 == 5 -> 404, n % 31 == 7 -> soft-404 text redirect-free 'thin' page).
/robots.txt, /json (egress country), /ping (connectivity). `offline_until` makes every connection die mid-request."""
from __future__ import annotations

from r2ai.paths import ROOT

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def article(n: int) -> str:
    paras = ''.join(f'<p>Đoạn {i} của bài viết số {n}: điều trị bệnh nhân bằng phương pháp hiện đại, theo dõi sát tình trạng sức khỏe.</p>' for i in range(8))
    return f'<html><head><title>Bài viết {n}</title></head><body><nav><a href="/">Trang chủ</a></nav><article><h1>Bài viết {n}</h1>{paras}</article></body></html>'


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _offline(self):
        srv = self.server
        return time.time() < srv.offline_until

    def do_GET(self):
        srv = self.server
        with srv.lock:
            srv.hits += 1
            srv.paths.append(self.path)
            srv.hosts.append((self.headers.get('Host', '').split(':')[0], self.path))
        if self._offline():
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            self.close_connection = True
            return
        if srv.delay:
            time.sleep(srv.delay)
        path = self.path.split('?')[0]
        if path == '/json':
            return self._send(200, json.dumps({'ip': '1.2.3.4', 'country': srv.country}), 'application/json')
        if path == '/ping':
            return self._send(200, 'pong', 'text/plain')
        if path == '/robots.txt':
            return self._send(200, 'User-agent: *\nDisallow: /private\n', 'text/plain')
        if path.startswith('/d/'):
            return self._send(503, '<html><title>Service Unavailable</title></html>')
        if path.startswith('/c/'):
            if 'D1N=abc123' not in (self.headers.get('Cookie') or ''):
                return self._send(200, '<script>document.cookie="D1N=abc123"+"; path=/";window.location.reload(true);</script>')
            return self._send(200, article(int(path[3:])))
        if path.startswith('/p/'):
            n = int(path[3:])
            if n % 23 == 5:
                return self._send(404, '<html><title>404</title><body>Không tìm thấy trang</body></html>')
            if n % 31 == 7:
                return self._send(200, '<html><title>Ngắn</title><body><p>Quá ngắn.</p></body></html>')
            return self._send(200, article(n))
        self._send(404, 'nope', 'text/plain')

    def _send(self, code, body, ctype='text/html; charset=utf-8'):
        data = body.encode('utf-8')
        try:
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            pass


class MockServer:
    def __init__(self):
        self.srv = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
        self.srv.daemon_threads = True
        self.srv.offline_until = 0.0
        self.srv.country = 'VN'
        self.srv.delay = 0.0
        self.srv.hits = 0
        self.srv.paths = []
        self.srv.hosts = []
        self.srv.lock = threading.Lock()
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()

    def offline_for(self, seconds: float):
        self.srv.offline_until = time.time() + seconds

    @property
    def hits(self):
        return self.srv.hits
