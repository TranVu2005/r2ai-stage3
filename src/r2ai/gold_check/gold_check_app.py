"""Single-file Flask UI: python gold_check_app.py, then localhost:8765."""
from __future__ import annotations

from r2ai.paths import auxiliary_disabled

import argparse
import json
import secrets
import sys
import threading
from pathlib import Path


from flask import Flask, jsonify, render_template_string, request
from r2ai.gold_check.gold_check_common import (OUT, FOUND, PAGE_TYPES, CorpusIndex, ResultsStore,
                               data_path, domain, normalize_url, load_tokenizer,
                               make_fetcher, timestamp, validate_url, verify_url)

HTML = r'''<!doctype html>
<html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gold nằm ở đâu?</title>
<style>
:root{font:16px/1.6 system-ui,sans-serif;color:#142b3b;background:#f0f4f7}body{max-width:1000px;margin:auto;padding:24px}h1{font-size:25px}header,nav,.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}header{justify-content:space-between}.card{background:white;border:1px solid #d1dce4;border-radius:12px;padding:22px;margin:16px 0}button,.link{font:inherit;border:1px solid #9aafbd;border-radius:6px;padding:6px 12px;color:#173c53;background:#f7fafc;cursor:pointer;text-decoration:none}button.primary{background:#12604e;color:white;border-color:#12604e}button:disabled{opacity:.55;cursor:wait}a{color:#12604e}label{display:block;font-weight:600;margin:14px 0 5px}textarea,select{box-sizing:border-box;font:inherit;border:1px solid #a0b4c2;border-radius:6px;padding:9px;width:100%;background:white}textarea{resize:vertical}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}progress{width:240px;accent-color:#12604e}small,.muted{color:#536b7b}.phrase{white-space:pre-wrap;overflow-wrap:anywhere;font-weight:600}.result{border-top:1px solid #cbd8e1;padding:12px 0;overflow-wrap:anywhere}.error{color:#a32126}.good{color:#12604e}.status{font-weight:600}#message{white-space:pre-wrap}#query-text{font-size:18px}button:focus-visible,a:focus-visible,textarea:focus-visible,select:focus-visible{outline:3px solid #b47517;outline-offset:3px}
</style>
<header><h1>Gold nằm ở đâu?</h1><div><div id="progress-count">Đã ghi {{ completed }}/{{ total }} query</div><progress id="progress" value="{{ completed }}" max="{{ total }}"></progress></div></header>
<nav><a class="link" href="/?i={{ [0, position-1]|max }}">← Trước</a><span>Query {{ position+1 }}/{{ total }}</span><a class="link" href="/?i={{ [total-1, position+1]|min }}">Sau →</a><button id="report-button" type="button">Xuất báo cáo</button><span id="report-status" role="status"></span></nav>
<section class="card"><div class="muted">ID {{ task.id }} · {{ task.word_count }} từ · {{ task.stratum }}</div><pre id="query-text">{{ task.query }}</pre></section>
<section class="card"><h2>Chuỗi tìm kiếm</h2><p class="muted">Bạn mở link và tìm thủ công. App chỉ fetch URL bạn dán.</p>
{% for search in task.searches %}<div class="result"><div class="phrase">{{ search.phrase }}</div><div class="row"><button class="copy" data-phrase="{{ search.phrase }}">Copy</button>{% for engine,url in search.links.items() %}<a class="link" href="{{ url }}" target="_blank" rel="noopener noreferrer">{{ engine }}</a>{% endfor %}</div></div>{% endfor %}</section>
<section class="card"><form id="annotation">
<label for="urls">URL (mỗi dòng một URL)</label><textarea id="urls" rows="4" placeholder="https://...">{{ urls }}</textarea>
<div class="row"><div style="flex:1"><label for="found">Found</label><select id="found">{% for f in found_options %}<option {% if f==saved_found %}selected{% endif %}>{{ f }}</option>{% endfor %}</select></div><div style="flex:1"><label for="page-type">Page type</label><select id="page-type">{% for f in page_types %}<option {% if f==saved_type %}selected{% endif %}>{{ f }}</option>{% endfor %}</select></div></div>
<label for="notes">Notes</label><textarea id="notes" rows="3">{{ saved_notes }}</textarea><div class="row" style="margin-top:18px"><button class="primary" id="save" type="submit">Lưu & xác minh</button><button id="save-next" type="button">Lưu & tiếp →</button><small>Ctrl+Enter: lưu & tiếp</small></div></form>
<p id="message" role="status" aria-live="polite"></p><p class="muted">Sau khi lưu lần đầu, có thể chỉnh nhãn riêng cho từng URL bên dưới rồi lưu lại. Nhãn query dùng cho thống kê query; nhãn URL dùng cho thống kê domain và đối chiếu máy.</p><div id="results"></div></section>
<script>
const queryId={{ task.id|tojson }}, csrf={{ csrf|tojson }}, nextUrl={{ next_url|tojson }};
const originalRows={{ saved_rows|tojson }};
let busy=false;
function node(tag,text,cls){const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;return n}
function bool(x){return x===true || String(x).toLowerCase()==='true'}
function urlSelect(r,key,options,value){const s=node('select','','url-'+key);s.disabled=busy;s.dataset.url=r.url;s.dataset.override=String(value!==(key==='found'?r.found:r.page_type));s.setAttribute('aria-label','URL '+key+': '+r.url);for(const v of options){const o=node('option',v);o.value=v;s.append(o)}s.value=value;s.addEventListener('change',()=>s.dataset.override='true');return s}
function display(rows){const root=document.querySelector('#results');root.replaceChildren();for(const r of rows){if(!r.url)continue;const d=node('div','','result');d.append(node('strong',r.url));const labels=node('div','','row');labels.append(node('span','Nhãn URL:'),urlSelect(r,'found',['verbatim','partial','paraphrase','none'],r.url_found||r.found),urlSelect(r,'page-type',['hỏi đáp','bài viết','khác'],r.url_page_type||r.page_type));d.append(labels);d.append(node('div',(bool(r.in_corpus)?'Trong corpus: ':'Ngoài corpus. ID: ')+JSON.stringify(r.doc_ids),bool(r.in_corpus)?'good':''));if(r.suggested_doc_ids?.length && String(r.suggested_doc_ids)!=='[]')d.append(node('div','Gợi ý bỏ query string (KHÔNG tính in_corpus): '+JSON.stringify(r.suggested_doc_ids)));d.append(node('div','Xác minh: '+r.verification_status));if(r.verification_status==='ok'){d.append(node('div','Verbatim: '+r.verbatim_hit+' · LCS: '+Number(r.lcs_ratio).toFixed(4)+' · tokenizer: '+r.tokenizer));d.append(node('div','Khớp: '+r.match_kind+' · ký tự ['+r.match_start+', '+r.match_end+') trong file text'));d.append(node('pre','300 ký tự ngay sau đoạn khớp:\n'+(r.after_match||'')));}if(r.final_url&&r.final_url!==r.url)d.append(node('div','Redirect: '+r.final_url+' · corpus cuối: '+r.final_in_corpus));if(r.error)d.append(node('div',r.error,'error'));root.append(d)}}
display(originalRows);
async function post(path,data){const res=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Gold-CSRF':csrf},body:JSON.stringify(data)});const value=await res.json();if(!res.ok)throw Error(value.error||res.statusText);return value}
function urlAnnotations(){const a={};document.querySelectorAll('.url-found,.url-page-type').forEach(s=>{if(!a[s.dataset.url])a[s.dataset.url]={};a[s.dataset.url][s.classList.contains('url-found')?'found':'page_type']=s.value});return a}
async function save(next){if(busy)return;busy=true;document.querySelectorAll('#annotation textarea,#annotation select,#results select,#save,#save-next').forEach(b=>b.disabled=true);const msg=document.querySelector('#message');try{msg.textContent='Đang lưu nhãn và tra corpus…';const r=await post('/api/save',{query_id:queryId,urls:document.querySelector('#urls').value,found:document.querySelector('#found').value,page_type:document.querySelector('#page-type').value,notes:document.querySelector('#notes').value,url_annotations:urlAnnotations()});display(r.rows);document.querySelector('#progress').value=r.completed;document.querySelector('#progress-count').textContent='Đã ghi '+r.completed+'/{{ total }} query';msg.textContent='Đã lưu nhãn. Đang fetch và xác minh từng URL…';const v=await post('/api/verify',{query_id:queryId});display(v.rows);msg.textContent='Đã lưu kết quả. '+(v.rows.some(r=>r.verification_status==='error')?'Có lỗi xác minh; nhãn đã được giữ.':'');if(next)location.href=nextUrl;}catch(e){msg.textContent=e.message;}finally{busy=false;document.querySelectorAll('#annotation textarea,#annotation select,#results select,#save,#save-next').forEach(b=>b.disabled=false)}}
for(const [id,cls] of [['found','url-found'],['page-type','url-page-type']])document.querySelector('#'+id).addEventListener('change',e=>document.querySelectorAll('.'+cls).forEach(s=>{if(s.dataset.override!=='true')s.value=e.target.value}));
document.querySelector('#annotation').addEventListener('submit',e=>{e.preventDefault();save(false)});document.querySelector('#save-next').addEventListener('click',()=>save(true));document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key==='Enter'){e.preventDefault();save(true)}});
document.querySelectorAll('.copy').forEach(b=>b.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(b.dataset.phrase);b.textContent='Đã copy';}catch(e){document.querySelector('#message').textContent='Không copy được; chọn chuỗi rồi Ctrl+C.';}}));
document.querySelector('#report-button').addEventListener('click',async()=>{try{const r=await post('/api/report',{});document.querySelector('#report-status').textContent='Đã xuất: '+r.path;}catch(e){document.querySelector('#report-status').textContent=e.message;}});
window.addEventListener('beforeunload',e=>{if(busy){e.preventDefault();e.returnValue=''}});
</script></html>'''


def create_app(tasks_path=OUT / 'gold_check_tasks.json', results_path=OUT / 'gold_check_results.csv', corpus_path=None, index_path=OUT / 'gold_check_corpus.sqlite', pages_path=OUT / 'gold_check_pages', tokenizer=None, fetcher=None):
    tasks_path = Path(tasks_path)
    payload = json.loads(tasks_path.read_text(encoding='utf-8'))
    tasks = payload['tasks'] if isinstance(payload, dict) else payload
    if not tasks:
        raise ValueError('Tasks rỗng')
    by_id = {str(t['id']): t for t in tasks}
    index = CorpusIndex(corpus_path or data_path('links_corpus.parquet'), index_path)
    store = ResultsStore(results_path)
    fetcher = fetcher or make_fetcher()
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024
    csrf = secrets.token_urlsafe(32)
    operation_lock = threading.Lock()
    tokenizer_box = [tokenizer]
    app.extensions['gold_store'] = store
    app.extensions['gold_index'] = index
    app.extensions['gold_csrf'] = csrf

    @app.before_request
    def guard():
        # Bind loopback and reject cross-origin mutation / DNS rebinding.
        if request.host.split(':')[0] not in ('127.0.0.1', 'localhost', '[::1]'):
            return jsonify(error='Localhost only'), 403
        if request.method == 'POST' and request.headers.get('X-Gold-CSRF') != csrf:
            return jsonify(error='Invalid local session'), 403

    @app.get('/')
    def page():
        try:
            position = max(0, min(len(tasks) - 1, int(request.args.get('i', next((i for i, t in enumerate(tasks) if not store.for_query(t['id'])), len(tasks) - 1)))))
        except ValueError:
            position = 0
        task = tasks[position]
        rows = store.for_query(task['id'])
        r = rows[0] if rows else {}
        completed = sum(bool(store.for_query(t['id'])) for t in tasks)
        return render_template_string(HTML, task=task, position=position, total=len(tasks), completed=completed, urls='\n'.join(r['url'] for r in rows if r.get('url')), saved_found=r.get('found', 'none'), saved_type=r.get('page_type', 'hỏi đáp'), saved_notes=r.get('notes', ''), saved_rows=rows, found_options=FOUND, page_types=PAGE_TYPES, csrf=csrf, next_url='/?i=' + str(min(position + 1, len(tasks) - 1)))

    @app.post('/api/save')
    def save():
        data = request.get_json(silent=True) or {}
        task = by_id.get(str(data.get('query_id')))
        if task is None or data.get('found') not in FOUND or data.get('page_type') not in PAGE_TYPES:
            return jsonify(error='Query/nhãn không hợp lệ'), 400
        try:
            urls = list(dict.fromkeys(validate_url(u.strip()) for u in data.get('urls', '').splitlines() if u.strip()))
        except (ValueError, TypeError, AttributeError) as e:
            return jsonify(error=str(e)), 400
        if data['found'] != 'none' and not urls:
            return jsonify(error='Nhãn tìm thấy cần ít nhất một URL.'), 400
        if len(urls) > 30:
            return jsonify(error='Tối đa 30 URL/query.'), 400
        annotations = data.get('url_annotations') or {}
        if not isinstance(annotations, dict):
            return jsonify(error='Nhãn URL không hợp lệ'), 400
        for url in urls:
            item = annotations.get(url, {})
            if not isinstance(item, dict) or item.get('found', data['found']) not in FOUND or item.get('page_type', data['page_type']) not in PAGE_TYPES:
                return jsonify(error='Nhãn URL không hợp lệ'), 400
        if not operation_lock.acquire(blocking=False):
            return jsonify(error='Đang xác minh; chờ hoàn tất rồi lưu.'), 409
        try:
            rows = []
            for url in urls or ['']:
                row = dict(query_id=str(task['id']), query=task['query'], word_count=task['word_count'], stratum=task['stratum'], found=data['found'], page_type=data['page_type'], notes=str(data.get('notes', '')), url=url, url_norm=normalize_url(url) if url else '', domain=domain(url) if url else '', saved_at=timestamp(), verification_status='pending' if url else 'not_applicable')
                item = annotations.get(url, {})
                row.update(url_found=item.get('found', data['found']), url_page_type=item.get('page_type', data['page_type']))
                if url:
                    row.update(index.lookup(url))
                rows.append(row)
            store.replace_query(task['id'], rows)
            return jsonify(rows=rows, completed=sum(bool(store.for_query(t['id'])) for t in tasks))
        finally:
            operation_lock.release()

    @app.post('/api/verify')
    def verify():
        data = request.get_json(silent=True) or {}
        task = by_id.get(str(data.get('query_id')))
        if task is None:
            return jsonify(error='Query không hợp lệ'), 400
        if not operation_lock.acquire(blocking=False):
            return jsonify(error='Một lượt xác minh đang chạy'), 409
        try:
            rows = store.for_query(task['id'])
            for row in rows:
                if not row.get('url'):
                    continue
                try:
                    if tokenizer_box[0] is None:
                        tokenizer_box[0] = load_tokenizer()
                    row.update(verify_url(task, row['url'], index, fetcher, tokenizer_box[0], pages_path))
                except Exception as e:
                    row.update(verification_status='error', error=str(e))
                store.replace_query(task['id'], rows)  # checkpoint after every URL
            return jsonify(rows=rows)
        finally:
            operation_lock.release()

    @app.post('/api/report')
    def report():
        from r2ai.gold_check.gold_check_report import generate_report
        path = generate_report(tasks_path, results_path, index.domain_counts(), Path(results_path).parent / 'gold_check_report.md')
        return jsonify(path=str(path.resolve()))

    return app


def main():
    auxiliary_disabled()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--tasks', type=Path, default=OUT / 'gold_check_tasks.json')
    p.add_argument('--results', type=Path, default=OUT / 'gold_check_results.csv')
    p.add_argument('--corpus', type=Path, default=data_path('links_corpus.parquet'))
    p.add_argument('--index', type=Path, default=OUT / 'gold_check_corpus.sqlite')
    p.add_argument('--pages', type=Path, default=OUT / 'gold_check_pages')
    p.add_argument('--tokenizer', type=str)
    p.add_argument('--download-tokenizer', action='store_true')
    args = p.parse_args()
    if not args.tasks.exists():
        import pyarrow.parquet as pq
        from r2ai.gold_check.gold_check_sample import make_tasks
        from collections import Counter
        from r2ai.gold_check.gold_check_common import stratum
        queries = pq.read_table(data_path('query.parquet'), columns=['id', 'query']).to_pylist()
        args.tasks.parent.mkdir(parents=True, exist_ok=True)
        args.tasks.write_text(json.dumps(dict(seed=42, population=len(queries), population_strata=dict(Counter(stratum(len(r['query'].split())) for r in queries)), tasks=make_tasks(queries)), ensure_ascii=False, indent=2), encoding='utf-8')
    tokenizer = load_tokenizer(args.tokenizer, args.download_tokenizer)
    app = create_app(args.tasks, args.results, args.corpus, args.index, args.pages, tokenizer=tokenizer)
    print(f'Open http://127.0.0.1:{args.port}', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
