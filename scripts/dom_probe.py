from r2ai.paths import auxiliary_disabled

import sys, re, json, pathlib
from lxml import html as lh
def desc(e):
    c = (e.get('class') or '').strip().replace('\n',' ')[:60]
    return f"{e.tag}{'#'+e.get('id') if e.get('id') else ''}{'.'+c.replace(' ','.') if c else ''}"
def chain(e):
    out=[]
    while e is not None and e.tag not in ('html',):
        out.append(desc(e)); e=e.getparent()
    return ' < '.join(out[:7])
if __name__ == '__main__':
    auxiliary_disabled()
    for f in sys.argv[1:]:
        raw = pathlib.Path(f).read_bytes().decode('utf-8','replace')
        t = lh.fromstring(raw)
        print('=====', pathlib.Path(f).name, len(raw))
        print('title:', ' '.join(t.xpath('//title/text()'))[:100])
        print('h1:', [re.sub(r'\s+',' ',h.text_content())[:90] for h in t.xpath('//h1')][:3])
        print('og:desc:', (t.xpath('//meta[@property="og:description"]/@content') or t.xpath('//meta[@name="description"]/@content') or [''])[0][:100])
        ld = t.xpath('//script[@type="application/ld+json"]/text()')
        types=[]
        for s in ld:
            try:
                j=json.loads(s)
                for o in (j if isinstance(j,list) else j.get('@graph',[j])):
                    types.append(o.get('@type'))
            except Exception: pass
        print('ld types:', types)
        for n in t.xpath('//script|//style|//noscript'): n.drop_tree()
        # score containers by direct-ish paragraph text
        from collections import Counter
        score=Counter(); 
        for p in t.xpath('//p'):
            L=len(p.text_content().strip())
            if L<30: continue
            a=p.getparent(); depth=0
            while a is not None and depth<3:
                score[a]+=L; a=a.getparent(); depth+=1
        best = sorted(score.items(), key=lambda kv:-kv[1])[:4]
        for e,s in best: print(f'  {s:6d}  {chain(e)}')
