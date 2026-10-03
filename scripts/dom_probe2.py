from r2ai.paths import auxiliary_disabled

import sys, re
from lxml import html as lh
if __name__ == '__main__':
    auxiliary_disabled()
    f, sel = sys.argv[1], sys.argv[2]
    t = lh.fromstring(open(f, encoding='utf-8').read())
    for n in t.xpath('//script|//style|//noscript'): n.drop_tree()
    el = t.cssselect(sel)[0]
    print('##', f.split('/')[-1], sel, 'text', len(el.text_content()))
    for c in el:
        txt = re.sub(r'\s+', ' ', c.text_content()).strip()
        print(f"  <{c.tag} .{(c.get('class') or '')[:50]} #{c.get('id') or ''}> {txt[:70]!r} ({len(txt)})")
    sap = t.xpath('//*[contains(@class,"sapo") or contains(@class,"desc") or contains(@class,"lead") or contains(@class,"chapeau") or contains(@class,"summary") or contains(@class,"excerpt")]')
    for s in sap[:8]:
        txt = re.sub(r'\s+', ' ', s.text_content()).strip()
        if 40 < len(txt) < 600: print('  SAPO?', s.tag, s.get('class'), repr(txt[:80]))
