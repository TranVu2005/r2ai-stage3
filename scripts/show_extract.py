from r2ai.paths import FIXTURES_DIR, auxiliary_disabled

import sys, glob
from vicrawl.extractors import extract_doc
if __name__ == '__main__':
    auxiliary_disabled()
    name = sys.argv[1]
    files = sorted(glob.glob(str(FIXTURES_DIR / f'{sys.argv[2]}__*.html')))
    for f in files:
        d = extract_doc(open(f, encoding='utf-8').read(), name)
        print('=====', f.split('/')[-1], '| extractor', d.extractor, '| paras', len(d.paragraphs), '| chars', sum(map(len, d.paragraphs)))
        print('TITLE:', d.title); print('DESC :', d.description[:150]); print('HEADS:', d.headings[:5])
        if d.question: print('Q:', d.question[:200].replace('\n', ' // ')); print('A:', d.answer[:200].replace('\n', ' // '))
        for p in d.paragraphs[:3]: print('  +', p[:110])
        print('  ...')
        for p in d.paragraphs[-4:]: print('  -', p[:110])
