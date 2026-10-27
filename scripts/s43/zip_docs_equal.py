"""S43 check: ``relevant_docs`` of two submission ZIPs identical per query (same ids, same order), streamed (no full JSON load).

    python scripts/s43/zip_docs_equal.py A.zip B.zip
"""
import hashlib
import json
import re
import sys
import zipfile

PAT = re.compile(rb'"id":(\d+),"relevant_docs":\[([0-9,]*)\]')


def docs(path: str) -> dict[int, str]:
    out, buf = {}, b""
    with zipfile.ZipFile(path) as z, z.open(z.infolist()[0]) as f:
        while True:
            block = f.read(1 << 24)
            buf += block
            end = 0
            for m in PAT.finditer(buf):
                out[int(m.group(1))] = hashlib.sha256(m.group(2)).hexdigest()
                end = m.end()
            buf = buf[end:] if end else buf[-(1 << 22):]
            if not block:
                return out


def main() -> int:
    a, b = docs(sys.argv[1]), docs(sys.argv[2])
    diff = sorted(q for q in set(a) | set(b) if a.get(q) != b.get(q))
    res = {"queries_a": len(a), "queries_b": len(b), "queries_differing": len(diff), "first_differing": diff[:10],
           "equal": not diff and len(a) == len(b)}
    print(json.dumps(res))
    return 0 if res["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
