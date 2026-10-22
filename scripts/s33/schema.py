import pyarrow.parquet as pq, pandas as pd, sys
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90)
for f in sys.argv[1:]:
    pf = pq.ParquetFile(f); md = pf.metadata
    print('=====', f, 'rows', md.num_rows, 'row_groups', md.num_row_groups, 'created_by', md.created_by)
    print(pf.schema_arrow)
    kv = pf.schema_arrow.metadata or {}
    for k, v in kv.items(): print('META', k[:60], v[:400])
    df = pf.read().to_pandas()
    print(df.head(5).to_string()); print(df.tail(3).to_string())
    for c in df.columns:
        s = df[c]
        print(f'-- {c}: dtype {s.dtype}, nunique {s.nunique()}, null {s.isna().sum()}')
        if s.dtype == object: print('   len min/med/max', s.astype(str).str.len().describe()[['min','50%','max']].tolist())
        if s.nunique() < 30: print('   values', s.value_counts().to_dict())
