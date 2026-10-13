# S9 — Corpus v2 migration

Nhánh `feat/corpus-v2` (worktree `D:/GitHub/r2ai-stage3-corpusv2`), base `main@c0d5c87` (không phải `483f4df`: Vu duyệt base này). Đo 13/10/2026 13:40–14:10 (UTC+7). Không fetch nội dung, không sửa pipeline retrieval, không nộp LB, chưa push.

## 0. Cần Vu quyết

1. **Số liệu lệch BTC.** File v2 có **1.220.598** URL, BTC thông báo 1.194.469 (lệch +26.129). vi thực là **740.598**, không phải 714.469; phần lệch đúng bằng 26.129 URL `vinmec.com/vie/…` (id cũ, URL cũ, nằm trong v1). zh 240.000 và en 240.000 khớp thông báo. Vu đã chọn dùng file như hiện có; cần xác nhận lại với BTC rằng vinmec thuộc corpus v2 (nếu bị loại thì 26.129 id vi đó thành "ngoài corpus").
2. **K80 LZMA không xuống dưới 99 MB nếu giữ cấu hình nén mặc định.** Nội dung K80 không có doc/chunk nào ngoài v2 nên JSON không đổi (SHA khớp bản gốc); ZIP_LZMA mặc định (từ điển 8 MiB) vẫn ra 99.035.081 B > 99.000.000. Bản `sub_K80_LZMA_v2.zip` dưới đây dùng **từ điển LZMA 64 MiB** → 78.734.571 B. Python `zipfile` và `tar.exe` (bsdtar của Windows) đều giải nén ra đúng SHA JSON; chưa biết reader phía BTC có chịu từ điển 64 MiB không (Python và libarchive thì chịu). Nếu cần bám cấu hình mặc định thì K80 không đạt "< 99 MB" mà không đổi nội dung. Bản mặc định giữ ở `out/runs/s9/extra/sub_K80_LZMA_v2_dict8.zip` để so sánh.
3. **Hai file K80.** Đã dùng `sub_K80_LZMA.zip` từ worktree upload-probe (Vu xác nhận). `out/runs/s6/S6-a-K80/S6_a_K80.zip` (101,13 MB, JSON 658.962.109 B) là file khác, chưa lọc.

## 1. Tải và kiểm file (bước 1)

| Mục | Giá trị |
|:--|:--|
| Dataset | `AIGuruTinix/ViBioMIR` (public, không gated), last_modified 2026-10-12 13:52:14 UTC |
| Revision hash | `71be0e1239a19dfd7a717e2bc13e3bec21bd0334` |
| File | `links_corpus_vi_en_zh.parquet`, 37.689.434 B, SHA256 `87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7` |
| Lưu tại | `data/raw/links_corpus_vi_en_zh.parquet` (ignored); `query.parquet` trên HF vẫn 81.826 B như v1 (chưa so hash) |
| Schema | `id: int64`, `url: string` — **không có cột lang** (parquet-cpp-arrow 17.0.0, không metadata) |
| Dòng | 1.220.598; `id` duy nhất, `url` duy nhất, không null |
| id min/max | 583 / 7.466.161 (v1: 583 / 4.420.561, 4.394.718 dòng) |

`lang` suy từ host bằng bảng tường minh trong `src/r2ai/corpus_v2.py` (en: `ncbi.nlm.nih.gov`; zh: 39.net, 120ask, cnkang, familydoctor, zysjonline, zhongyibaodian, a-hospital, youlai, pmphai, msdmanuals, iiyi, wujue, jb39, zydcd, people.com.cn, baidu, qihuangzhishu; vi: `.vn` và vinmec, hellobacsi, vnexpress.net, benhvienvietduc.org, pharmacity.io). Host lạ làm `build()` lỗi, không đoán. Kiểm chéo: zh = 240.000, en = 240.000, vi = 740.598 — khớp thông báo BTC cho en/zh, và vi trùng đúng số vi của v1 (740.598).

## 2. Diff v1 → v2 (bước 2)

v1 `links_corpus.parquet` (OLD, chỉ đọc) so với v2, theo `id`. Id cũ **không đổi URL** (0/980.598).

| Trạng thái | vi | zh | en | Tổng |
|:--|--:|--:|--:|--:|
| Giữ (id ∈ v1 ∩ v2) | 740.598 | 240.000 | 0 | 980.598 |
| Rút (id ∈ v1, ∉ v2) | 0 | 3.414.119 (+1 id `fk.99.com.cn`, host chưa có trong bảng nên chưa phân loại) | 0 | 3.414.120 |
| Mới (id ∉ v1) | 0 | 0 | 240.000 | 240.000 |
| v1 | 740.598 | 3.654.119 (+1) | 0 | 4.394.718 |
| v2 | 740.598 | 240.000 | 240.000 | 1.220.598 |

**(a) 240k en là ID mới hoàn toàn.** Không id nào có trong v1; dải 4.421.055–7.466.161 (v1 kết thúc ở 4.420.561); 100% host `pubmed.ncbi.nlm.nih.gov`, URL dạng `/<PMID>/`.

**(b) vi trùng 100% với v1.** 740.598/740.598 id vi của v2 nằm trong v1 với cùng URL, và v1 không có id vi nào bị rút. Con số 714.469 trong thông báo không khớp (xem mục 0.1).

zh bị rút mạnh: còn 6,6% (240.000/3.654.119). Bảng theo host (v1 / giữ / rút / mới / v2); bảng đầy đủ dạng CSV ở `out/runs/s9/diff_lang_host_status.csv`.

| lang | host | v1 | giữ | rút | mới | v2 |
|:--|:--|--:|--:|--:|--:|--:|
| en | pubmed.ncbi.nlm.nih.gov | 0 | 0 | 0 | 240000 | 240000 |
| unk | fk.99.com.cn | 1 | 0 | 1 | 0 | 0 |
| vi | suckhoecongdongonline.vn | 154503 | 154503 | 0 | 0 | 154503 |
| vi | suckhoedoisong.vn | 85823 | 85823 | 0 | 0 | 85823 |
| vi | nhathuoclongchau.com.vn | 79598 | 79598 | 0 | 0 | 79598 |
| vi | thanhnien.vn | 72111 | 72111 | 0 | 0 | 72111 |
| vi | laodong.vn | 31282 | 31282 | 0 | 0 | 31282 |
| vi | phunusuckhoe.giadinhonline.vn | 27577 | 27577 | 0 | 0 | 27577 |
| vi | vinmec.com | 26129 | 26129 | 0 | 0 | 26129 |
| vi | medlatec.vn | 24762 | 24762 | 0 | 0 | 24762 |
| vi | giadinhonline.vn | 16729 | 16729 | 0 | 0 | 16729 |
| vi | baohaiphong.vn | 16286 | 16286 | 0 | 0 | 16286 |
| vi | baonghean.vn | 15941 | 15941 | 0 | 0 | 15941 |
| vi | vietnamnet.vn | 15543 | 15543 | 0 | 0 | 15543 |
| vi | baodanang.vn | 14500 | 14500 | 0 | 0 | 14500 |
| vi | tiemchunglongchau.com.vn | 13281 | 13281 | 0 | 0 | 13281 |
| vi | baocantho.com.vn | 12016 | 12016 | 0 | 0 | 12016 |
| vi | hellobacsi.com | 11747 | 11747 | 0 | 0 | 11747 |
| vi | baoangiang.com.vn | 10738 | 10738 | 0 | 0 | 10738 |
| vi | vov.vn | 9784 | 9784 | 0 | 0 | 9784 |
| vi | khoahocphothong.vn | 8143 | 8143 | 0 | 0 | 8143 |
| vi | youmed.vn | 7189 | 7189 | 0 | 0 | 7189 |
| vi | vov2.vov.vn | 6899 | 6899 | 0 | 0 | 6899 |
| vi | phuyen.baodaklak.vn | 6252 | 6252 | 0 | 0 | 6252 |
| vi | suckhoeviet.org.vn | 5627 | 5627 | 0 | 0 | 5627 |
| vi | baoquangtri.vn | 5483 | 5483 | 0 | 0 | 5483 |
| vi | pmc-ecm-healthblog.beta.pharmacity.io | 5333 | 5333 | 0 | 0 | 5333 |
| vi | nhandan.vn | 5225 | 5225 | 0 | 0 | 5225 |
| vi | baovinhlong.com.vn | 5113 | 5113 | 0 | 0 | 5113 |
| vi | tamanhhospital.vn | 4261 | 4261 | 0 | 0 | 4261 |
| vi | tuoitre.vn | 3893 | 3893 | 0 | 0 | 3893 |
| vi | qdnd.vn | 3700 | 3700 | 0 | 0 | 3700 |
| vi | vietnamplus.vn | 3695 | 3695 | 0 | 0 | 3695 |
| vi | baogialai.com.vn | 3290 | 3290 | 0 | 0 | 3290 |
| vi | tienphong.vn | 3021 | 3021 | 0 | 0 | 3021 |
| vi | thaythuocvietnam.vn | 2927 | 2927 | 0 | 0 | 2927 |
| vi | baoquangninh.vn | 2912 | 2912 | 0 | 0 | 2912 |
| vi | bachmai.gov.vn | 2783 | 2783 | 0 | 0 | 2783 |
| vi | baochinhphu.vn | 2302 | 2302 | 0 | 0 | 2302 |
| vi | dantri.com.vn | 2063 | 2063 | 0 | 0 | 2063 |
| vi | vnexpress.net | 2055 | 2055 | 0 | 0 | 2055 |
| vi | baolangson.vn | 1981 | 1981 | 0 | 0 | 1981 |
| vi | sggp.org.vn | 1853 | 1853 | 0 | 0 | 1853 |
| vi | baophutho.vn | 1288 | 1288 | 0 | 0 | 1288 |
| vi | hanoimoi.vn | 1283 | 1283 | 0 | 0 | 1283 |
| vi | benhviennhitrunguong.gov.vn | 1200 | 1200 | 0 | 0 | 1200 |
| vi | baotayninh.vn | 1198 | 1198 | 0 | 0 | 1198 |
| vi | baothanhhoa.vn | 782 | 782 | 0 | 0 | 782 |
| vi | benhvienvietduc.org | 425 | 425 | 0 | 0 | 425 |
| vi | pharmacity.vn | 69 | 69 | 0 | 0 | 69 |
| vi | mega.vietnamplus.vn | 2 | 2 | 0 | 0 | 2 |
| vi | special.vietnamplus.vn | 1 | 1 | 0 | 0 | 1 |
| zh | cnkang.com | 963438 | 40046 | 923392 | 0 | 40046 |
| zh | 120ask.com | 918479 | 98160 | 820319 | 0 | 98160 |
| zh | familydoctor.com.cn | 447453 | 26819 | 420634 | 0 | 26819 |
| zh | ask.39.net | 337375 | 32772 | 304603 | 0 | 32772 |
| zh | zysjonline.com | 243347 | 6591 | 236756 | 0 | 6591 |
| zh | a-hospital.com | 169339 | 4701 | 164638 | 0 | 4701 |
| zh | zhongyibaodian.net | 146700 | 6061 | 140639 | 0 | 6061 |
| zh | zydcd.com | 78680 | 204 | 78476 | 0 | 204 |
| zh | wujue.com | 53341 | 708 | 52633 | 0 | 708 |
| zh | youlai.cn | 46119 | 4139 | 41980 | 0 | 4139 |
| zh | baby.39.net | 27621 | 3368 | 24253 | 0 | 3368 |
| zh | bingli.iiyi.com | 24990 | 5 | 24985 | 0 | 5 |
| zh | qihuangzhishu.com | 23637 | 29 | 23608 | 0 | 29 |
| zh | jbk.39.net | 19142 | 2033 | 17109 | 0 | 2033 |
| zh | care.39.net | 15134 | 1282 | 13852 | 0 | 1282 |
| zh | test.pmphai.com | 15120 | 2989 | 12131 | 0 | 2989 |
| zh | fitness.39.net | 12066 | 185 | 11881 | 0 | 185 |
| zh | woman.39.net | 11687 | 687 | 11000 | 0 | 687 |
| zh | food.39.net | 11024 | 594 | 10430 | 0 | 594 |
| zh | fk.39.net | 9812 | 928 | 8884 | 0 | 928 |
| zh | cancer.39.net | 8415 | 736 | 7679 | 0 | 736 |
| zh | jb39.com | 7306 | 527 | 6779 | 0 | 527 |
| zh | nk.39.net | 6650 | 762 | 5888 | 0 | 762 |
| zh | msdmanuals.cn | 6173 | 905 | 5268 | 0 | 905 |
| zh | heart.39.net | 5965 | 1055 | 4910 | 0 | 1055 |
| zh | article.iiyi.com | 5641 | 729 | 4912 | 0 | 729 |
| zh | man.39.net | 5320 | 382 | 4938 | 0 | 382 |
| zh | gan.39.net | 5027 | 294 | 4733 | 0 | 294 |
| zh | tnb.39.net | 4178 | 717 | 3461 | 0 | 717 |
| zh | wei.39.net | 3894 | 225 | 3669 | 0 | 225 |
| zh | baidianfeng.familydoctor.com.cn | 3798 | 60 | 3738 | 0 | 60 |
| zh | shen.39.net | 3223 | 325 | 2898 | 0 | 325 |
| zh | ek.39.net | 3222 | 448 | 2774 | 0 | 448 |
| zh | baidu.com | 2631 | 11 | 2620 | 0 | 11 |
| zh | gk.39.net | 2059 | 90 | 1969 | 0 | 90 |
| zh | gc.39.net | 1746 | 59 | 1687 | 0 | 59 |
| zh | pf.39.net | 1538 | 85 | 1453 | 0 | 85 |
| zh | health.people.com.cn | 1348 | 50 | 1298 | 0 | 50 |
| zh | sj.39.net | 942 | 197 | 745 | 0 | 197 |
| zh | byby.39.net | 439 | 37 | 402 | 0 | 37 |
| zh | familydoctor.cn | 44 | 2 | 42 | 0 | 2 |
| zh | qy.familydoctor.com.cn | 40 | 2 | 38 | 0 | 2 |
| zh | v.familydoctor.com.cn | 10 | 0 | 10 | 0 | 0 |
| zh | ask.familydoctor.com.cn | 3 | 1 | 2 | 0 | 1 |
| zh | yanglao.familydoctor.com.cn | 2 | 0 | 2 | 0 | 0 |
| zh | ypk.familydoctor.com.cn | 1 | 0 | 1 | 0 | 0 |

## 3. Module và dữ liệu (bước 3)

- `data/corpus_v2_ids.parquet` (ignored, 19.947.666 B): cột `id, url, lang, host`, sắp theo `id`, zstd. Sinh bằng `python -m r2ai.corpus_v2 build`.
- `src/r2ai/corpus_v2.py`: `is_member(id)` (bool và non-int luôn False, giống validator), `member_mask(ids)`, `load_ids()` (mảng int64 đã sắp, tải lại khi mtime/size đổi), `build()` (kiểm schema/null/duy nhất/host lạ, ghi nguyên tử, qua `assert_writable`).
- `src/r2ai/submit/corpus_filter.py`: lọc submission về v2, xem mục 5.
- Test: `tests/test_corpus_v2.py` (20) và `tests/test_corpus_filter.py` (5), dữ liệu tổng hợp, không mạng.

## 4. Tài sản còn hiệu lực trong v2 (bước 4)

Nguồn số: `out/runs/s9/assets.json` (script `scripts/s9_measure_assets.py`, chỉ đọc; SQLite mở `mode=ro`). Doc mang `doc_ids` (cả nhóm id của một URL): "any" = ít nhất một id ∈ v2, "all" = mọi id ∈ v2. Chunk tính theo `doc_id` chính (cái validator kiểm).

| Tài sản | Tổng | Còn hiệu lực | % |
|:--|--:|--:|--:|
| vi: doc đã extract (`data/docs_vi`) | 632.208 doc | 632.208 | 100 |
| vi: doc trong index (`data/chunks/docs.parquet`) | 623.805 doc | 623.805 | 100 |
| vi: chunk t256 trong index | 3.909.586 chunk | 3.909.586 | 100 |
| vi: cache rerank D100 (cặp query–doc) | 258.117 cặp / 96.868 doc | 100% | 100 |
| vi: cache rerank 200 | 309.536 cặp / 116.149 doc | 100% | 100 |
| vi: cache rerank 300 (P300) | 385.614 cặp / 143.352 doc | 100% | 100 |
| zh: cache rerank b001+sample (`retrieval-b001/rerank`, dense_merged) | 180.000 cặp / 112.690 doc | 153.387 cặp / 91.780 doc | 85,2 / 81,4 |
| zh: cache rerank hybrid_merged | 180.000 cặp / 112.620 doc | 153.734 cặp / 91.904 doc | 85,4 / 81,6 |
| zh: **ZG** id nối (`ZF/ZG/query_append.json`) | 5.547 id | 1.899 | 34,2 |
| zh: **ZG** doc chính | 3.611 | 1.645 | 45,6 |
| zh: ZG, số query có nối zh | 636 | 553 sau lọc (TB 1,58 id/query sau lọc, trước lọc 4,62) | — |
| zh: doc `sample-v2` | 35.686 doc | 3.270 (any) / 1.487 (all) | 9,2 / 4,2 |
| zh: doc `b001` | 743.945 doc | 48.567 / 32.256 | 6,5 / 4,3 |
| zh: doc `b002` (120ask + cnkang) | 290.744 doc | 28.165 / 15.784 | 9,7 / 5,4 |
| zh: chunk (30 segment, sample-v2 + b001 + b002) | 6.984.226 chunk | 512.851 (doc_id chính) / 526.381 (có id nhóm ∈ v2) | 7,3 / 7,5 |
| CC (S3) mục tiêu có bản chụp | 68.164 URL | 68.164 | 100 |
| CC đã fetch `ok` (`state/cc_longchau/fetch.db`) | 20.303 URL | 20.303 | 100 |
| CC doc đã extract (`docs_cc_longchau`) | 18.816 doc | 18.816 | 100 |
| CC chunk (seg20000 183.940 + seg20001 34.622) | 218.562 | 218.562 | 100 |
| CC queue S3 còn lại (mục tiêu − đã xử lý) | 47.861 URL | 47.861 | 100 |

Ghi chú:

- Toàn bộ vi và CC (đều là `nhathuoclongchau.com.vn`, vi) giữ nguyên 100% vì vi không đổi.
- zh chỉ còn dưới 10% mỗi batch. Doc zh theo domain còn trong v2 (any): `test.pmphai.com` 19,8% (b001), `msdmanuals.cn` 15,2%, `120ask.com` 12,7% (b001) và 22,6% (b002), `cnkang.com` 5,8% (b001) và 8,0% (b002), `zydcd.com` 0,16%, `qihuangzhishu.com` 0% (sample-v2). Số theo domain nằm trong `assets.json` (`zh_docs.<batch>.by_domain`).
- **b002 (120ask + cnkang)**: batch đã được cắt trong `state/zh_full_ingest/manifest.db` (2 segment `active`, 290.744 doc `ok`+`thin`), khác với PROGRESS (ghi "chưa cut"). Phần giao với v2 là 28.165 doc (9,7%): 7.767 của 120ask và 20.398 của cnkang. Theo D6/D8, b002 vẫn treo; nếu ingest lại chỉ phần giao thì cần ≈ 28k doc, không phải 290k.
- Hàng đợi crawl b002 (`state/zh_full/crawl.db`, snapshot 13:52; crawler còn chạy nên số sẽ đổi): 120ask `ok` 139.982 (21.169 có id ∈ v2), `pending` 363.958 (74.898 có id ∈ v2); cnkang `ok` 533.108 (35.856), `pending` 127.631 (3.350). Tức phần pending còn đáng crawl chỉ ≈ 78k URL thay vì 491k.
- Chunk zh: 13.530 chunk có `doc_id` chính ∉ v2 nhưng nhóm `doc_ids_group` có id ∈ v2. Dùng nguyên chunk đó cần đổi `doc_id` sang id nhóm trong v2 (không làm ở S9, chỉ ghi nhận cho S10/S11).
- ZG phải dựng lại trên v2: chỉ 34% id nối sống sót, và 553/636 query còn ít nhất một doc zh.

## 5. Ba file nộp: doc/chunk ngoài v2 và bản `*_v2.zip` (bước 5)

Lọc: bỏ id ngoài v2 khỏi `relevant_docs`, bỏ chunk có `doc_id` ngoài v2; giữ nguyên thứ tự, nội dung chunk và layout writer (một query một dòng, dòng không đổi giữ nguyên từng byte). Nén `ZIP_LZMA`, đúng một file `.json` ở gốc. Số liệu: `out/runs/s9/filter_*.json`.

| File nguồn | Query | Doc id (cặp) | Doc ngoài v2 | Chunk | Chunk ngoài v2 | Query bị đổi |
|:--|--:|--:|--:|--:|--:|--:|
| K80 LZMA (`upload-probe/K80/sub_K80_LZMA.zip`) | 1.200 | 163.613 | **0** | 95.999 | **0** | 0 |
| B-cut115 (`cc-longchau/VS-cc-B-cut115/VS_cc_B_cut115.zip`) | 1.200 | 163.630 | **0** | 60.000 | **0** | 0 |
| cc-A-cut115 (`cc-longchau/VS-cc-A-cut115/VS_cc_A_cut115.zip`) | 1.200 | 163.779 | **0** | 60.000 | **0** | 0 |

Cả ba chỉ gồm doc vi (VS-append + doc CC longchau), nên không có gì bị loại và JSON của bản `_v2` **giống hệt từng byte** bản gốc (SHA JSON khớp). Chỉ khác ở cách nén.

| Bản xuất (`out/runs/s9/`) | ZIP (B) | Nén | Từ điển | SHA256 JSON | Giải nén |
|:--|--:|:--|:--|:--|:--|
| `sub_K80_LZMA_v2.zip` | 78.734.571 | LZMA | **64 MiB** | `91fd6049…73a44` (khớp bản K80 ghi ở LOG) | Python + bsdtar OK |
| `VS_cc_B_cut115_v2.zip` | 64.194.627 | LZMA | mặc định 8 MiB | `a9f268fb…ef923` | Python + bsdtar OK |
| `VS_cc_A_cut115_v2.zip` | 62.630.600 | LZMA | mặc định 8 MiB | `99a9e11d…a2bfd` | Python + bsdtar OK |

Cả ba < 99 MB, một file JSON ở gốc ZIP (`<tên>_v2.json`). Các bản gốc của B và A là deflate 104,4 / 101,6 MB nên LZMA nhẹ hơn nhiều; K80 gốc đã là LZMA nên chỉ nhỏ đi khi tăng từ điển (xem mục 0.2). Validator đầy đủ (kiểm verbatim với docs) không chạy lại vì JSON không đổi; log validator gốc của B/A: 1.200 query / 60.000 chunk / 0 lỗi.

## 6. Top host en/zh của v2 (bước 6, cho S10/S11)

**en chỉ có đúng 1 host** (`pubmed.ncbi.nlm.nih.gov`, 240.000 URL, 100%), nên "top 30 en" thực tế là 1 dòng. zh có 43 host; top 30 chiếm 239.384/240.000 URL. Sinh bằng `scripts/s9_top_hosts.py`.

### en: 240000 URL, 1 host (hiển thị 1, chiếm 240000 URL)

| # | host | URL | % của lang | ví dụ |
|--:|:--|--:|--:|:--|
| 1 | pubmed.ncbi.nlm.nih.gov | 240000 | 100.00 | https://pubmed.ncbi.nlm.nih.gov/11197/ |

### zh: 240000 URL, 43 host (hiển thị 30, chiếm 239384 URL)

| # | host | URL | % của lang | ví dụ |
|--:|:--|--:|--:|:--|
| 1 | 120ask.com | 98160 | 40.90 | http://www.120ask.com/question/1000036.htm |
| 2 | cnkang.com | 40046 | 16.69 | http://www.cnkang.com/cm/201404/1145800.html |
| 3 | ask.39.net | 32772 | 13.65 | https://ask.39.net/question/100015049.html |
| 4 | familydoctor.com.cn | 26819 | 11.17 | https://www.familydoctor.com.cn/baby/a/201005/119629.html |
| 5 | zysjonline.com | 6591 | 2.75 | https://zysjonline.com/articles/misc/10944/ |
| 6 | zhongyibaodian.net | 6061 | 2.53 | https://zhongyibaodian.net/a/100023.html |
| 7 | a-hospital.com | 4701 | 1.96 | https://www.a-hospital.com/w/%CE%91-%E5%B9%B2%E6%89%B0%E7%B4%A0 |
| 8 | youlai.cn | 4139 | 1.72 | https://www.youlai.cn/yyk/article/187456.html |
| 9 | baby.39.net | 3368 | 1.40 | http://baby.39.net/a/090415/842083.html |
| 10 | test.pmphai.com | 2989 | 1.25 | https://test.pmphai.com/jeesitede/appcase/toPcDetail?sessionId=&knowledgeLibPrefix=case&id=11912 |
| 11 | jbk.39.net | 2033 | 0.85 | https://jbk.39.net/aechmbxcd/ |
| 12 | care.39.net | 1282 | 0.53 | http://care.39.net/a/100418/1236024.html |
| 13 | heart.39.net | 1055 | 0.44 | http://heart.39.net/a/080117/228542.html |
| 14 | fk.39.net | 928 | 0.39 | http://fk.39.net/a/080822/620947.html |
| 15 | msdmanuals.cn | 905 | 0.38 | https://www.msdmanuals.cn/home/blood-disorders/anemia/anemia-due-to-excessive-bleeding |
| 16 | nk.39.net | 762 | 0.32 | http://nk.39.net/a/100226/1165394.html |
| 17 | cancer.39.net | 736 | 0.31 | http://cancer.39.net/a/20101014/1499401.html |
| 18 | article.iiyi.com | 729 | 0.30 | https://article.iiyi.com/detail/10052.html |
| 19 | tnb.39.net | 717 | 0.30 | http://tnb.39.net/a/100225/1163553.html |
| 20 | wujue.com | 708 | 0.29 | http://www.wujue.com/Article/yyzx/201203/49964.html |
| 21 | woman.39.net | 687 | 0.29 | http://woman.39.net/a/100525/1304083.html |
| 22 | food.39.net | 594 | 0.25 | http://food.39.net/a/100417/1232752.html |
| 23 | jb39.com | 527 | 0.22 | https://jb39.com/jibing/AiZiBingRenDeJiXingLanWeiYan256234.htm |
| 24 | ek.39.net | 448 | 0.19 | http://ek.39.net/a/090609/893034.html |
| 25 | man.39.net | 382 | 0.16 | http://man.39.net/a/081016/683568.html |
| 26 | shen.39.net | 325 | 0.14 | http://shen.39.net/a/110711/1743462.html |
| 27 | gan.39.net | 294 | 0.12 | http://gan.39.net/a/100402/1207384.html |
| 28 | wei.39.net | 225 | 0.09 | http://wei.39.net/a/101225/1576119.html |
| 29 | zydcd.com | 204 | 0.09 | http://www.zydcd.com/YiAn/25215eizz.html |
| 30 | sj.39.net | 197 | 0.08 | http://sj.39.net/a/100327/1199782.html |


## 7. Tái lập

```
PYTHONPATH=src python -m r2ai.corpus_v2 build
PYTHONPATH=src python scripts/s9_measure_assets.py      # out/runs/s9/assets.json
PYTHONPATH=src python scripts/s9_top_hosts.py
PYTHONPATH=src python -m r2ai.submit.corpus_filter SRC DST_v2.zip [--lzma-dict-mib 64]
```

Biến môi trường lấy từ `.env` (đã sao vào worktree). Tải v2 cần `HF_HUB_OFFLINE=0` cho riêng lệnh tải (`.env` đặt =1).
