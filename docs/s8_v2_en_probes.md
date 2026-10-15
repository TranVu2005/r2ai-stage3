# S8 — probe en E1/E2 trên C1

Cập nhật: 2026-10-15 07:51 (UTC+7) · GPT-6 (Codex) · HEAD trước commit `462f250b491ffa311aa05df6d024959b69c5ce91`.

Base C1: Final **0,2655**; Doc F2/P/R **0,2687/0,1567/0,3886**; Chunk F2/P/R **0,2623/0,1857/0,3402** (LB do Vu cung cấp, 15/10/2026). C1 K150, dict256; giữ toàn bộ doc/chunk và thứ tự. Không có doc/chunk en sẵn trong C1.

## Cấu hình và số liệu thực

| Probe | τ doc / cap | Chunk en | Doc thêm / query | Chunk thêm / query | Query thêm doc | ZIP byte |
|---|---|---|---:|---:|---:|---:|
| E1 | ≥0,96 / 10 | ≥0,96 / top1 | 1837 / 1.530833 | 414 / 0.345000 | 414/1200 | 82283105 |
| E2 | ≥−0,873 / 30 | ≥0,96 / top3 | 10241 / 8.534167 | 923 / 0.769167 | 858/1200 | 82575852 |

Cap tính theo doc vật lý; mỗi alias v2 một ID/chunk. Loại toàn bộ `title_only` khỏi doc và chunk. Chỉ thêm chunk khi toàn bộ ID của doc đã có trong relevant_docs. Không thêm ID bảo vệ, không cắt vi/ZH. Nguyên `record_text` S10 (title + abstract/heading) dùng cho cả builder và validator.

ZIP-LZMA dict **268435456 B**, props **5d00000010**, lc/lp/pb **3/0/2**; trần **102600000 B**. Một JSON/ZIP.

### Phân bố theo query

- E1: doc histogram `{"0": 786, "1": 128, "2": 63, "3": 36, "4": 33, "5": 17, "6": 16, "7": 9, "8": 6, "9": 9, "10": 97}`; chunk histogram `{"0": 786, "1": 414}`.
  Doc median/P75/P90/P95/max=0/1/7/10/10; chunk=0/1/1/1/1. Không nhận doc: 786 query; không nhận chunk: 786 query.
- E2: doc histogram `{"0": 342, "1": 144, "2": 92, "3": 64, "4": 42, "5": 33, "6": 31, "7": 26, "8": 28, "9": 22, "10": 21, "11": 16, "12": 18, "13": 13, "14": 17, "15": 7, "16": 15, "17": 12, "18": 10, "19": 8, "20": 8, "21": 15, "22": 7, "23": 5, "24": 9, "25": 7, "26": 6, "27": 9, "28": 6, "29": 4, "30": 163}`; chunk histogram `{"0": 786, "1": 128, "2": 63, "3": 223}`.
  Doc median/P75/P90/P95/max=3/14/30/30/30; chunk=0/1/3/3/3. Không nhận doc: 342 query; không nhận chunk: 786 query.

## Dự báo có điều kiện, chưa là điểm LB

Precision áp cho cả doc và chunk en ở từng kịch bản. Gdoc=61,104348, ηchunk=79,491028 lấy từ inverse zh theo giả định Vu đã duyệt; F/P/R base C1 thật. Audit cung cấp k_q/c_q/m_q thực; giả định gold chung G và Fdoc_q≈macro; chunk dùng P/R macro làm proxy mỗi query, η chuyển hit sang recall. Không xác định precision thật hay khoảng tin cậy từ các giả định này.

`ΔDoc = mean[(5p − 0.2687) m_doc /(4G + k_q + m_doc)]`. Chunk: `h0=0.1857*c_q, r0=0.3402; h1=h0+p*m_chunk, r1=r0+p*m_chunk/η`; `ΔChunk=mean[f(h1,r1,c_q+m_chunk)−f(h0,r0,c_q)]`, `f(h,r,c)=5hr/(4h+rc)`. `ΔFinal=(ΔDoc+ΔChunk)/2`; baseline0,2655.

| Probe | Precision | ΔDoc F2 | ΔChunk F2 | ΔFinal | Final dự báo |
|---|---:|---:|---:|---:|---:|
| E1 | 3% | -0.000428 | -0.000165 | -0.000296 | 0.265204 |
| E1 | 6% | +0.000113 | -0.000042 | +0.000035 | 0.265535 |
| E1 | 10% | +0.000834 | +0.000122 | +0.000478 | 0.265978 |
| E2 | 3% | -0.002323 | -0.000359 | -0.001341 | 0.264159 |
| E2 | 6% | +0.000612 | -0.000087 | +0.000263 | 0.265763 |
| E2 | 10% | +0.004526 | +0.000275 | +0.002401 | 0.267901 |

## Gate và provenance

Pin nguồn/code/base được đối chiếu trước/sau; toàn bộ vi/CC tree 18069 file và 57 segment zh dùng manifest C1 đã xác minh. Readback giải nén SHA + đối chiếu đầy đủ từng query/prefix/source; không thay metric/dedupe. Native validator hoạt động đủ schema/v2/protection/dedupe/near/verbatim; release_check chạy `--derived` với HF live, không skip HF; JSON probe có gate SHA riêng khóa trước nén.

- Main snapshot commit `7177652c0117bc5c966091672a34713022f21092`; `configs/best-c1.yaml` trong snapshot SHA `120165ea3ff1e30d528c4ef37945f29e077df39d858f32daf31f9204f43d8f24`. Core unchanged, private EN adapter chỉ đổi language guard thêm mới sang en và nguồn native en.
- C1 ZIP SHA `e7c84e2c37c996fe3d046bf22a4d25b2fd9e8e1c2568d928fa45797108720c13`; JSON SHA `83375428b79957b1580299f97b05299bb292957826bd4cb8bdc602a83b8dd5f3`.
- Candidate `D:/GitHub/r2ai-stage3/artifacts/cand_en.parquet`: SHA `7b586de5df63d55c6040a6157d1248d5d08f11ffd0c2e7ca5e936222df6c4a91`, 60000 rows/1200 queries/top50.
- Records `D:/GitHub/r2ai-stage3/data/pubmed_en/records.parquet`: SHA `92e38b55ac23f41978a37e4d4baf69c6daef23fa15b33790b6a8762658a7da61`. Exact S10 `record_text` copied into pinned source_s10_build.py; no fetch/model run.
- QA search readonly 8 files, duy nhất `D:/GitHub/r2ai-stage3/out/runs/s10-en-pubmed-qa/qa_cc_sample.json` khớp SHA `1aff2b9cbe2d5e85a30768b3caa327e25f9ed78c0bf768afe7456076c50eec6b`; accepted n3 14/10, passed full-body/section gate. Không tạo lại/bỏ QA.

### E1

- ZIP: [E1.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-en/E1/E1.zip); **82283105 B**; headroom **20316895 B**.
- ZIP SHA `02d51168d155fe44331373fae2bef58273f071b141d5ae5e3694aa97cb880144`; JSON SHA `8ad394603e93640410e60d04e5021c86e226585754abef1e605ae2635b30681b`.
- Native: `0` errors / `1200` rows / `126622` chunks; release exit0/HF verified. Sidecars approval/validator/release-attempt log+JSON/release/build-plan/memory cạnh ZIP.
- RAM đo RSS nhóm tiến trình, mẫu10s: peak 5.569 GiB; chạy tuần tự. Không diễn giải RSS này là toàn bộ committed memory.

### E2

- ZIP: [E2.zip](C:/Users/Admin/.codex/worktrees/diag-lang/r2ai-stage3/out/diag-lang/v2/probes-en/E2/E2.zip); **82575852 B**; headroom **20024148 B**.
- ZIP SHA `1a5ad841bf671e6f1ca96c5bff2eeab6041ab7623be6b6bef7be65be26b8ae82`; JSON SHA `6486d2ec05e959d46bb4ab406ed34192c8df3023835704e2af6588334fe2742b`.
- Native: `0` errors / `1200` rows / `127131` chunks; release exit0/HF verified. Sidecars approval/validator/release-attempt log+JSON/release/build-plan/memory cạnh ZIP.
- RAM đo RSS nhóm tiến trình, mẫu10s: peak 6.032 GiB; chạy tuần tự. Không diễn giải RSS này là toàn bộ committed memory.

## Kiểm thử và review

57/57 scoped CPU tests passed1,55s: 32 EN +25 compose snapshot. RED→GREEN cho selection/source/gates và regression HF lỗi→retry thành công. Một review độc lập:0 Critical/1 Important; retryHF fixed trong một pass, tests GREEN.

Deferred minor: chưa assert module origin trong wrapper. Invocation dùng explicit `PYTHONPATH=<probe-root>/code-main/src`, child native/release kế thừa; snapshot SHA đã kiểm. Nếu tự đổi launch env, có nguy cơ import sai core; chưa dùng invocation đó. HF attempted/verified tách riêng trong record retry. Native/HF/ZIP/RAM mà reviewer chưa kết luận đã kiểm bằng artifact thật; chất lượng E1/E2 còn chờ LB.

Ruling: giữ snapshot code main và adapter en riêng vì compose gốc chỉ hỗ trợ zh; serializer/rank/schema/metric cũ giữ nguyên. Plan pre-review `E1.plan.json` được giữ làm bằng chứng; bản final `*.build-plan.json` pin code sau fix, cùng expected JSON, không fallback dữ liệu. Giữ workspace/output theo yêu cầu Vu.

Không GPU/crawl/push/nộp. Vu duyệt và nộp E1 trước/E2 sau, từ07:00; chạy lại release_check ngay trước upload nếu thời điểm kiểm HF đã cũ.

Builder/adapter và test được giữ ở `out/diag-lang/v2/probes-en/` (ignored); snapshot `code-main/src` dùng qua PYTHONPATH tường minh. Pin đầy đủ và audit từng query trong `E1/E1.build-plan.json`, `E2/E2.build-plan.json`; dữ liệu tổng hợp `summary.json`. Tracked chỉ docs.
