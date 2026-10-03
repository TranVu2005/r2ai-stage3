> Tài liệu legacy được giữ để tham khảo; root/module/CLI hiện hành theo README.md. Writer phụ đang khóa; không chạy lệnh legacy vào OLD.

# eval/

* `scorer.py` – local re-implementation of the organisers' metric (Doc F2 + Chunk F2, final = mean), aligned with the
  official spec (NFKC, punctuation kept, no merging of duplicate preds); assumptions in the module docstring.
  `python -m eval.scorer --pred X --gold Y [--macro all|has_gold]`, `--bench` for timing.
* `validate.py` – submission checks (1,200 int ids, docs in links_corpus, chunk_text verbatim, optional chunk_order, ZIP with 1 file).
  `python -m eval.validate out/submissions/<name>.zip`
* tests: `tests/test_scorer.py`
* gold-check tooling from stage 1 lives at the repo root (`gold_check_*.py`, see `README-gold-check.md`)
