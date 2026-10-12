# Danh mục artifact và SHA256 — 12/10/2026

Đây là inventory các file cục bộ; không file nào được upload trong S1. Hash được tính trực tiếp từ file hiện có. Các đường dẫn `out/` là output ignored, không commit.

## Pin ngoài repo

| File | Byte | SHA256 |
|---|---:|---|
| `D:/r2ai-pins/slug_pool_pin.parquet` | 30.920.581 | `196f2e50cd914de4bf55c812519ec5d05afde1ea31a1608b207442a24b00371f` |
| `D:/r2ai-pins/slug_pool_pin.parquet.sha256` | 88 | `1f0717fdb7195174418dca2ccc31b9eb0cba74286ad23252cd575d4515a7fc33` |

## `out/runs/selective-check/VS-append/`

| File | Byte | SHA256 |
|---|---:|---|
| `query_append.json` | 243.521 | `38a2a256a8c4f909de705e10e1898a2f6ae98f3104c8ab5522aad189e95a0009` |
| `stats.json` | 961 | `8f1c07b025a23fd6f308c663d0fe7936162c17a9437fd0388e1bd0bf9ebb6119` |
| `validator.log` | 58 | `f58a10f63983ca9dce798916a25caacb9f808604d39797d92067980d9966d422` |
| `VS_append.json` | 402.631.653 | `9c91c952863c4e33b992053d665d547ba1f483cd0be487047386ba09af43e2e0` |
| `VS_append.zip` | 101.761.767 | `48da9d87a1140a732eb06caa5c5573a6bd68d877f155067718831f01930853fe` |

## ZF/ZG submission JSON và ZIP

| Biến thể | File | Byte | SHA256 |
|---|---|---:|---|
| ZF-100 | `ZF_100.json` | 404.218.983 | `cb8194198b0a0ceed3b7ca5a1ff9433ff00bf99fcaf33152c1794fb4e4ca8083` |
| ZF-100 | `ZF_100.zip` | 102.648.895 | `da2eba59683dfcbd7976904c600dcf90fa98face3f7fb0208a83082f4c0ed500` |
| ZF-10 | `ZF_10.json` | 402.792.318 | `4d45a345d7aff398b6dafe48ea5acf7c543b03fe22668be62dfed7a9ed866502` |
| ZF-10 | `ZF_10.zip` | 101.863.630 | `517680fd8694d3e70d492b8e4e52634f383c815c096e30eb4b039bb5b1b2b839` |
| ZF-120ask-5 | `ZF_120ask_5.json` | 402.725.563 | `86e9da85adda062125fcaa0f2162362970c586cb35ea686f729f75eaef57412b` |
| ZF-120ask-5 | `ZF_120ask_5.zip` | 101.823.214 | `e55d8e4a652b76c5bc8808fa05cac57d65f6547a7cd8f744f544319f972f09bf` |
| ZG | `ZG.json` | 402.680.563 | `229632ce08c37e9a09be1811fab9e32936a6b567810d7a705e4be8debf31d356` |
| ZG | `ZG.zip` | 101.793.531 | `fa70e1824a90ab4d2b9858976395c97641a3222851087fe6041d539f5e9dd81f` |
| ZGp | `ZGp.json` | 402.663.139 | `0e087e45d2255e95af7927585db641203e2790022dad3b32e85e93040bd6328d` |
| ZGp | `ZGp.zip` | 101.782.241 | `3ab232dbf09f5f1f26f260609400da203c8ab0d3b7da09526a3cd5feb390d0c5` |
| ZGp-hi | `ZGp_hi.json` | 402.647.345 | `307805080766a506ed63b81a851b763a22930ee6e6379ab749ed6af93f74c2de` |
| ZGp-hi | `ZGp_hi.zip` | 101.772.103 | `7de4d847712ad8bea399fc68082c06afc7ec5c9247831d68d901d6066936cd68` |

Artifacts are under `D:/GitHub/r2ai-stage3/out/runs/zh-full-ingest/ZF/<variant>/`.

### D4 removal record

`ZF-5p` and `ZF-10p` directories were removed from the ignored output tree. These are their JSON/ZIP checksums recorded immediately before deletion; these files are no longer present.

| Removed variant | File | Byte | SHA256 before deletion |
|---|---|---:|---|
| ZF-5p | `ZF_5p.json` | 402.683.791 | `aad99d60b60beae0ede5fd59e2fc2bd491875056bad53cbc0217bf1d4bbea4db1` |
| ZF-5p | `ZF_5p.zip` | 101.796.855 | `44d632cf7be339fd6c3aa1b38b0e9c34b9abaae5e88a7f70c2d1981d383b33f36` |
| ZF-10p | `ZF_10p.json` | 402.736.095 | `536dce2169d32eab9ecc1097aeea508ecfdef50d87071dce77cd3e91adc50ef` |
| ZF-10p | `ZF_10p.zip` | 101.829.122 | `953834131b70f10d8bde5ba514a2d38ec8931f67bdb50f3b6308d286c4c57e1d` |

## S1 replay

`out/runs/s1-replay/VS_append.json`: 402.631.653 byte, SHA256 `9c91c952863c4e33b992053d665d547ba1f483cd0be487047386ba09af43e2e0` (khớp JSON pin). `VS_append.zip`: 101.761.767 byte, SHA256 cuối `48da9d87a1140a732eb06caa5c5573a6bd68d877f155067718831f01930853fe` (khớp ZIP pin). Builder ban đầu tạo SHA `15f757624eaa953af17c9602e5b4254d2881e0687076611b91a3866135a6a1b7`; byte diff cho thấy chỉ khác 6 byte timestamp metadata trong ZIP. Đã đặt timestamp entry về giá trị của artifact tham chiếu; JSON, CRC và dữ liệu nén không đổi.
