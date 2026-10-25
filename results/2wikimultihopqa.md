## 2wikimultihopqa — 1000 questions, 6119 passages

Graph: 18134 nodes (6119 passages, 5896 entities, 6119 documents), 164038 exact edges, 5.5 MB; float32 passage embeddings: 18.8 MB.

| Method | Full precision | Index MB | R@2 | R@5 | All@5 | ms/q | nodes scored/q | accepted (k-free) |
|---|---|---|---|---|---|---|---|---|
| Exact float32 scan | yes | 18.8 | 0.661 | 0.719 | 0.427 | 6.1 | 6119 | — |
| TurboVec flat 2-bit scan | no | 1.6 | 0.658 | 0.714 | 0.418 | 5.2 | 6119 | — |
| TurboVec flat 2-bit + exact rescore(100) | yes | 1.6 | 0.661 | 0.719 | 0.427 | 5.5 | 6119 | — |
| BM25 keyword | no | 2.3 | 0.550 | 0.655 | 0.325 | 0.3 | 12921 | — |
| KG entity link + 1-hop | no | 5.5 | 0.254 | 0.279 | 0.090 | 0.1 | 38 | 38 |
| 2-bit beam walk (ef=200) | no | 5.5 | 0.568 | 0.619 | 0.353 | 2.4 | 1306 | — |
| 2-bit threshold expansion (delta=0.08) | no | 5.5 | 0.416 | 0.436 | 0.124 | 0.5 | 398 | 3 |
| Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0) | no | 5.5 | 0.236 | 0.312 | 0.118 | 3.4 | 1782 | 75 |
| Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0) | no | 5.5 | 0.288 | 0.465 | 0.179 | 8.6 | 7719 | 74 |
| Fused 2-bit + Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0, lambda=0.05) | no | 5.5 | 0.467 | 0.515 | 0.251 | 4.0 | 1782 | 1205 |
| Fused 2-bit + Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0, lambda=0.1) | no | 5.5 | 0.466 | 0.517 | 0.255 | 3.7 | 1782 | 1205 |
| Fused 2-bit + Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0, lambda=0.2) | no | 5.5 | 0.455 | 0.521 | 0.261 | 3.5 | 1782 | 1205 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.05) | no | 5.5 | 0.641 | 0.700 | 0.398 | 9.1 | 7719 | 1176 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.1) | no | 5.5 | 0.638 | 0.703 | 0.404 | 9.3 | 7719 | 1176 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.2) | no | 5.5 | 0.618 | 0.706 | 0.407 | 8.9 | 7719 | 1176 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.1) + exact rescore | yes | 5.5 | 0.642 | 0.699 | 0.400 | 9.5 | 7719 | 1176 |
| Hybrid RRF: flat 2-bit + BM25 | no | 3.9 | 0.630 | 0.706 | 0.399 | 5.8 | 19040 | — |
| Hybrid RRF: flat 2-bit + BM25 + fused PPR SCAN | no | 9.4 | 0.632 | 0.707 | 0.408 | 14.9 | 26758 | — |
| Hybrid RRF: flat+rescore + BM25 + fused PPR SCAN | yes | 9.4 | 0.631 | 0.707 | 0.407 | 15.6 | 26758 | — |
