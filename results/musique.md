## musique — 1000 questions, 11656 passages

Graph: 32511 nodes (11656 passages, 11017 entities, 9838 documents), 341219 exact edges, 10.2 MB; float32 passage embeddings: 35.8 MB.

| Method | Full precision | Index MB | R@2 | R@5 | All@5 | ms/q | nodes scored/q | accepted (k-free) |
|---|---|---|---|---|---|---|---|---|
| Exact float32 scan | yes | 35.8 | 0.399 | 0.532 | 0.180 | 11.3 | 11656 | — |
| TurboVec flat 2-bit scan | no | 3.0 | 0.390 | 0.522 | 0.173 | 10.5 | 11656 | — |
| TurboVec flat 2-bit + exact rescore(100) | yes | 3.0 | 0.399 | 0.532 | 0.180 | 10.2 | 11656 | — |
| BM25 keyword | no | 5.0 | 0.345 | 0.443 | 0.102 | 0.6 | 36457 | — |
| KG entity link + 1-hop | no | 10.2 | 0.218 | 0.278 | 0.051 | 0.2 | 83 | 83 |
| 2-bit beam walk (ef=200) | no | 10.2 | 0.372 | 0.491 | 0.161 | 3.4 | 2182 | — |
| 2-bit threshold expansion (delta=0.08) | no | 10.2 | 0.319 | 0.397 | 0.085 | 1.0 | 784 | 6 |
| Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0) | no | 10.2 | 0.132 | 0.204 | 0.045 | 3.6 | 2038 | 74 |
| Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0) | no | 10.2 | 0.161 | 0.269 | 0.052 | 14.3 | 13471 | 76 |
| Fused 2-bit + Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0, lambda=0.05) | no | 10.2 | 0.314 | 0.425 | 0.135 | 4.5 | 2038 | 1423 |
| Fused 2-bit + Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0, lambda=0.1) | no | 10.2 | 0.313 | 0.425 | 0.136 | 3.9 | 2038 | 1423 |
| Fused 2-bit + Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0, lambda=0.2) | no | 10.2 | 0.297 | 0.420 | 0.134 | 4.0 | 2038 | 1423 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.05) | no | 10.2 | 0.392 | 0.520 | 0.174 | 14.8 | 13471 | 1400 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.1) | no | 10.2 | 0.390 | 0.524 | 0.177 | 14.7 | 13471 | 1400 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.2) | no | 10.2 | 0.374 | 0.520 | 0.174 | 14.5 | 13471 | 1400 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.1) + exact rescore | yes | 10.2 | 0.398 | 0.534 | 0.182 | 16.2 | 13471 | 1400 |
| Hybrid RRF: flat 2-bit + BM25 | no | 8.0 | 0.395 | 0.537 | 0.179 | 12.5 | 48113 | — |
| Hybrid RRF: flat 2-bit + BM25 + fused PPR SCAN | no | 18.2 | 0.399 | 0.542 | 0.185 | 28.9 | 61585 | — |
| Hybrid RRF: flat+rescore + BM25 + fused PPR SCAN | yes | 18.2 | 0.404 | 0.537 | 0.182 | 26.8 | 61585 | — |
