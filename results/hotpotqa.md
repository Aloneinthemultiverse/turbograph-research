## hotpotqa — 1000 questions, 9811 passages

Graph: 30170 nodes (9811 passages, 10548 entities, 9811 documents), 288321 exact edges, 9.3 MB; float32 passage embeddings: 30.1 MB.

| Method | Full precision | Index MB | R@2 | R@5 | All@5 | ms/q | nodes scored/q | accepted (k-free) |
|---|---|---|---|---|---|---|---|---|
| Exact float32 scan | yes | 30.1 | 0.744 | 0.864 | 0.749 | 9.9 | 9811 | — |
| TurboVec flat 2-bit scan | no | 2.6 | 0.741 | 0.858 | 0.738 | 9.3 | 9811 | — |
| TurboVec flat 2-bit + exact rescore(100) | yes | 2.6 | 0.744 | 0.864 | 0.749 | 10.1 | 9811 | — |
| BM25 keyword | no | 4.6 | 0.565 | 0.727 | 0.492 | 0.6 | 27982 | — |
| KG entity link + 1-hop | no | 9.3 | 0.456 | 0.515 | 0.311 | 0.2 | 61 | 61 |
| 2-bit beam walk (ef=200) | no | 9.3 | 0.729 | 0.845 | 0.719 | 3.2 | 1703 | — |
| 2-bit threshold expansion (delta=0.08) | no | 9.3 | 0.592 | 0.628 | 0.346 | 0.5 | 418 | 2 |
| Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0) | no | 9.3 | 0.353 | 0.493 | 0.277 | 3.5 | 1956 | 69 |
| Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0) | no | 9.3 | 0.366 | 0.538 | 0.290 | 13.2 | 11473 | 70 |
| Fused 2-bit + Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0, lambda=0.05) | no | 9.3 | 0.683 | 0.793 | 0.664 | 4.2 | 1956 | 1323 |
| Fused 2-bit + Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0, lambda=0.1) | no | 9.3 | 0.680 | 0.794 | 0.667 | 4.1 | 1956 | 1323 |
| Fused 2-bit + Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0, lambda=0.2) | no | 9.3 | 0.650 | 0.791 | 0.660 | 3.9 | 1956 | 1323 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.05) | no | 9.3 | 0.742 | 0.863 | 0.745 | 13.6 | 11473 | 1291 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.1) | no | 9.3 | 0.736 | 0.862 | 0.742 | 13.7 | 11473 | 1291 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.2) | no | 9.3 | 0.707 | 0.860 | 0.737 | 13.3 | 11473 | 1291 |
| Fused 2-bit + Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0, lambda=0.1) + exact rescore | yes | 9.3 | 0.742 | 0.862 | 0.745 | 13.5 | 11473 | 1291 |
| Hybrid RRF: flat 2-bit + BM25 | no | 7.2 | 0.671 | 0.858 | 0.728 | 9.6 | 37793 | — |
| Hybrid RRF: flat 2-bit + BM25 + fused PPR SCAN | no | 16.5 | 0.705 | 0.870 | 0.754 | 22.0 | 49266 | — |
| Hybrid RRF: flat+rescore + BM25 + fused PPR SCAN | yes | 16.5 | 0.707 | 0.872 | 0.757 | 22.6 | 49266 | — |
