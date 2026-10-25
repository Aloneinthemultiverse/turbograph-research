## 2wikimultihopqa — TurboVec-HippoRAG, 1000 questions, 6119 passages

Graph: 90808 nodes (6119 passages, 39834 entities, 44855 facts), 783378 exact edges. Index: 2-bit 29.1 MB vs float32 285.2 MB (9.8x smaller).

| Method | Full precision | Index MB | R@2 | R@5 | All@5 | ms/q | nodes scored/q |
|---|---|---|---|---|---|---|---|
| Exact float32 scan | yes | 18.8 | 0.661 | 0.719 | 0.427 | 8.7 | 6119 |
| Dense 2-bit (TurboVec flat) | no | 1.6 | 0.658 | 0.714 | 0.417 | 7.9 | 6119 |
| BM25 keyword | no | 2.3 | 0.550 | 0.655 | 0.325 | 0.4 | 12921 |
| HippoRAG 2 (float vectors, global PPR) | yes | 285.2 | 0.607 | 0.800 | 0.578 | 274.0 | 50974 |
| HippoRAG 2 graph + 2-bit vectors (global PPR) | no | 29.1 | 0.608 | 0.800 | 0.582 | 265.7 | 50974 |
| TurboVec-HippoRAG (2-bit, local push, HippoRAG edges) | no | 29.1 | 0.581 | 0.692 | 0.396 | 69.6 | 50974 |
| TurboVec-HippoRAG + fact nodes + SIMILAR edges (2-bit, local push) | no | 29.1 | 0.591 | 0.737 | 0.483 | 69.5 | 50974 |
| TurboVec-HippoRAG fused (2-bit, local push, lambda=0.1) | no | 29.1 | 0.656 | 0.725 | 0.434 | 70.0 | 50974 |
| Hybrid RRF: dense 2-bit + BM25 + TurboVec-HippoRAG | no | 33.0 | 0.644 | 0.724 | 0.431 | 77.3 | 70014 |
