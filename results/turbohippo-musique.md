## musique — TurboVec-HippoRAG, 1000 questions, 11656 passages

Graph: 220951 nodes (11656 passages, 98065 entities, 111230 facts), 1819694 exact edges. Index: 2-bit 70.2 MB vs float32 693.3 MB (9.9x smaller).

| Method | Full precision | Index MB | R@2 | R@5 | All@5 | ms/q | nodes scored/q |
|---|---|---|---|---|---|---|---|
| Exact float32 scan | yes | 35.8 | 0.400 | 0.532 | 0.180 | 17.5 | 11656 |
| Dense 2-bit (TurboVec flat) | no | 3.0 | 0.390 | 0.524 | 0.177 | 15.0 | 11656 |
| BM25 keyword | no | 5.0 | 0.345 | 0.443 | 0.102 | 0.9 | 36457 |
| HippoRAG 2 (float vectors, global PPR) | yes | 693.3 | 0.408 | 0.600 | 0.294 | 685.3 | 122886 |
| HippoRAG 2 graph + 2-bit vectors (global PPR) | no | 70.2 | 0.401 | 0.590 | 0.277 | 657.6 | 122886 |
| TurboVec-HippoRAG (2-bit, local push, HippoRAG edges) | no | 70.2 | 0.383 | 0.498 | 0.168 | 162.7 | 122886 |
| TurboVec-HippoRAG + fact nodes + SIMILAR edges (2-bit, local push) | no | 70.2 | 0.398 | 0.551 | 0.227 | 162.1 | 122886 |
| TurboVec-HippoRAG fused (2-bit, local push, lambda=0.1) | no | 70.2 | 0.415 | 0.555 | 0.201 | 162.1 | 122886 |
| Hybrid RRF: dense 2-bit + BM25 + TurboVec-HippoRAG | no | 78.2 | 0.425 | 0.561 | 0.211 | 178.2 | 170999 |
