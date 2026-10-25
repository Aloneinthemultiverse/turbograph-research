## hotpotqa — TurboVec-HippoRAG, 1000 questions, 9811 passages

Graph: 204633 nodes (9811 passages, 90330 entities, 104492 facts), 1645246 exact edges. Index: 2-bit 64.7 MB vs float32 641.8 MB (9.9x smaller).

| Method | Full precision | Index MB | R@2 | R@5 | All@5 | ms/q | nodes scored/q |
|---|---|---|---|---|---|---|---|
| Exact float32 scan | yes | 30.1 | 0.744 | 0.865 | 0.750 | 13.7 | 9811 |
| Dense 2-bit (TurboVec flat) | no | 2.6 | 0.743 | 0.858 | 0.737 | 12.3 | 9811 |
| BM25 keyword | no | 4.6 | 0.565 | 0.727 | 0.492 | 0.7 | 27982 |
| HippoRAG 2 (float vectors, global PPR) | yes | 641.8 | 0.612 | 0.870 | 0.765 | 598.2 | 114303 |
| HippoRAG 2 graph + 2-bit vectors (global PPR) | no | 64.7 | 0.608 | 0.865 | 0.756 | 581.0 | 114303 |
| TurboVec-HippoRAG (2-bit, local push, HippoRAG edges) | no | 64.7 | 0.584 | 0.742 | 0.529 | 145.7 | 114303 |
| TurboVec-HippoRAG + fact nodes + SIMILAR edges (2-bit, local push) | no | 64.7 | 0.624 | 0.846 | 0.721 | 142.4 | 114303 |
| TurboVec-HippoRAG fused (2-bit, local push, lambda=0.1) | no | 64.7 | 0.729 | 0.871 | 0.761 | 151.4 | 114303 |
| Hybrid RRF: dense 2-bit + BM25 + TurboVec-HippoRAG | no | 71.9 | 0.697 | 0.874 | 0.761 | 161.6 | 152096 |
