# TurboVec knowledge graph — side-by-side retrieval benchmark

Corpus: 35387 chunks, 907 documents. Graph: 52055 nodes, 1163195 edges, 19.4 MB (2-bit codes + exact edges); float32 embeddings of the chunks alone: 108.7 MB.

100 single-hop queries (span of one chunk), 100 two-hop queries (spans of two chunks in different documents sharing a rare term).

| Method | Full precision at query time | Index MB | 1-hop R@10 vs exact | 1-hop source@1 | 1-hop source@10 | 1-hop MRR | 2-hop both@10 | 2-hop both@20 | 2-hop any@10 | ms/query | nodes scored/query | accepted set (k-free) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Exact float32 scan | yes | 108.7 | 1.000 | 0.490 | 0.790 | 0.613 | 0.250 | 0.290 | 0.630 | 50.2 | 35387 | — |
| TurboVec flat 2-bit scan | no | 9.2 | 0.814 | 0.440 | 0.810 | 0.575 | 0.250 | 0.270 | 0.590 | 42.9 | 35387 | — |
| TurboVec flat 2-bit + exact rescore(100) | yes | 9.2 | 0.999 | 0.490 | 0.790 | 0.613 | 0.250 | 0.290 | 0.630 | 43.8 | 35387 | — |
| BM25 keyword | no | 35.2 | 0.375 | 0.960 | 1.000 | 0.974 | 0.970 | 0.980 | 1.000 | 2.3 | 114884 | — |
| KG entity link + 1-hop | no | 19.4 | 0.296 | 0.280 | 0.550 | 0.379 | 0.340 | 0.410 | 0.660 | 1.2 | 296 | 296 |
| 2-bit beam walk (ef=200) | no | 19.4 | 0.731 | 0.490 | 0.800 | 0.604 | 0.250 | 0.290 | 0.610 | 18.9 | 7541 | — |
| 2-bit threshold expansion (delta=0.08) | no | 19.4 | 0.668 | 0.490 | 0.810 | 0.604 | 0.250 | 0.270 | 0.600 | 15.0 | 9507 | 79 |
| 2-bit threshold expansion (delta=0.12) | no | 19.4 | 0.722 | 0.490 | 0.800 | 0.604 | 0.250 | 0.280 | 0.600 | 37.9 | 22111 | 899 |
| Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0) | no | 19.4 | 0.073 | 0.060 | 0.300 | 0.121 | 0.350 | 0.410 | 0.540 | 6.7 | 2396 | 54 |
| Push-PPR LOCAL (a=0.15, eps=1.0E-5, g=4.0) | no | 19.4 | 0.072 | 0.060 | 0.290 | 0.118 | 0.350 | 0.430 | 0.540 | 34.6 | 11127 | 498 |
| Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=1.0) | no | 19.4 | 0.071 | 0.030 | 0.290 | 0.087 | 0.330 | 0.390 | 0.500 | 5.2 | 2375 | 55 |
| Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0) | no | 19.4 | 0.107 | 0.060 | 0.360 | 0.144 | 0.380 | 0.490 | 0.550 | 48.2 | 37144 | 63 |
| Push-PPR LOCAL (a=0.15, eps=1.0E-4, g=4.0) + exact rescore | yes | 19.4 | 0.202 | 0.290 | 0.440 | 0.348 | 0.400 | 0.490 | 0.640 | 5.0 | 2396 | 54 |
| Push-PPR SCAN (a=0.15, eps=1.0E-4, g=4.0) + exact rescore | yes | 19.4 | 0.650 | 0.490 | 0.780 | 0.605 | 0.310 | 0.400 | 0.640 | 47.4 | 37144 | 63 |
| Hybrid RRF: flat 2-bit + BM25 | no | 44.4 | 0.586 | 0.730 | 0.950 | 0.810 | 0.670 | 0.900 | 0.930 | 44.9 | 150271 | — |
| Hybrid RRF: flat 2-bit + BM25 + Push-PPR SCAN | no | 63.8 | 0.637 | 0.620 | 0.930 | 0.737 | 0.580 | 0.720 | 0.860 | 92.2 | 187416 | — |
| Hybrid RRF: flat+rescore + BM25 + Push-PPR SCAN (Auralis-style) | yes | 63.8 | 0.650 | 0.630 | 0.940 | 0.739 | 0.600 | 0.720 | 0.860 | 94.4 | 187416 | — |

Example two-hop queries:

- shared term `TABLES`: "the exhaust. What we need in this system is a sure way of detecting the presence of a fla ... the evicting partition’s tag array is modified, plus the shared data entry’s refcount and" → Lessons In Electric Circuits, Volume IV - Digital — Full book #255 | Partitioned Tags, Shared Data: Reconciling Strict Cache Isolation with Write-Shared Coherence #73
- shared term `SPIKE318`: ".spike301(spike[301]), .spike302(spike[302]), .spike303(spike[303]), .spike304(spike[304] ... .spike302(spike[302]), .spike303(spike[303]), .spike304(spike[304]), .spike305(spike[305]" → Akilax0/e19-4yp-RISC-V-SoC-With-Neuromorphic-Accelerator-for-SNNs — code/ref/e18-10-neurons/src/accelerator_initialize_junk.v #7 | Akilax0/e19-4yp-RISC-V-SoC-With-Neuromorphic-Accelerator-for-SNNs — code/ref/e18-main/src/accelerator_initialize_scaled.v #7
- shared term `MFB21`: "https://support.lenovo.com/us/en/product_security/row_hammer https://users.sussex.ac.uk/~ ... Memory without Accessing Them: An Experimental Study of DRAM Disturbance Errors,” arXiv," → ScaleDisturb: Exploiting Temporal Asymmetry to Amplify Read Disturbance in Modern DRAM Chips #70 | DejaVu: Why You Should Write to Your DRAM Rows Twice, Carefully #78
