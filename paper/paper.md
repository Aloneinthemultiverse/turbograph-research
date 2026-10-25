# TurboVec-GraphRAG: Retrieval over a 2-bit Quantized Knowledge Graph with Exact Relationships

*Draft — Sujit Narrayan M. Numbers in §5 are produced by `turbograph-research` (`Bench`, `BenchHippo`).*

## Abstract

Graph-based retrieval-augmented generation (GraphRAG) recovers multi-hop evidence that
dense retrieval misses, but it stores full-precision embeddings for every passage and
entity on top of the graph itself. Vector quantization (TurboQuant/TurboVec) shrinks
embeddings to 2 bits per dimension, but a flat quantized index has no notion of
relationships. We study **TurboVec-GraphRAG**: a single structure in which every node
(passage, entity, document) is a 2-bit TurboQuant code and every relationship is an exact,
lossless edge. Retrieval is a graph traversal scored only with 2-bit codes. We compare
eleven retrieval mechanisms side by side — flat exact and 2-bit scans, BM25, entity-link
k-hop, 2-bit beam walk, k-free threshold expansion, local push Personalized PageRank
(PPR), a fused 2-bit + PPR ranking, and reciprocal-rank hybrids — on a 35k-passage
engineering corpus and on the HippoRAG multi-hop benchmarks (HotpotQA, 2WikiMultiHopQA,
MuSiQue). [Headline result to be filled from §5.]

## 1 Introduction

- Dense retrieval ranks passages by similarity to the whole question; multi-hop questions
  need passages that are each similar to only part of it.
- GraphRAG (Microsoft GraphRAG, LightRAG, HippoRAG) adds entities and relations, but the
  memory cost is graph + full-precision vectors for passages, entities and relations.
- TurboQuant-style quantization (random rotation, Lloyd–Max 2-bit codebook, per-vector
  bias correction) cuts vectors ~12× with small inner-product error and no training.
- **Question:** can one structure carry both — 2-bit semantics and exact relationships —
  and retrieve multi-hop evidence by traversal without ever reading full-precision vectors?

**Contributions.**
1. The TurboVec knowledge graph: 2-bit node codes + exact typed adjacency in one index,
   including navigable SIMILAR edges so the same graph serves as the vector index.
2. Retrieval mechanisms whose per-step cost is O(degree): greedy descent, k-free threshold
   expansion, and local push PPR with 2-bit-weighted edges; plus a fused ranking.
3. A side-by-side benchmark (open source, Java) on a private engineering corpus and on
   the three HippoRAG datasets, reporting recall, latency, nodes scored and index bytes.

## 2 Related work

- **Vector quantization:** product quantization, FAISS FastScan, RaBitQ, TurboQuant;
  `turbovec` (2/4-bit, brute-force SIMD scan, staged rescoring).
- **Graph ANN:** NSW/HNSW, DiskANN (graph traversal over PQ codes) — traversal over
  compressed codes is standard for *similarity*; we add *knowledge* edges.
- **GraphRAG:** entity linking + k-hop subgraphs (GraftNet, PullNet), relation-path search
  (Think-on-Graph), PPR over passage–entity graphs (HippoRAG 1/2), dual-level entity and
  relation retrieval (LightRAG), community summaries (Microsoft GraphRAG), prize-collecting
  Steiner subgraphs (G-Retriever).
- **Local PPR:** Andersen–Chung–Lang push; work bounded by 1/(ε·α), independent of graph size.

## 3 The TurboVec knowledge graph

**Codec.** Each vector is normalised, padded to a power of two, rotated by a seeded
randomized Hadamard transform, and each coordinate quantised to the 4 Lloyd–Max levels of
N(0,1) (±0.4528, ±1.5104)·D^-½. A per-vector gain 1/⟨x, Q(x)⟩ removes inner-product
bias. Queries stay full precision (rotated once) and are scored asymmetrically against
codes without decompression. Mean absolute error vs exact cosine on random 768-d vectors:
0.008 (gain-corrected).

**Nodes.** Passages; entities (technical identifiers such as AXI4, CVA6 in engineering
text; capitalised names and passage titles in Wikipedia text) kept if they occur in ≥2
passages and ≤2% of them; documents. Entity and document codes are the 2-bit codes of the
mean of their passages' rotated vectors.

**Edges (exact).** MENTIONS (passage–entity), PART_OF (passage–document), NEXT (adjacent
passages), SIMILAR (≤24 per passage, built by inserting passages in random order with a
2-bit beam search, ef=48, M=12; parallel batches of 256 after the first 1000). Each edge
is a 4-byte target + 1-byte type. TransE-style relation vectors r = mean(dst − src) shift
the query when crossing knowledge edges.

**Size.** Bytes = nodes × (D/4 + 4) + edges × 5. The engineering corpus (35,387 passages):
19.4 MB including 1.16 M edges, vs 108.7 MB for float32 passage embeddings alone.

## 4 Retrieval mechanisms compared

| Mechanism | Uses graph | Full precision at query time | Cost |
|---|---|---|---|
| Exact float32 scan | no | yes | O(N·d) |
| Flat 2-bit scan (TurboVec) | no | no | O(N·d/4) |
| Flat 2-bit + exact rescore of 100 | no | yes (100 vectors) | O(N·d/4) |
| BM25 | no | no | O(postings) |
| Entity link + 1-hop | yes | no | O(deg(entities)) |
| 2-bit beam walk (ef=200) | yes | no | O(ef·deg) |
| 2-bit threshold expansion (δ) — k-free | yes | no | O(Σ deg of accepted) |
| Push-PPR LOCAL — seeds: query entities + greedy-descent peaks | yes | no | O(1/(ε·α)) pushes |
| Push-PPR SCAN — seeds: query entities + 2-bit flat-scan peaks | yes | no | O(N·d/4) + pushes |
| Fused 2-bit + Push-PPR (λ) | yes | no | as PPR |
| RRF hybrids | mixed | mixed | sum |

Push weights: w(u→v) = ρ(rel) · max(0.01, s₂(q, v))^γ · idf(v), with ρ = {MENTIONS 1,
PART_OF 0.3, NEXT 0.7, SIMILAR 1}, γ = 4, α = 0.15, ε = 1e-4. Fused score:
s₂(q, v) + λ · p(v) / max p.

## 5 Experiments

### 5.1 Engineering corpus (Auralis, 35,387 passages, 907 documents)

Queries are spans cut from passages (100 single-hop) and joined spans of two passages
from different documents sharing a rare entity (100 two-hop). *Caveat: spans are verbatim,
which favours BM25; §5.2 uses natural questions.*

| Method | Full prec. | Index MB | 1-hop source@10 | 2-hop both@10 | ms | nodes scored |
|---|---|---|---|---|---|---|
| Exact float32 | yes | 108.7 | 0.79 | 0.25 | 50 | 35,387 |
| Flat 2-bit | no | 9.2 | 0.81 | 0.25 | 43 | 35,387 |
| 2-bit beam walk | no | 19.4 | 0.80 | 0.25 | 19 | 7,541 |
| Threshold expansion δ=0.08 | no | 19.4 | 0.81 | 0.25 | 15 | 9,507 |
| Push-PPR LOCAL | no | 19.4 | 0.30 | 0.35 | 7 | 2,396 |
| Push-PPR LOCAL + rescore | yes | 19.4 | 0.44 | 0.40 | 5 | 2,396 |
| BM25 | no | 35.2 | 1.00 | 0.97 | 2 | — |

### 5.2 HippoRAG benchmarks (1000 questions each)

[Tables from results/hotpotqa.md, results/2wikimultihopqa.md, results/musique.md.]

### 5.3 Ablations

[λ sweep; LOCAL vs SCAN seeds; γ; ε (work vs recall).]

## 6 Discussion and limitations

- Entities are extracted by patterns, not an LLM; typed fact edges (subject–relation–object)
  are future work and are expected to help most on MuSiQue.
- Embeddings are nomic-embed-text (768-d); HippoRAG's published numbers use other
  retrievers (Contriever, ColBERTv2, NV-Embed-v2), so absolute numbers are not directly
  comparable; the comparison inside this paper is controlled (same embeddings, same graph).
- Single machine, CPU; latency includes Java overhead, not SIMD kernels.

## 7 Reproducibility

`turbograph-research` (Java 21): `Bench` (engineering corpus), `BenchHippo <dataset>`;
datasets from the HippoRAG repository (`reproduce/dataset`), converted by
`scripts/prep_hipporag.py`.
