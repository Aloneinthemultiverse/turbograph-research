# turbograph-research

Side-by-side benchmark of retrieval over a **TurboVec knowledge graph**: one structure
where every node (chunk, term, document) is a 2-bit TurboQuant code and the
knowledge-graph edges are stored exactly. Plain Java 21 + JDBC; reads the Auralis
knowledge base (Postgres :5433) and embeds queries with Ollama `nomic-embed-text`.

Methods (all in `Methods.java`): exact float32 scan, flat 2-bit scan (with/without exact
rescore), BM25, KG entity-link + 1-hop, 2-bit beam walk, k-free 2-bit threshold expansion,
local push Personalized PageRank (LOCAL / SCAN seeds, with/without rescore), and RRF hybrids.

```bash
./mvnw -q -o compile dependency:build-classpath -Dmdep.outputFile=cp.txt
java -Xmx6g -Djdk.net.unixdomain.tmpdir=C:/Temp/auralis -cp "target/classes;$(cat cp.txt)" ai.auralis.turbograph.Bench 100
```

Results: `results/results.md`, `results/results.csv`. Query embeddings are cached in
`results/cache/queries.tsv`.
