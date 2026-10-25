package ai.auralis.turbograph;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.stream.IntStream;

import ai.auralis.turbograph.TwoBitCodec.Code;

/**
 * The TurboVec knowledge graph: ONE structure holding, for every node, a 2-bit
 * TurboQuant code and an exact (lossless) adjacency list.
 * <p>
 * Nodes [0, chunks) are text chunks, then terms (technical identifiers found in
 * the text), then documents. Edge types: MENTIONS (chunk-term), PART_OF
 * (chunk-document), NEXT (adjacent chunks), SIMILAR (navigable small-world links
 * between chunks, built by 2-bit beam insertion). Full-precision embeddings are
 * not part of the structure; they live in {@link Corpus} for ground truth and
 * optional exact re-scoring.
 */
public final class TurboGraph {

    static final String[] RELS = {"MENTIONS", "PART_OF", "NEXT", "SIMILAR"};
    static final int MENTIONS = 0;
    static final int PART_OF = 1;
    static final int NEXT = 2;
    static final int SIMILAR = 3;

    enum Kind { CHUNK, TERM, DOCUMENT }

    final TwoBitCodec codec;
    final int chunks;
    final int terms;
    final int docs;
    final Kind[] kind;
    final String[] label;
    final Code[] code;
    final int[][] adj;
    final byte[][] rel;
    /** Inverse document frequency of term nodes (0 for other kinds). */
    final float[] idf;
    final Map<String, Integer> termNode;
    final int entry;
    final float[][] relVec;
    final long buildMs;
    /** The corpus's entity extractor, applied to queries. */
    final java.util.function.Function<String, Set<String>> queryEntities;

    private TurboGraph(TwoBitCodec codec, int chunks, int terms, int docs, Kind[] kind, String[] label, Code[] code,
                       int[][] adj, byte[][] rel, float[] idf, Map<String, Integer> termNode, int entry,
                       float[][] relVec, long buildMs, java.util.function.Function<String, Set<String>> queryEntities) {
        this.codec = codec;
        this.chunks = chunks;
        this.terms = terms;
        this.docs = docs;
        this.kind = kind;
        this.label = label;
        this.code = code;
        this.adj = adj;
        this.rel = rel;
        this.idf = idf;
        this.termNode = termNode;
        this.entry = entry;
        this.relVec = relVec;
        this.buildMs = buildMs;
        this.queryEntities = queryEntities;
    }

    int size() {
        return kind.length;
    }

    long edges() {
        long e = 0;
        for (int[] a : adj) {
            e += a.length;
        }
        return e;
    }

    /** Bytes of the structure: 2-bit codes (+4-byte gain) and edges (int target + byte type). */
    long bytes() {
        return (long) size() * (codec.paddedDim() / 4 + 4) + edges() * 5;
    }

    static TurboGraph build(Corpus c, int m, int mMax, int efBuild) {
        long t0 = System.currentTimeMillis();
        TwoBitCodec codec = new TwoBitCodec(c.emb[0].length, 0x5EEDL);
        int nC = c.n;
        List<Set<String>> chunkTerms = new ArrayList<>(nC);
        Map<String, Integer> df = new HashMap<>();
        for (int i = 0; i < nC; i++) {
            Set<String> ts = c.termsOf(i);
            chunkTerms.add(ts);
            ts.forEach(t -> df.merge(t, 1, Integer::sum));
        }
        int maxDf = Math.max(3, nC / 50);
        List<String> kept = df.entrySet().stream()
                .filter(e -> e.getValue() >= 2 && e.getValue() <= maxDf)
                .sorted((a, b) -> Integer.compare(b.getValue(), a.getValue()))
                .limit(20_000).map(Map.Entry::getKey).sorted().toList();
        Map<String, Integer> termIdx = new HashMap<>();
        for (String t : kept) {
            termIdx.put(t, termIdx.size());
        }
        int nT = kept.size();
        int nD = c.docTitle.length;
        int n = nC + nT + nD;
        int D = codec.paddedDim();
        Kind[] kind = new Kind[n];
        String[] label = new String[n];
        Code[] code = new Code[n];
        float[] idf = new float[n];
        List<List<int[]>> edges = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            edges.add(new ArrayList<>());
        }
        float[][] rotated = new float[nC][];
        IntStream.range(0, nC).parallel().forEach(i -> {
            rotated[i] = codec.rotate(c.emb[i]);
            code[i] = codec.encodeRotated(rotated[i]);
        });
        float[][] termSum = new float[nT][];
        float[][] docSum = new float[nD][];
        for (int i = 0; i < nC; i++) {
            kind[i] = Kind.CHUNK;
            label[i] = c.label(i);
            int d = nC + nT + c.doc[i];
            link(edges, i, d, PART_OF);
            docSum[c.doc[i]] = add(docSum[c.doc[i]], rotated[i], D);
            if (i > 0 && c.doc[i - 1] == c.doc[i]) {
                link(edges, i - 1, i, NEXT);
            }
            for (String t : chunkTerms.get(i)) {
                Integer ti = termIdx.get(t);
                if (ti != null) {
                    link(edges, i, nC + ti, MENTIONS);
                    termSum[ti] = add(termSum[ti], rotated[i], D);
                }
            }
        }
        Map<String, Integer> termNode = new HashMap<>();
        for (int t = 0; t < nT; t++) {
            int id = nC + t;
            kind[id] = Kind.TERM;
            label[id] = kept.get(t);
            code[id] = codec.encodeRotated(termSum[t]);
            idf[id] = (float) Math.log((double) nC / df.get(kept.get(t)));
            termNode.put(kept.get(t), id);
        }
        for (int d = 0; d < nD; d++) {
            int id = nC + nT + d;
            kind[id] = Kind.DOCUMENT;
            label[id] = c.docTitle[d];
            code[id] = codec.encodeRotated(docSum[d] == null ? new float[D] : docSum[d]);
        }

        // SIMILAR edges: navigable small world built with 2-bit beam search. The first
        // 1000 chunks are inserted one by one; after that, batches of 256 search the
        // graph in parallel (read-only) and are linked in sequence.
        int[] order = IntStream.range(0, nC).toArray();
        SplittableRandom rnd = new SplittableRandom(42);
        for (int i = nC - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        int[][] sim = new int[nC][];
        int[] simLen = new int[nC];
        int entry = order[0];
        sim[entry] = new int[mMax + 1];
        int k = 1;
        while (k < nC) {
            int batch = k < 1000 ? 1 : Math.min(256, nC - k);
            final int from = k;
            int[][] found = new int[batch][];
            IntStream.range(0, batch).parallel().forEach(b -> {
                int v = order[from + b];
                found[b] = beam(codec, codec.decodeRotated(code[v]), entry, code, sim, simLen, efBuild, m);
            });
            for (int b = 0; b < batch; b++) {
                int v = order[from + b];
                sim[v] = new int[mMax + 1];
                for (int u : found[b]) {
                    sim[v][simLen[v]++] = u;
                    sim[u][simLen[u]++] = v;
                    if (simLen[u] > mMax) {
                        prune(codec, u, code, sim, simLen, mMax);
                    }
                }
            }
            k += batch;
        }
        for (int i = 0; i < nC; i++) {
            for (int j = 0; j < simLen[i]; j++) {
                edges.get(i).add(new int[] {sim[i][j], SIMILAR});
            }
        }
        int[][] adj = new int[n][];
        byte[][] rel = new byte[n][];
        for (int i = 0; i < n; i++) {
            List<int[]> es = edges.get(i);
            adj[i] = new int[es.size()];
            rel[i] = new byte[es.size()];
            for (int j = 0; j < es.size(); j++) {
                adj[i][j] = es.get(j)[0];
                rel[i][j] = (byte) es.get(j)[1];
            }
        }
        // TransE relation vectors r = mean(dst - src) over knowledge edges.
        float[][] relVec = new float[RELS.length][D];
        int[] relCount = new int[RELS.length];
        for (int i = 0; i < nC; i += 4) {
            for (int j = 0; j < adj[i].length; j++) {
                int r = rel[i][j];
                if (r == SIMILAR || adj[i][j] < i) {
                    continue;
                }
                float[] y = codec.decodeRotated(code[adj[i][j]]);
                float[] x = codec.decodeRotated(code[i]);
                for (int d = 0; d < D; d++) {
                    relVec[r][d] += y[d] - x[d];
                }
                relCount[r]++;
            }
        }
        for (int r = 0; r < RELS.length; r++) {
            for (int d = 0; d < D && relCount[r] > 0; d++) {
                relVec[r][d] /= relCount[r];
            }
        }
        return new TurboGraph(codec, nC, nT, nD, kind, label, code, adj, rel, idf, termNode, entry, relVec,
                System.currentTimeMillis() - t0, c::queryEntities);
    }

    static int[] beam(TwoBitCodec codec, float[] q, int entry, Code[] code, int[][] sim, int[] simLen,
                              int ef, int m) {
        BitSet seen = new BitSet();
        PriorityQueue<float[]> cand = new PriorityQueue<>((a, b) -> Float.compare(b[0], a[0]));
        PriorityQueue<float[]> best = new PriorityQueue<>((a, b) -> Float.compare(a[0], b[0]));
        float s0 = codec.score(q, code[entry]);
        cand.add(new float[] {s0, entry});
        best.add(new float[] {s0, entry});
        seen.set(entry);
        while (!cand.isEmpty()) {
            float[] c = cand.poll();
            if (best.size() >= ef && c[0] < best.peek()[0]) {
                break;
            }
            int u = (int) c[1];
            int[] nb = sim[u];
            int len = simLen[u];
            for (int j = 0; j < len && nb != null; j++) {
                int v = nb[j];
                if (seen.get(v)) {
                    continue;
                }
                seen.set(v);
                float s = codec.score(q, code[v]);
                if (best.size() < ef || s > best.peek()[0]) {
                    cand.add(new float[] {s, v});
                    best.add(new float[] {s, v});
                    if (best.size() > ef) {
                        best.poll();
                    }
                }
            }
        }
        List<float[]> all = new ArrayList<>(best);
        all.sort((a, b) -> Float.compare(b[0], a[0]));
        int[] out = new int[Math.min(m, all.size())];
        for (int i = 0; i < out.length; i++) {
            out[i] = (int) all.get(i)[1];
        }
        return out;
    }

    static void prune(TwoBitCodec codec, int u, Code[] code, int[][] sim, int[] simLen, int mMax) {
        float[] qu = codec.decodeRotated(code[u]);
        Integer[] idx = new Integer[simLen[u]];
        float[] sc = new float[simLen[u]];
        for (int j = 0; j < simLen[u]; j++) {
            idx[j] = j;
            sc[j] = codec.score(qu, code[sim[u][j]]);
        }
        java.util.Arrays.sort(idx, (a, b) -> Float.compare(sc[b], sc[a]));
        int[] keep = new int[mMax + 1];
        for (int j = 0; j < mMax; j++) {
            keep[j] = sim[u][idx[j]];
        }
        sim[u] = keep;
        simLen[u] = mMax;
    }

    private static void link(List<List<int[]>> edges, int a, int b, int rel) {
        edges.get(a).add(new int[] {b, rel});
        edges.get(b).add(new int[] {a, rel});
    }

    private static float[] add(float[] acc, float[] x, int d) {
        if (acc == null) {
            acc = new float[d];
        }
        for (int i = 0; i < d; i++) {
            acc[i] += x[i];
        }
        return acc;
    }
}
